"""Face tracking using MediaPipe."""

import asyncio
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from loguru import logger
import math

from shortgen.core.exceptions import AnalysisError
from shortgen.core.models import FacePosition


class FaceTracker:
    """Track faces in video for smart cropping."""

    def __init__(
        self,
        sample_rate: int = 5,  # Sample every N frames
        min_detection_confidence: float = 0.5,
        model_path: str = "models/face_landmarker.task", 
    ):
        self.sample_rate = sample_rate
        self.min_detection_confidence = min_detection_confidence
        self.model_path = model_path
        
        self.history_length = 10 

    async def track(self, video_path: str) -> list[FacePosition]:
        """
        Track face positions throughout video and focus on the active speaker.
        """
        path = Path(video_path)
        if not path.exists():
            raise AnalysisError(f"Video file not found: {video_path}")

        logger.info(f"Tracking faces and identifying speaker: {path.name}")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._track_sync,
            str(path),
        )

        return result

    def _calculate_mar(self, landmarks, width: int, height: int) -> float:
        """Calculate Mouth Aspect Ratio (MAR) to detect speaking."""
        # MediaPipe inner lip indices: Top(13), Bottom(14), Left Corner(78), Right Corner(308)
        top = landmarks[13]
        bottom = landmarks[14]
        left = landmarks[78]
        right = landmarks[308]

        # Convert normalized coordinates to absolute pixels for accurate distance
        v_dist = math.hypot((top.x - bottom.x) * width, (top.y - bottom.y) * height)
        h_dist = math.hypot((left.x - right.x) * width, (left.y - right.y) * height)

        if h_dist == 0:
            return 0.0
        return v_dist / h_dist

    def _track_sync(self, video_path: str) -> list[FacePosition]:
        """Synchronous face tracking with speaker detection."""
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"MediaPipe model missing at {self.model_path}. "
                    "Please download 'face_landmarker.task' from Google MediaPipe."
                )

            # Initialize Landmarker
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=3,
                min_face_detection_confidence=self.min_detection_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False
            )
            landmarker = vision.FaceLandmarker.create_from_options(options)

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            face_positions: list[FacePosition] = []
            frame_number = 0
            
            tracked_faces = {} 
            next_face_id = 0

            # === NEW: Sticky Focus Variables ===
            active_face_id = None
            frames_since_switch = 0
            missing_frames = 0  # <--- Moved to a standalone variable
            cooldown_frames = int(fps * 1.5)

            with tqdm(total=total_frames, desc="Tracking Faces", unit="fr") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_number % self.sample_rate == 0:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                        results = landmarker.detect(mp_image)
                        timestamp = frame_number / fps
                        current_frame_faces = []

                        if results.face_landmarks:
                            # ... [Coordinate and MAR math stays the same] ...
                            for i, landmarks in enumerate(results.face_landmarks):
                                x_coords = [lm.x for lm in landmarks]
                                y_coords = [lm.y for lm in landmarks]
                                min_x, max_x = min(x_coords), max(x_coords)
                                min_y, max_y = min(y_coords), max(y_coords)
                                width, height = max_x - min_x, max_y - min_y
                                cx, cy = min_x + (width / 2), min_y + (height / 2)
                                mar = self._calculate_mar(landmarks, frame_width, frame_height)
                                
                                current_frame_faces.append({
                                    'cx': cx, 'cy': cy, 'width': width, 'height': height,
                                    'mar': mar, 'confidence': 0.9
                                })

                            for face in current_frame_faces:
                                best_id = None
                                min_dist = float('inf')
                                for fid, data in tracked_faces.items():
                                    dist = math.hypot(face['cx'] - data['cx'], face['cy'] - data['cy'])
                                    if dist < 0.15 and dist < min_dist:
                                        min_dist = dist
                                        best_id = fid

                                if best_id is not None:
                                    tracked_faces[best_id].update({'cx': face['cx'], 'cy': face['cy']})
                                    tracked_faces[best_id]['mar_history'].append(face['mar'])
                                    if len(tracked_faces[best_id]['mar_history']) > self.history_length:
                                        tracked_faces[best_id]['mar_history'].pop(0)
                                    face['id'] = best_id
                                else:
                                    face['id'] = next_face_id
                                    tracked_faces[next_face_id] = {
                                        'cx': face['cx'], 'cy': face['cy'], 'mar_history': [face['mar']]
                                    }
                                    next_face_id += 1

                            # === FIXED: Grace Period Logic ===
                            face_scores = {}
                            for face in current_frame_faces:
                                history = tracked_faces[face['id']]['mar_history']
                                face_scores[face['id']] = max(history) - min(history) if len(history) > 1 else 0.0

                            best_face = None
                            active_face_present = any(f['id'] == active_face_id for f in current_frame_faces)

                            if not active_face_present and active_face_id is not None:
                                missing_frames += self.sample_rate
                                if missing_frames > int(fps * 1.0):
                                    active_face_id = None
                                    missing_frames = 0
                            else:
                                missing_frames = 0

                            if active_face_id is None and current_frame_faces:
                                best_face = max(current_frame_faces, key=lambda f: face_scores[f['id']])
                                if face_scores[best_face['id']] < 0.02:
                                    best_face = max(current_frame_faces, key=lambda f: f['width'] * f['height'])
                                active_face_id = best_face['id']
                                frames_since_switch = 0
                            elif current_frame_faces:
                                current_score = face_scores.get(active_face_id, 0.0)
                                challengers = [f for f in current_frame_faces if f['id'] != active_face_id]
                                
                                if challengers:
                                    top_challenger = max(challengers, key=lambda f: face_scores[f['id']])
                                    challenger_score = face_scores[top_challenger['id']]
                                    
                                    if challenger_score > (current_score + 0.015) and frames_since_switch > cooldown_frames:
                                        active_face_id = top_challenger['id']
                                        best_face = top_challenger
                                        frames_since_switch = 0
                                    else:
                                        active_faces = [f for f in current_frame_faces if f['id'] == active_face_id]
                                        best_face = active_faces[0] if active_faces else None
                                else:
                                    active_faces = [f for f in current_frame_faces if f['id'] == active_face_id]
                                    best_face = active_faces[0] if active_faces else None

                            frames_since_switch += self.sample_rate

                            if best_face:
                                face_positions.append(
                                    FacePosition(
                                        frame_number=frame_number, timestamp=timestamp,
                                        center_x=best_face['cx'], center_y=best_face['cy'],
                                        width=best_face['width'], height=best_face['height'],
                                        confidence=best_face['confidence'],
                                    )
                                )
                        else:
                            # Add to missing frames if no faces detected at all but we had an active target
                            if active_face_id is not None:
                                missing_frames += self.sample_rate
                                if missing_frames > int(fps * 1.0):
                                    active_face_id = None
                                    missing_frames = 0
                                    
                            face_positions.append(
                                FacePosition(
                                    frame_number=frame_number, timestamp=timestamp,
                                    center_x=0.5, center_y=0.5, width=0.0, height=0.0, confidence=0.0,
                                )
                            )

                    frame_number += 1
                    pbar.update(1)

                    # Log every 10% instead of every 30 seconds
                    if frame_number % max(1, (total_frames // 10)) == 0:
                        progress = (frame_number / total_frames) * 100
                        logger.info(f"Tracking Progress: {progress:.0f}%")

            cap.release()
            landmarker.close()
            return face_positions

        except Exception as e:
            logger.warning(f"MediaPipe failed: {e}. Falling back.")
            return self._fallback_tracking(video_path)

    def _fallback_tracking(self, video_path: str) -> list[FacePosition]:
        """Fallback face detection using OpenCV Haar cascades."""
        try:
            import cv2

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            face_positions: list[FacePosition] = []
            frame_number = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_number % self.sample_rate == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    height, width = gray.shape

                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )

                    timestamp = frame_number / fps

                    if len(faces) > 0:
                        x, y, w, h = faces[0]  # Use first face
                        face_positions.append(
                            FacePosition(
                                frame_number=frame_number,
                                timestamp=timestamp,
                                center_x=(x + w / 2) / width,
                                center_y=(y + h / 2) / height,
                                width=w / width,
                                height=h / height,
                                confidence=0.7,  # Haar doesn't give confidence
                            )
                        )
                    else:
                        face_positions.append(
                            FacePosition(
                                frame_number=frame_number,
                                timestamp=timestamp,
                                center_x=0.5,
                                center_y=0.5,
                                width=0.0,
                                height=0.0,
                                confidence=0.0,
                            )
                        )

                frame_number += 1

            cap.release()
            return face_positions

        except Exception as e:
            logger.error(f"Fallback face tracking failed: {e}")
            return []

    def smooth_positions(
        self,
        positions: list[FacePosition],
        window_size: int = 5,
    ) -> list[FacePosition]:
        """Apply smoothing to face positions to reduce jitter."""
        if len(positions) < window_size:
            return positions

        smoothed = []
        half_window = window_size // 2

        for i, pos in enumerate(positions):
            # Get window of positions
            start = max(0, i - half_window)
            end = min(len(positions), i + half_window + 1)
            window = positions[start:end]

            # Calculate weighted average (center-weighted)
            total_weight = 0.0
            avg_x = 0.0
            avg_y = 0.0

            for j, w_pos in enumerate(window):
                if w_pos.confidence > 0:
                    # Weight by confidence and distance from center
                    distance_weight = 1.0 - abs(j - half_window) / (half_window + 1)
                    weight = w_pos.confidence * distance_weight
                    avg_x += w_pos.center_x * weight
                    avg_y += w_pos.center_y * weight
                    total_weight += weight

            if total_weight > 0:
                smoothed.append(
                    FacePosition(
                        frame_number=pos.frame_number,
                        timestamp=pos.timestamp,
                        center_x=avg_x / total_weight,
                        center_y=avg_y / total_weight,
                        width=pos.width,
                        height=pos.height,
                        confidence=pos.confidence,
                    )
                )
            else:
                smoothed.append(pos)

        return smoothed