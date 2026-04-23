"""Face tracking using MediaPipe."""

import asyncio
from pathlib import Path
from typing import Optional

from loguru import logger

from shortgen.core.exceptions import AnalysisError
from shortgen.core.models import FacePosition


class FaceTracker:
    """Track faces in video for smart cropping."""

    def __init__(
        self,
        sample_rate: int = 5,  # Sample every N frames
        min_detection_confidence: float = 0.5,
        model_path: str = "models/blazer_face_short_range.tflite", # ADDED: Tasks API requires a model file
    ):
        self.sample_rate = sample_rate
        self.min_detection_confidence = min_detection_confidence
        self.model_path = model_path

    async def track(self, video_path: str) -> list[FacePosition]:
        """
        Track face positions throughout video.

        Args:
            video_path: Path to video file

        Returns:
            List of FacePosition objects
        """
        path = Path(video_path)
        if not path.exists():
            raise AnalysisError(f"Video file not found: {video_path}")

        logger.info(f"Tracking faces: {path.name}")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._track_sync,
            str(path),
        )

        return result

    def _track_sync(self, video_path: str) -> list[FacePosition]:
        """Synchronous face tracking implementation."""
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            # Check if model exists
            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"MediaPipe model missing at {self.model_path}. "
                    "Please download blazer_face_short_range.tflite."
                )

            # Initialize MediaPipe Tasks Face Detector
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=self.min_detection_confidence
            )
            detector = vision.FaceDetector.create_from_options(options)

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Get frame dimensions to calculate relative positions
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            face_positions: list[FacePosition] = []
            frame_number = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample every N frames
                if frame_number % self.sample_rate == 0:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to MediaPipe Image format
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    
                    # Process frame
                    results = detector.detect(mp_image)
                    timestamp = frame_number / fps

                    if results.detections:
                        # Use the first (most confident) detection
                        detection = results.detections[0]
                        bbox = detection.bounding_box

                        # Tasks API returns absolute pixels. Convert to relative (0.0 to 1.0)
                        rel_width = bbox.width / frame_width
                        rel_height = bbox.height / frame_height
                        rel_center_x = (bbox.origin_x / frame_width) + (rel_width / 2)
                        rel_center_y = (bbox.origin_y / frame_height) + (rel_height / 2)

                        # Tasks API stores the score inside categories
                        confidence = detection.categories[0].score

                        face_positions.append(
                            FacePosition(
                                frame_number=frame_number,
                                timestamp=timestamp,
                                center_x=rel_center_x,
                                center_y=rel_center_y,
                                width=rel_width,
                                height=rel_height,
                                confidence=confidence,
                            )
                        )
                    else:
                        # No face detected - use center as fallback
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

                # Progress logging
                if frame_number % (fps * 30) == 0:  # Log every 30 seconds
                    progress = frame_number / total_frames * 100
                    logger.debug(f"Face tracking progress: {progress:.1f}%")

            cap.release()
            detector.close() # Clean up the detector
            logger.info(f"Tracked faces in {len(face_positions)} frames")
            return face_positions

        except Exception as e:
            logger.warning(f"MediaPipe Tasks failed: {e}. Falling back to OpenCV.")
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