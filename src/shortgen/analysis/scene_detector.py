"""Scene change detection using PySceneDetect."""

import asyncio
from pathlib import Path

from loguru import logger

from shortgen.core.exceptions import AnalysisError


class SceneDetector:
    """Detect scene changes in video content."""

    def __init__(
        self,
        threshold: float = 27.0,
        min_scene_len: int = 15,  # frames
    ):
        self.threshold = threshold
        self.min_scene_len = min_scene_len

    async def detect(self, video_path: str) -> list[float]:
        """
        Detect scene changes in video.

        Args:
            video_path: Path to video file

        Returns:
            List of timestamps (seconds) where scene changes occur
        """
        path = Path(video_path)
        if not path.exists():
            raise AnalysisError(f"Video file not found: {video_path}")

        logger.info(f"Detecting scenes: {path.name}")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._detect_sync,
            str(path),
        )

        return result

    def _detect_sync(self, video_path: str) -> list[float]:
        """Synchronous scene detection implementation."""
        try:
            from scenedetect import SceneManager, open_video
            from scenedetect.detectors import ContentDetector

            # Open video
            video = open_video(video_path)

            # Create scene manager with content detector
            scene_manager = SceneManager()
            scene_manager.add_detector(
                ContentDetector(
                    threshold=self.threshold,
                    min_scene_len=self.min_scene_len,
                )
            )

            # Detect scenes
            scene_manager.detect_scenes(video)
            scene_list = scene_manager.get_scene_list()

            # Extract timestamps
            scene_times = []
            for scene in scene_list:
                start_time = scene[0].get_seconds()
                scene_times.append(start_time)

            logger.info(f"Detected {len(scene_times)} scene changes")
            return scene_times

        except ImportError:
            logger.warning("scenedetect not available, using fallback")
            return self._fallback_detection(video_path)
        except Exception as e:
            raise AnalysisError(f"Scene detection failed: {e}") from e

    def _fallback_detection(self, video_path: str) -> list[float]:
        """Fallback scene detection using OpenCV frame differencing."""
        try:
            import cv2
            import numpy as np

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            scene_times = []
            prev_frame = None
            frame_count = 0
            diff_threshold = 30.0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to grayscale and resize for faster processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (160, 90))

                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(gray, prev_frame)
                    mean_diff = np.mean(diff)

                    if mean_diff > diff_threshold:
                        timestamp = frame_count / fps
                        # Avoid detecting too close scenes
                        if not scene_times or (timestamp - scene_times[-1]) > 1.0:
                            scene_times.append(timestamp)

                prev_frame = gray
                frame_count += 1

            cap.release()
            logger.info(f"Fallback detection found {len(scene_times)} scene changes")
            return scene_times

        except Exception as e:
            logger.error(f"Fallback scene detection failed: {e}")
            return []
