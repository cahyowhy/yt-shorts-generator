"""Smart vertical cropping that follows faces/speakers."""

from typing import Optional

import numpy as np
from loguru import logger

from shortgen.core.models import CropWindow, FacePosition


class SmartCropper:
    """
    Calculate dynamic crop windows for converting horizontal video to vertical.

    This is the key differentiator - instead of static center crop,
    we follow faces/speakers throughout the video.
    """

    def __init__(
        self,
        smoothing_window: int = 15,  # Frames to smooth over
        max_velocity: float = 0.02,  # Max crop movement per frame (normalized)
        default_position: float = 0.5,  # Default x position when no face
    ):
        self.smoothing_window = smoothing_window
        self.max_velocity = max_velocity
        self.default_position = default_position

    def calculate_crop_windows(
        self,
        source_resolution: tuple[int, int],
        target_aspect_ratio: tuple[int, int],  # e.g., (9, 16)
        face_positions: list[FacePosition],
        fps: float,
        duration: float,
    ) -> list[CropWindow]:
        """
        Calculate crop windows for each frame.

        Args:
            source_resolution: (width, height) of source video
            target_aspect_ratio: (width, height) ratio for output
            face_positions: Face tracking data
            fps: Video frame rate
            duration: Video duration in seconds

        Returns:
            List of CropWindow objects, one per frame
        """
        src_width, src_height = source_resolution
        target_w, target_h = target_aspect_ratio

        # Calculate crop dimensions maintaining target aspect ratio
        target_ratio = target_w / target_h
        source_ratio = src_width / src_height

        if source_ratio > target_ratio:
            # Source is wider - crop width
            crop_height = src_height
            crop_width = int(src_height * target_ratio)
        else:
            # Source is taller - crop height
            crop_width = src_width
            crop_height = int(src_width / target_ratio)

        # Maximum horizontal offset
        max_x_offset = src_width - crop_width

        logger.debug(
            f"Crop dimensions: {crop_width}x{crop_height} "
            f"from {src_width}x{src_height}"
        )

        # Build frame-by-frame target positions from face data
        total_frames = int(fps * duration)
        target_positions = self._interpolate_face_positions(
            face_positions, total_frames, fps, src_width, crop_width 
        )

        # Apply smoothing to reduce jitter
        smoothed_positions = self._smooth_positions(target_positions)

        # Apply velocity limiting for smooth motion
        final_positions = self._limit_velocity(smoothed_positions)

        # Convert to crop windows
        crop_windows = []
        for frame_num, x_pos in enumerate(final_positions):
            # Convert normalized position to pixel offset
            x_offset = int(x_pos * max_x_offset)
            x_offset = max(0, min(x_offset, max_x_offset))

            # Vertical offset - typically center or slight upper
            y_offset = max(0, (src_height - crop_height) // 3)  # Slight upper bias

            crop_windows.append(
                CropWindow(
                    timestamp=frame_num / fps,
                    x_offset=x_offset,
                    y_offset=y_offset,
                    width=crop_width,
                    height=crop_height,
                )
            )

        logger.info(f"Generated {len(crop_windows)} crop windows")
        return crop_windows

    def _interpolate_face_positions(
        self,
        face_positions: list[FacePosition],
        total_frames: int,
        fps: float,
        src_width: int = 1920,
        crop_width: int = 1080,
    ) -> list[float]:
        """Interpolate face positions using exact pixel centering math."""
        if not face_positions:
            return [self.default_position] * total_frames

        # Create time-indexed position map
        position_map: dict[int, float] = {}
        for fp in face_positions:
            frame = int(fp.timestamp * fps)
            if fp.confidence > 0.3:
                # Calculate the exact normalized crop position to center this face
                position_map[frame] = self._exact_face_to_crop(fp.center_x, src_width, crop_width)

        # Interpolate missing frames
        result = []
        last_position = self.default_position

        for frame in range(total_frames):
            if frame in position_map:
                last_position = position_map[frame]
            result.append(last_position)

        return result

    def _exact_face_to_crop(self, face_center_x: float, src_width: int, crop_width: int) -> float:
        """
        Calculates the exact normalized position [0.0 - 1.0] needed to perfectly 
        center the face in the cropped window.
        """
        # Where is the face in exact pixels on the original video?
        face_pixel_x = face_center_x * src_width
        
        # To center the face, the left edge of the crop must be half a crop-width to the left
        target_x_offset_px = face_pixel_x - (crop_width / 2)
        
        # Maximum possible sliding distance for the crop window
        max_x_offset = src_width - crop_width
        
        if max_x_offset <= 0:
            return 0.5 # Safety fallback
            
        # Convert the pixel offset back into a 0.0 to 1.0 ratio for the smoothing algorithm
        crop_pos = target_x_offset_px / max_x_offset
        
        # Clamp to edges so we don't crop outside the video bounds
        return max(0.0, min(1.0, crop_pos))

    def _smooth_positions(self, positions: list[float]) -> list[float]:
        """Apply moving average smoothing."""
        if len(positions) <= self.smoothing_window:
            return positions

        # Use convolution for efficient smoothing
        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        padded = np.pad(
            positions,
            (self.smoothing_window // 2, self.smoothing_window // 2),
            mode="edge",
        )
        smoothed = np.convolve(padded, kernel, mode="valid")

        return smoothed.tolist()

    def _limit_velocity(self, positions: list[float]) -> list[float]:
        """Limit frame-to-frame movement for smooth panning."""
        if not positions:
            return positions

        result = [positions[0]]

        for i in range(1, len(positions)):
            target = positions[i]
            current = result[-1]
            delta = target - current

            # Limit the change per frame
            if abs(delta) > self.max_velocity:
                delta = self.max_velocity if delta > 0 else -self.max_velocity

            result.append(current + delta)

        return result

    def get_static_crop(
        self,
        source_resolution: tuple[int, int],
        target_aspect_ratio: tuple[int, int],
        position: float = 0.5,
    ) -> CropWindow:
        """Get a static center crop (fallback mode)."""
        src_width, src_height = source_resolution
        target_w, target_h = target_aspect_ratio

        target_ratio = target_w / target_h

        if src_width / src_height > target_ratio:
            crop_height = src_height
            crop_width = int(src_height * target_ratio)
        else:
            crop_width = src_width
            crop_height = int(src_width / target_ratio)

        x_offset = int((src_width - crop_width) * position)
        y_offset = (src_height - crop_height) // 2

        return CropWindow(
            timestamp=0.0,
            x_offset=x_offset,
            y_offset=y_offset,
            width=crop_width,
            height=crop_height,
        )
