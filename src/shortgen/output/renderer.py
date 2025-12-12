"""Final video rendering using FFmpeg."""

import asyncio
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger

from shortgen.config import settings
from shortgen.core.exceptions import RenderError
from shortgen.core.models import CropWindow, Platform
from shortgen.processing.captioner import Caption, Captioner


class VideoRenderer:
    """Render final short videos with cropping and captions."""

    def __init__(self):
        self.captioner = Captioner()

    async def render(
        self,
        input_path: Path,
        output_path: Path,
        crop_windows: list[CropWindow],
        captions: list[Caption],
        platform: Platform = Platform.YOUTUBE_SHORTS,
    ) -> Path:
        """
        Render final video with dynamic cropping and captions.

        Args:
            input_path: Path to input video clip
            output_path: Path for output video
            crop_windows: List of crop windows for dynamic cropping
            captions: List of captions to burn in
            platform: Target platform for format settings

        Returns:
            Path to rendered video
        """
        logger.info(f"Rendering: {output_path.name}")

        # Generate caption file
        caption_file = None
        if captions:
            caption_file = await self._write_caption_file(captions)

        try:
            # For now, use static crop (first window)
            # Dynamic cropping requires more complex FFmpeg filter chains
            if crop_windows:
                crop = crop_windows[0]
            else:
                crop = CropWindow(
                    timestamp=0,
                    x_offset=0,
                    y_offset=0,
                    width=settings.output_resolution_width,
                    height=settings.output_resolution_height,
                )

            await self._render_with_ffmpeg(
                input_path=input_path,
                output_path=output_path,
                crop=crop,
                caption_file=caption_file,
                platform=platform,
            )

            return output_path

        finally:
            # Cleanup caption file
            if caption_file and Path(caption_file).exists():
                Path(caption_file).unlink()

    async def _write_caption_file(self, captions: list[Caption]) -> str:
        """Write captions to temporary ASS file."""
        ass_content = self.captioner.to_ass(captions)

        # Create temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".ass",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(ass_content)
            return f.name

    async def _render_with_ffmpeg(
        self,
        input_path: Path,
        output_path: Path,
        crop: CropWindow,
        caption_file: Optional[str],
        platform: Platform,
    ) -> None:
        """Execute FFmpeg render command."""

        # Build filter chain
        filters = []

        # Crop filter
        filters.append(
            f"crop={crop.width}:{crop.height}:{crop.x_offset}:{crop.y_offset}"
        )

        # Scale to output resolution
        filters.append(
            f"scale={settings.output_resolution_width}:{settings.output_resolution_height}"
        )

        # Add captions if available
        if caption_file:
            # Escape special characters in path
            escaped_path = caption_file.replace("\\", "/").replace(":", "\\:")
            filters.append(f"ass='{escaped_path}'")

        filter_chain = ",".join(filters)

        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-vf", filter_chain,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "192k",
            "-r", str(settings.output_fps),
            "-movflags", "+faststart",
        ]

        # Platform-specific settings
        if platform == Platform.YOUTUBE_SHORTS:
            cmd.extend(["-b:v", "8M", "-maxrate", "12M", "-bufsize", "16M"])
        elif platform == Platform.INSTAGRAM_REELS:
            cmd.extend(["-b:v", "6M", "-maxrate", "8M", "-bufsize", "12M"])
        elif platform == Platform.TIKTOK:
            cmd.extend(["-b:v", "6M", "-maxrate", "8M", "-bufsize", "12M"])

        cmd.append(str(output_path))

        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise RenderError(f"FFmpeg render failed: {error_msg}")

            if not output_path.exists():
                raise RenderError(f"Output file not created: {output_path}")

            logger.debug(f"Rendered successfully: {output_path}")

        except FileNotFoundError:
            raise RenderError(
                "FFmpeg not found. Please install FFmpeg: "
                "https://ffmpeg.org/download.html"
            )

    async def render_with_dynamic_crop(
        self,
        input_path: Path,
        output_path: Path,
        crop_windows: list[CropWindow],
        captions: list[Caption],
        fps: float,
    ) -> Path:
        """
        Advanced render with frame-by-frame dynamic cropping.

        This is more complex and slower but provides smooth panning.
        Uses FFmpeg's zoompan filter or frame-by-frame processing.
        """
        # TODO: Implement dynamic cropping
        # Options:
        # 1. Generate crop coordinates file and use sendcmd filter
        # 2. Use Python (moviepy/opencv) for frame-by-frame processing
        # 3. Use zoompan filter with keyframes

        logger.warning("Dynamic cropping not yet implemented, using static crop")
        return await self.render(
            input_path=input_path,
            output_path=output_path,
            crop_windows=crop_windows,
            captions=captions,
        )
