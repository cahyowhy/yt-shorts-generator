"""Video segment extraction using FFmpeg."""

import asyncio
import uuid
from pathlib import Path
from typing import Optional

from loguru import logger

from shortgen.config import settings
from shortgen.core.exceptions import ProcessingError


class VideoClipper:
    """Extract video segments using FFmpeg."""

    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or settings.data_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def extract(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Extract a segment from video.

        Args:
            video_path: Source video path
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Optional output path (auto-generated if not provided)

        Returns:
            Path to extracted clip
        """
        if output_path is None:
            clip_id = uuid.uuid4().hex[:8]
            output_path = self.temp_dir / f"clip_{clip_id}.mp4"

        duration = end_time - start_time

        logger.debug(f"Extracting clip: {start_time:.1f}s - {end_time:.1f}s")

        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-ss", str(start_time),  # Seek position (before input for fast seek)
            "-i", str(video_path),
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(output_path),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise ProcessingError(f"FFmpeg failed: {error_msg}")

            if not output_path.exists():
                raise ProcessingError(f"Output file not created: {output_path}")

            logger.debug(f"Extracted clip: {output_path}")
            return output_path

        except FileNotFoundError:
            raise ProcessingError(
                "FFmpeg not found. Please install FFmpeg: "
                "https://ffmpeg.org/download.html"
            )

    async def extract_audio(
        self,
        video_path: str,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Extract audio track from video."""
        if output_path is None:
            audio_id = uuid.uuid4().hex[:8]
            output_path = self.temp_dir / f"audio_{audio_id}.wav"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",
            "-ar", "16000",  # 16kHz for Whisper
            "-ac", "1",  # Mono
            str(output_path),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()

            if process.returncode != 0 or not output_path.exists():
                raise ProcessingError("Failed to extract audio")

            return output_path

        except FileNotFoundError:
            raise ProcessingError("FFmpeg not found")

    def cleanup_temp_files(self) -> int:
        """Remove temporary files."""
        count = 0
        for file in self.temp_dir.glob("clip_*.mp4"):
            file.unlink()
            count += 1
        for file in self.temp_dir.glob("audio_*.wav"):
            file.unlink()
            count += 1
        logger.info(f"Cleaned up {count} temporary files")
        return count
