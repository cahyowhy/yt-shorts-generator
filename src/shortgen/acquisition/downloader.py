"""Video downloader using yt-dlp."""

import asyncio
import re
from pathlib import Path
from typing import Optional

import yt_dlp
from loguru import logger

from shortgen.config import settings
from shortgen.core.exceptions import DownloadError
from shortgen.core.models import VideoMetadata


class VideoDownloader:
    """Download videos from YouTube using yt-dlp."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or settings.data_dir / "downloads"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        patterns = [
            r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})",
            r"youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        raise DownloadError(f"Could not extract video ID from URL: {url}")

    def _get_ydl_opts(self, video_id: str) -> dict:
        """Get yt-dlp options."""
        return {
            "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
            "outtmpl": str(self.output_dir / f"{video_id}.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
        }

    async def download(self, url: str) -> VideoMetadata:
        """
        Download a video from YouTube.

        Args:
            url: YouTube video URL

        Returns:
            VideoMetadata with file path and video information
        """
        video_id = self._extract_video_id(url)
        logger.info(f"Downloading video: {video_id}")

        # Run yt-dlp in executor to not block async loop
        loop = asyncio.get_event_loop()
        metadata = await loop.run_in_executor(
            None,
            self._download_sync,
            url,
            video_id,
        )

        return metadata

    def _download_sync(self, url: str, video_id: str) -> VideoMetadata:
        """Synchronous download implementation."""
        ydl_opts = self._get_ydl_opts(video_id)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first
                info = ydl.extract_info(url, download=False)

                if info is None:
                    raise DownloadError(f"Could not extract video info: {url}")

                # Check if already downloaded
                expected_path = self.output_dir / f"{video_id}.mp4"
                if not expected_path.exists():
                    # Download the video
                    ydl.download([url])

                # Find the downloaded file
                file_path = self._find_downloaded_file(video_id)

                if file_path is None:
                    raise DownloadError(f"Downloaded file not found for: {video_id}")

                return VideoMetadata(
                    url=url,
                    video_id=video_id,
                    title=info.get("title", "Unknown"),
                    duration=float(info.get("duration", 0)),
                    width=info.get("width", 1920),
                    height=info.get("height", 1080),
                    fps=float(info.get("fps", 30)),
                    file_path=str(file_path),
                )

        except yt_dlp.utils.DownloadError as e:
            raise DownloadError(f"Failed to download video: {e}") from e
        except Exception as e:
            raise DownloadError(f"Unexpected error during download: {e}") from e

    def _find_downloaded_file(self, video_id: str) -> Optional[Path]:
        """Find the downloaded video file."""
        # Check common extensions
        for ext in ["mp4", "webm", "mkv"]:
            path = self.output_dir / f"{video_id}.{ext}"
            if path.exists():
                return path

        # Fallback: search for any file starting with video_id
        for path in self.output_dir.glob(f"{video_id}.*"):
            if path.suffix.lower() in [".mp4", ".webm", ".mkv"]:
                return path

        return None

    async def get_info(self, url: str) -> dict:
        """Get video information without downloading."""
        loop = asyncio.get_event_loop()

        def _get_info():
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": False,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=False)

        return await loop.run_in_executor(None, _get_info)
