"""Video downloader using yt-dlp."""

import asyncio
import re
import time
from pathlib import Path
from typing import Optional

import yt_dlp
from loguru import logger

from shortgen.config import settings
from shortgen.core.exceptions import DownloadError
from shortgen.core.models import VideoMetadata

from pathlib import Path

COOKIE_PATH = Path.cwd() / 'cookies.txt'

class VideoDownloader:
    """Download videos from YouTube using yt-dlp."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or settings.data_dir / "downloads"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.subtitles_dir = self.output_dir / "subtitles"
        self.subtitles_dir.mkdir(parents=True, exist_ok=True)

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
        """Get yt-dlp options for video download."""
        return {
            "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
            "outtmpl": str(self.output_dir / f"{video_id}.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "socket_timeout": 30,
            '--cookies': COOKIE_PATH,
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            },
        }

    def _get_subtitle_download_opts(self, video_id: str, lang: str, allow_auto: bool = False) -> dict:
        """Get yt-dlp options for subtitle download with rate limit handling."""
        return {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": allow_auto,
            "subtitleslangs": [lang],   # correct yt-dlp key (was "subtitles", a no-op)
            "subtitlesformat": "srt",
            "outtmpl": str(self.subtitles_dir / f"{video_id}"),
            "socket_timeout": 30,
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            },
            '--cookies': COOKIE_PATH,
            "extractor_args": {
                "youtube": {
                    "skip_webpage": True,
                }
            },
        }

    async def download(self, url: str, overide_lang: Optional[str] = None) -> VideoMetadata:
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
            overide_lang
        )

        return metadata

    def _download_sync(self, url: str, video_id: str, overide_lang: Optional[str]) -> VideoMetadata:
        """Synchronous download implementation."""
        try:
            # Step 1: Extract info to detect language
            logger.info("Extracting video info...")
            info_opts = {
                "quiet": True,
                "--user-agent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36', 
                '--cookies': COOKIE_PATH,
                "no_warnings": True,
                "extract_flat": False,
                "skip_download": True,
                "socket_timeout": 30,
            }
            
            with yt_dlp.YoutubeDL(info_opts) as ydl:
                info = ydl.extract_info(url, download=False)

            if info is None:
                raise DownloadError(f"Could not extract video info: {url}")

            logger.debug(f"Subtitles in info: {list(info.get('subtitles', {}).keys())}")
            logger.debug(f"Auto captions in info: {list(info.get('automatic_captions', {}).keys())}")
            
            avail_langs = list(info.get('subtitles', {}).keys()) + list(info.get('automatic_captions', {}).keys())
            final_lang = ""
            matched_lang = None
            if overide_lang:
                matched_lang = next((l for l in avail_langs if l.startswith(overide_lang)), None)

            if matched_lang:
                final_lang = matched_lang
            else:
                final_lang = self._detect_original_language(info)    

            logger.info(f"Using video language: {final_lang}")
            
            # Step 2: Download subtitles (with delay to avoid rate limiting)
            subtitle_path = None
            try:
                logger.info("Waiting before subtitle download (rate limit prevention)...")
                time.sleep(5)  # Wait 5 seconds before downloading subtitles
                subtitle_path = self._download_subtitles(url, video_id, final_lang)
            except Exception as e:
                logger.warning(f"Subtitle download failed (will use Whisper): {e}")

            # Step 3: Download video file
            logger.info("Downloading video file...")
            ydl_opts = self._get_ydl_opts(video_id)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                expected_path = self.output_dir / f"{video_id}.mp4"
                if not expected_path.exists():
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
                original_lang=final_lang,
                subtitle_path=str(subtitle_path) if subtitle_path else None,
            )

        except yt_dlp.utils.DownloadError as e:
            raise DownloadError(f"Failed to download video: {e}") from e
        except Exception as e:
            raise DownloadError(f"Unexpected error during download: {e}") from e

    def _detect_original_language(self, info: dict) -> str:
        """
        Detect the original language of the video.
        
        Priority:
        1. Check for '-orig' suffix
        2. Check available captions
        
        Args:
            info: Video info dict from yt-dlp
            
        Returns:
            Language code (e.g., 'en', 'id', 'ko', 'ko-orig', 'id-orig')
        """
        auto_captions = info.get("automatic_captions", {})
        subtitles = info.get("subtitles", {})
        
        logger.debug(f"Auto captions available: {list(auto_captions.keys())}")
        logger.debug(f"Subtitles available: {list(subtitles.keys())}")
        
        # Priority 1: Look for '-orig' suffix (indicates original language)
        for lang in auto_captions.keys():
            if lang.endswith("-orig"):
                logger.info(f"Found original language marker: {lang}")
                return lang
        
        for lang in subtitles.keys():
            if lang.endswith("-orig"):
                logger.info(f"Found original subtitle language marker: {lang}")
                return lang
        
        # Priority 2: Check first available
        if auto_captions:
            first_lang = next(iter(auto_captions.keys()))
            logger.info(f"Using first available auto-caption language: {first_lang}")
            return first_lang
        
        if subtitles:
            first_lang = next(iter(subtitles.keys()))
            logger.info(f"Using first available subtitle language: {first_lang}")
            return first_lang
        
        logger.warning("Could not detect video language")
        return None

    def _download_subtitles(self, url: str, video_id: str, original_lang: str) -> Optional[Path]:
        """
        Download subtitles in priority order:
        1. Original language
        2. Indonesian (id)
        3. English (en)
        
        Args:
            url: YouTube video URL
            video_id: Video ID
            original_lang: Detected original language
            
        Returns:
            Path to first successfully downloaded subtitle file
        """
        # Build priority list: original first, then id, then en
        langs_to_try = []
        
        if original_lang:
            langs_to_try.append(original_lang)
        
        if original_lang != "id":
            langs_to_try.append("id")
        
        if original_lang != "en":
            langs_to_try.append("en")
            
        if original_lang != "ko":
            langs_to_try.append("ko")
        
        logger.info(f"Attempting to download subtitles for languages: {langs_to_try}")

        for i, lang in enumerate(langs_to_try):
            logger.info(f"Trying to download subtitles for: {lang} ({i+1}/{len(langs_to_try)})")
            result = self._download_lang_subtitles(url, video_id, lang)
            if result:
                return result
            
            # Longer pause between languages to avoid triggering rate limits
            if i < len(langs_to_try) - 1:
                logger.debug(f"Waiting 10 seconds before next subtitle language attempt...")
                time.sleep(10)

        logger.warning(f"Could not download subtitles for any language. Will use Whisper.")
        return None

    def _download_lang_subtitles(self, url: str, video_id: str, lang: str) -> Optional[Path]:
        """
        Download subtitles for a specific language with retry logic.

        Tries manual subtitles first, then falls back to auto-generated captions.
        Uses exponential backoff with jitter on 429 responses.

        Args:
            url: YouTube video URL
            video_id: Video ID
            lang: Language code to download

        Returns:
            Path to downloaded subtitle file or None
        """
        import random

        max_retries = 3
        # Try manual subs first (allow_auto=False), then auto-generated (allow_auto=True)
        sub_modes = [(False, "manual"), (True, "auto-generated")]

        for allow_auto, mode_name in sub_modes:
            logger.info(f"Trying {mode_name} subtitles for '{lang}'")
            for attempt in range(max_retries + 1):
                try:
                    ydl_opts = self._get_subtitle_download_opts(video_id, lang, allow_auto=allow_auto)
                    logger.info(
                        f"Downloading {mode_name} subtitles for '{lang}' "
                        f"(attempt {attempt + 1}/{max_retries + 1})"
                    )

                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([url])

                    subtitle_file = self._find_subtitle_file(video_id, lang)
                    if subtitle_file:
                        if subtitle_file.suffix == ".srt":
                            subtitle_file = self._fix_srt_overlaps(subtitle_file)
                        logger.info(f"Successfully downloaded {mode_name} subtitles for '{lang}': {subtitle_file}")
                        return subtitle_file

                    # No file written — subtitles simply don't exist in this mode
                    logger.debug(f"No {mode_name} subtitle file produced for '{lang}'")
                    break  # Don't retry; move to next mode

                except yt_dlp.utils.DownloadError as e:
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        if attempt < max_retries:
                            # Exponential backoff: 30s, 60s, 120s + up to 15s jitter
                            wait_time = (30 * (2 ** attempt)) + random.uniform(0, 15)
                            logger.warning(
                                f"Rate limited (429) on {mode_name} '{lang}' "
                                f"attempt {attempt + 1}/{max_retries + 1}. "
                                f"Retrying in {wait_time:.0f}s..."
                            )
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Max retries reached for {mode_name} '{lang}' after 429 errors")
                            break  # Give up this mode, try next
                    else:
                        logger.debug(f"Download error for {mode_name} '{lang}': {e}")
                        break  # Non-429 error — no point retrying

                except Exception as e:
                    logger.debug(f"Failed to download {mode_name} subtitles for '{lang}': {e}")
                    break

        return None

    def _fix_srt_overlaps(self, srt_path: Path) -> Path:
        """
        Fix YouTube's sliding-window SRT timestamps.

        YouTube auto-captions use a rolling format where each cue starts
        before the previous one ends, causing multiple lines to render
        simultaneously in ffmpeg. This trims every end time to the next
        cue's start time so the cues are strictly sequential.

        Args:
            srt_path: Path to the downloaded .srt file (edited in-place)

        Returns:
            Same path after fixing
        """
        text = srt_path.read_text(encoding="utf-8")

        block_pattern = re.compile(
            r"(\d+)\s*\n"
            r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s*\n"
            r"((?:.+\n?)+)",
            re.MULTILINE,
        )

        def ts_to_ms(ts: str) -> int:
            h, m, s = ts.split(":")
            s, ms = s.split(",")
            return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)

        def ms_to_ts(ms: int) -> str:
            h = ms // 3600000; ms %= 3600000
            m = ms // 60000;   ms %= 60000
            s = ms // 1000;    ms %= 1000
            return f"{h:02}:{m:02}:{s:02},{ms:03}"

        blocks = [
            (ts_to_ms(m.group(2)), ts_to_ms(m.group(3)), m.group(4).strip())
            for m in block_pattern.finditer(text)
        ]

        if not blocks:
            logger.warning(f"No SRT blocks parsed in {srt_path.name}, skipping overlap fix")
            return srt_path

        # Check if overlaps actually exist before rewriting
        has_overlaps = any(
            blocks[i][1] > blocks[i + 1][0]
            for i in range(len(blocks) - 1)
        )
        if not has_overlaps:
            logger.debug(f"No overlapping timestamps found in {srt_path.name}, skipping fix")
            return srt_path

        logger.info(f"Fixing overlapping SRT timestamps in {srt_path.name}")

        fixed = []
        for i, (start, end, content) in enumerate(blocks):
            if i + 1 < len(blocks):
                end = min(end, blocks[i + 1][0])
            if end > start:
                fixed.append((start, end, content))

        out_lines = [
            f"{i}\n{ms_to_ts(start)} --> {ms_to_ts(end)}\n{content}\n"
            for i, (start, end, content) in enumerate(fixed, 1)
        ]
        srt_path.write_text("\n".join(out_lines), encoding="utf-8")
        logger.info(f"Fixed {len(blocks) - len(fixed)} zero-duration cues, {len(fixed)} cues remaining")
        return srt_path

    def _find_subtitle_file(self, video_id: str, lang: str) -> Optional[Path]:
        """Find the downloaded subtitle file."""
        # yt-dlp saves subtitles as: video_id.lang.ext
        subtitle_exts = ["vtt", "srt", "ass", "json3"]
        
        for ext in subtitle_exts:
            path = self.subtitles_dir / f"{video_id}.{lang}.{ext}"
            if path.exists():
                logger.debug(f"Found subtitle: {path}")
                return path
        
        logger.debug(f"No subtitle file found for {video_id}.{lang}.*")
        return None

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