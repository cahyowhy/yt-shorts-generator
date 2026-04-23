"""Audio transcription using downloaded subtitles or OpenAI Whisper as fallback."""

import asyncio
from pathlib import Path
from typing import Literal, Optional

from loguru import logger

from shortgen.config import settings
from shortgen.core.exceptions import TranscriptionError
from shortgen.core.models import TranscriptWord

# Lazy import whisper to avoid loading on import
_whisper = None


def _get_whisper():
    """Lazy load whisper module."""
    global _whisper
    if _whisper is None:
        import whisper
        _whisper = whisper
    return _whisper


class Transcriber:
    """Transcribe audio using downloaded subtitles or OpenAI Whisper fallback."""

    def __init__(
        self,
        model_name: Literal["tiny", "base", "small", "medium", "large"] = "base",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or self._detect_device()
        self._model = None

    def _detect_device(self) -> str:
        """Detect best available device."""
        if settings.whisper_device != "auto":
            return settings.whisper_device

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        return "cpu"

    def _load_model(self):
        """Load Whisper model (lazy loading)."""
        if self._model is None:
            whisper = _get_whisper()
            logger.info(f"Loading Whisper model: {self.model_name} on {self.device}")
            self._model = whisper.load_model(self.model_name, device=self.device)
        return self._model

    async def transcribe(self, video_path: str, subtitle_path: Optional[str] = None) -> dict:
        """
        Transcribe audio from video file.
        
        Tries to use downloaded subtitles first, falls back to Whisper if not available.

        Args:
            video_path: Path to video file
            subtitle_path: Optional path to subtitle file (vtt, srt, etc.)

        Returns:
            Dictionary with 'text', 'segments', and 'words'
        """
        path = Path(video_path)
        if not path.exists():
            raise TranscriptionError(f"Video file not found: {video_path}")

        logger.info(f"Transcribing: {path.name}")

        # Run in executor to not block async loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._transcribe_sync,
            str(path),
            subtitle_path,
        )

        return result

    def _transcribe_sync(self, video_path: str, subtitle_path: Optional[str] = None) -> dict:
        """Synchronous transcription implementation."""
        try:
            # Try to use downloaded subtitles first
            if subtitle_path:
                logger.info(f"Attempting to load subtitles from: {subtitle_path}")
                result = self._parse_subtitles(subtitle_path)
                if result:
                    logger.info("Successfully loaded transcription from subtitles")
                    return result
                else:
                    logger.warning("Failed to parse subtitles, falling back to Whisper")
            else:
                logger.info("No subtitle file provided, using Whisper for transcription")

            # Fallback to Whisper
            return self._transcribe_with_whisper(video_path)

        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e

    def _parse_subtitles(self, subtitle_path: str) -> Optional[dict]:
        """
        Parse subtitle file (VTT, SRT, etc.) and convert to transcript format.

        Args:
            subtitle_path: Path to subtitle file

        Returns:
            Dictionary with transcript data or None if parsing fails
        """
        try:
            path = Path(subtitle_path)
            if not path.exists():
                logger.warning(f"Subtitle file not found: {subtitle_path}")
                return None

            ext = path.suffix.lower()
            
            if ext == ".vtt":
                return self._parse_vtt(path)
            elif ext == ".srt":
                return self._parse_srt(path)
            elif ext == ".ass":
                return self._parse_ass(path)
            else:
                logger.warning(f"Unsupported subtitle format: {ext}")
                return None

        except Exception as e:
            logger.error(f"Error parsing subtitles: {e}")
            return None

    def _parse_vtt(self, path: Path) -> dict:
        """Parse VTT subtitle file."""
        text_parts = []
        words = []
        segments = []

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        current_start = 0.0
        current_end = 0.0
        current_text = []

        for line in lines:
            line = line.strip()

            # Skip WEBVTT header and NOTE lines
            if line.startswith("WEBVTT") or line.startswith("NOTE"):
                continue

            # Parse timestamp line (HH:MM:SS.mmm --> HH:MM:SS.mmm [cue settings])
            # Cue settings like "align:start position:0%" may follow the end time — strip them.
            if " --> " in line:
                start_str, end_str = line.split(" --> ", 1)
                end_str = end_str.split()[0]  # drop any trailing cue settings
                current_start = self._vtt_time_to_seconds(start_str.strip())
                current_end = self._vtt_time_to_seconds(end_str.strip())

            # Accumulate text lines
            elif line and not line.startswith("WEBVTT"):
                current_text.append(line)

            # Empty line indicates end of subtitle block
            elif not line and current_text:
                text = " ".join(current_text).strip()
                if text:
                    text_parts.append(text)
                    
                    # Create segment
                    segments.append({
                        "start": current_start,
                        "end": current_end,
                        "text": text,
                    })

                    # Split text into words with approximate timing
                    words.extend(self._split_into_words(text, current_start, current_end))

                current_text = []

        # Handle last subtitle block if file doesn't end with empty line
        if current_text:
            text = " ".join(current_text).strip()
            if text:
                text_parts.append(text)
                segments.append({
                    "start": current_start,
                    "end": current_end,
                    "text": text,
                })
                words.extend(self._split_into_words(text, current_start, current_end))

        full_text = " ".join(text_parts)

        return {
            "text": full_text,
            "segments": segments,
            "words": words,
            "language": "en",
        }

    def _parse_srt(self, path: Path) -> dict:
        """Parse SRT subtitle file."""
        text_parts = []
        words = []
        segments = []

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split by double newlines to get subtitle blocks
        blocks = content.split("\n\n")

        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue

            # Skip sequence number (first line is usually a number)
            # Second line is timing
            timing_line = lines[1] if len(lines) > 1 else ""
            if " --> " not in timing_line:
                continue

            try:
                start_str, end_str = timing_line.split(" --> ")
                current_start = self._srt_time_to_seconds(start_str.strip())
                current_end = self._srt_time_to_seconds(end_str.strip())

                # Remaining lines are subtitle text
                text = " ".join(lines[2:]).strip()
                
                if text:
                    text_parts.append(text)
                    segments.append({
                        "start": current_start,
                        "end": current_end,
                        "text": text,
                    })
                    words.extend(self._split_into_words(text, current_start, current_end))

            except (ValueError, IndexError) as e:
                logger.debug(f"Skipping malformed SRT block: {e}")
                continue

        full_text = " ".join(text_parts)

        return {
            "text": full_text,
            "segments": segments,
            "words": words,
            "language": "en",
        }

    def _parse_ass(self, path: Path) -> dict:
        """Parse ASS subtitle file."""
        text_parts = []
        words = []
        segments = []

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        in_events = False
        for line in lines:
            line = line.strip()

            if line.startswith("[Events]"):
                in_events = True
                continue

            if not in_events or not line.startswith("Dialogue:"):
                continue

            # Parse ASS dialogue line format
            # Dialogue: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
            try:
                parts = line.split(",", 9)
                if len(parts) < 10:
                    continue

                start_str = parts[1].strip()
                end_str = parts[2].strip()
                text = parts[9].strip()

                # Remove ASS formatting codes
                text = self._remove_ass_codes(text)

                if text:
                    current_start = self._ass_time_to_seconds(start_str)
                    current_end = self._ass_time_to_seconds(end_str)

                    text_parts.append(text)
                    segments.append({
                        "start": current_start,
                        "end": current_end,
                        "text": text,
                    })
                    words.extend(self._split_into_words(text, current_start, current_end))

            except (ValueError, IndexError) as e:
                logger.debug(f"Skipping malformed ASS line: {e}")
                continue

        full_text = " ".join(text_parts)

        return {
            "text": full_text,
            "segments": segments,
            "words": words,
            "language": "en",
        }

    def _split_into_words(self, text: str, start_time: float, end_time: float) -> list[dict]:
        """
        Split text into words and distribute timing evenly.

        Args:
            text: Text to split
            start_time: Start time of the segment
            end_time: End time of the segment

        Returns:
            List of words with timing
        """
        words_list = text.split()
        if not words_list:
            return []

        duration = end_time - start_time
        word_duration = duration / len(words_list)

        result = []
        for i, word in enumerate(words_list):
            result.append({
                "word": word,
                "start": start_time + (i * word_duration),
                "end": start_time + ((i + 1) * word_duration),
                "confidence": 1.0,
            })

        return result

    def _vtt_time_to_seconds(self, time_str: str) -> float:
        """Convert VTT time format (HH:MM:SS.mmm) to seconds.

        Also tolerates MM:SS.mmm (no hours) and ignores any trailing cue
        settings that may have survived splitting (e.g. '750 align:start').
        """
        # Guard: take only the first whitespace-delimited token
        time_str = time_str.split()[0]
        parts = time_str.split(":")
        if len(parts) == 2:          # MM:SS.mmm
            hours, minutes, sec_part = 0, int(parts[0]), parts[1]
        else:                        # HH:MM:SS.mmm
            hours, minutes, sec_part = int(parts[0]), int(parts[1]), parts[2]
        seconds_parts = sec_part.split(".")
        seconds = int(seconds_parts[0])
        # Strip anything non-numeric from millis (e.g. "750 align" → "750")
        raw_millis = seconds_parts[1] if len(seconds_parts) > 1 else "0"
        millis = int("".join(c for c in raw_millis if c.isdigit()) or "0")
        return hours * 3600 + minutes * 60 + seconds + millis / 1000

    def _srt_time_to_seconds(self, time_str: str) -> float:
        """Convert SRT time format (HH:MM:SS,mmm) to seconds."""
        parts = time_str.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_parts = parts[2].split(",")
        seconds = int(seconds_parts[0])
        millis = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
        return hours * 3600 + minutes * 60 + seconds + millis / 1000

    def _ass_time_to_seconds(self, time_str: str) -> float:
        """Convert ASS time format (H:MM:SS.cc) to seconds."""
        parts = time_str.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_parts = parts[2].split(".")
        seconds = int(seconds_parts[0])
        centis = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
        return hours * 3600 + minutes * 60 + seconds + centis / 100

    def _remove_ass_codes(self, text: str) -> str:
        """Remove ASS formatting codes like {\\b1}, {\\c&H...}."""
        import re
        return re.sub(r"\{\\[^}]*\}", "", text)

    def _transcribe_with_whisper(self, video_path: str) -> dict:
        """Transcribe audio using Whisper model."""
        logger.info("Using Whisper for transcription")
        model = self._load_model()

        # Transcribe with word-level timestamps
        result = model.transcribe(
            video_path,
            word_timestamps=True,
            verbose=False,
        )

        # Extract word-level data
        words = []
        for segment in result.get("segments", []):
            for word_data in segment.get("words", []):
                words.append({
                    "word": word_data.get("word", "").strip(),
                    "start": word_data.get("start", 0.0),
                    "end": word_data.get("end", 0.0),
                    "confidence": word_data.get("probability", 1.0),
                })

        return {
            "text": result.get("text", ""),
            "segments": result.get("segments", []),
            "words": words,
            "language": result.get("language", "en"),
        }

    async def transcribe_audio(self, audio_path: str, subtitle_path: Optional[str] = None) -> dict:
        """Transcribe from audio file directly."""
        return await self.transcribe(audio_path, subtitle_path)