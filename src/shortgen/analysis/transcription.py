"""Audio transcription using OpenAI Whisper."""

import asyncio
from pathlib import Path
from typing import Literal, Optional

from loguru import logger

from shortgen.config import settings
from shortgen.core.exceptions import TranscriptionError

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
    """Transcribe audio using OpenAI Whisper."""

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

    async def transcribe(self, video_path: str) -> dict:
        """
        Transcribe audio from video file.

        Args:
            video_path: Path to video file

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
        )

        return result

    def _transcribe_sync(self, video_path: str) -> dict:
        """Synchronous transcription implementation."""
        try:
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

        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e

    async def transcribe_audio(self, audio_path: str) -> dict:
        """Transcribe from audio file directly."""
        return await self.transcribe(audio_path)
