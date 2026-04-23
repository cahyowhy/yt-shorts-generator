"""Caption generation for video overlays."""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from shortgen.config import settings
from shortgen.core.models import TranscriptWord


@dataclass
class Caption:
    """A single caption to display."""

    text: str
    start_time: float
    end_time: float
    style: str = "default"


class Captioner:
    """Generate captions from transcript words."""

    def __init__(
        self,
        max_words_per_caption: int = 2,  # Changed from 5 to 2 for vertical video
        max_chars_per_caption: int = 20, # Reduced to prevent aggressive wrapping
        min_duration: float = 0.5,
        max_duration: float = 3.0,
    ):
        self.max_words_per_caption = max_words_per_caption
        self.max_chars_per_caption = max_chars_per_caption
        self.min_duration = min_duration
        self.max_duration = max_duration

    def generate(
        self,
        words: list[TranscriptWord],
        style: str = "default",
    ) -> list[Caption]:
        """
        Generate captions from transcript words.

        Args:
            words: List of TranscriptWord with timing
            style: Caption style name

        Returns:
            List of Caption objects
        """
        if not words:
            return []

        captions = []
        current_words: list[TranscriptWord] = []
        current_chars = 0

        for word in words:
            word_text = word.word.strip()
            if not word_text:
                continue

            word_len = len(word_text) + 1  # +1 for space

            # Check if we should start a new caption
            should_break = (
                len(current_words) >= self.max_words_per_caption
                or current_chars + word_len > self.max_chars_per_caption
                or (current_words and word.start - current_words[-1].end > 0.5)
            )

            if should_break and current_words:
                caption = self._create_caption(current_words, style)
                if caption:
                    captions.append(caption)
                current_words = []
                current_chars = 0

            current_words.append(word)
            current_chars += word_len

        # Handle remaining words
        if current_words:
            caption = self._create_caption(current_words, style)
            if caption:
                captions.append(caption)

        logger.debug(f"Generated {len(captions)} captions")
        return captions

    def _create_caption(
        self,
        words: list[TranscriptWord],
        style: str,
    ) -> Optional[Caption]:
        """Create a caption from a group of words."""
        if not words:
            return None

        text = " ".join(w.word.strip() for w in words)
        start_time = words[0].start
        end_time = words[-1].end

        # Ensure minimum duration
        duration = end_time - start_time
        if duration < self.min_duration:
            end_time = start_time + self.min_duration
        elif duration > self.max_duration:
            end_time = start_time + self.max_duration

        return Caption(
            text=text,
            start_time=start_time,
            end_time=end_time,
            style=style,
        )

    def to_srt(self, captions: list[Caption]) -> str:
        """Convert captions to SRT format."""
        lines = []

        for i, caption in enumerate(captions, 1):
            start = self._format_srt_time(caption.start_time)
            end = self._format_srt_time(caption.end_time)

            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            lines.append(caption.text)
            lines.append("")

        return "\n".join(lines)

    def to_ass(self, captions: list[Caption]) -> str:
        """Convert captions to ASS format (better styling support)."""
        
        # Pulling the new styling variables from settings
        header = """[Script Info]
Title: Generated Captions
ScriptType: v4.00+
Collisions: Normal
PlayDepth: 0
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,{outline},{shadow},2,10,10,250,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".format(
            font_size=settings.caption_font_size,
            width=settings.output_resolution_width,
            height=settings.output_resolution_height,
            outline=settings.caption_outline_width,
            shadow=settings.caption_shadow_width
        )

        lines = [header]

        for caption in captions:
            start = self._format_ass_time(caption.start_time)
            end = self._format_ass_time(caption.end_time)
            text = caption.text.replace("\n", "\\N")

            lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

        return "\n".join(lines)

    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _format_ass_time(self, seconds: float) -> str:
        """Format time for ASS (H:MM:SS.cc)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centis = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"