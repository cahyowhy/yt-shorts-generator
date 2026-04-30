"""Data models for the ShortGen pipeline."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class Platform(str, Enum):
    """Supported output platforms."""

    YOUTUBE_SHORTS = "youtube_shorts"
    INSTAGRAM_REELS = "instagram_reels"
    TIKTOK = "tiktok"

class VideoMetadata(BaseModel):
    """Source video information."""

    url: str
    video_id: str
    title: str
    duration: float  # seconds
    width: int
    height: int
    fps: float
    file_path: str
    original_lang: str
    subtitle_path: Optional[str] = None  # Add this field

    @computed_field
    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    @computed_field
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0


class TranscriptWord(BaseModel):
    """Single word from transcription with timing."""

    word: str
    start: float
    end: float
    confidence: float = 1.0


class Segment(BaseModel):
    """A candidate segment for short generation."""

    start_time: float
    end_time: float

    # Scoring signals (populated during analysis)
    audio_energy: float = 0.0
    scene_changes: int = 0
    face_presence: float = 0.0  # 0-1, percentage of frames with faces
    transcript: str = ""
    transcript_words: list[TranscriptWord] = Field(default_factory=list)
    highlight_score: float = 0.0  # LLM-assigned score
    hook: Optional[str]
    hook_audio_path: Optional[str]

    # Composite score (computed by scorer)
    final_score: float = 0.0

    @computed_field
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def overlaps_with(self, other: "Segment", threshold: float = 0.5) -> bool:
        """Check if this segment overlaps significantly with another."""
        overlap_start = max(self.start_time, other.start_time)
        overlap_end = min(self.end_time, other.end_time)
        overlap_duration = max(0, overlap_end - overlap_start)

        min_duration = min(self.duration, other.duration)
        if min_duration == 0:
            return False

        return (overlap_duration / min_duration) >= threshold


class FacePosition(BaseModel):
    """Face bounding box for a frame."""

    frame_number: int
    timestamp: float
    center_x: float  # Normalized 0-1
    center_y: float  # Normalized 0-1
    width: float  # Normalized 0-1
    height: float  # Normalized 0-1
    confidence: float


class CropWindow(BaseModel):
    """Dynamic crop window for vertical format."""

    timestamp: float
    x_offset: int
    y_offset: int
    width: int
    height: int

    @computed_field
    @property
    def center_x(self) -> int:
        return self.x_offset + self.width // 2


class ScoringWeights(BaseModel):
    """Configurable weights for segment scoring."""

    audio_energy: float = Field(default=0.25, ge=0, le=1)
    scene_activity: float = Field(default=0.15, ge=0, le=1)
    face_presence: float = Field(default=0.20, ge=0, le=1)
    highlight_score: float = Field(default=0.40, ge=0, le=1)

    def normalize(self) -> "ScoringWeights":
        """Ensure weights sum to 1."""
        total = (
            self.audio_energy
            + self.scene_activity
            + self.face_presence
            + self.highlight_score
        )
        if total == 0:
            return ScoringWeights()

        return ScoringWeights(
            audio_energy=self.audio_energy / total,
            scene_activity=self.scene_activity / total,
            face_presence=self.face_presence / total,
            highlight_score=self.highlight_score / total,
        )


class JobStatus(str, Enum):
    """Processing job status."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    ANALYZING = "analyzing"
    SCORING = "scoring"
    PROCESSING = "processing"
    RENDERING = "rendering"
    COMPLETE = "complete"
    FAILED = "failed"


class ProcessingJob(BaseModel):
    """Job tracking model."""

    job_id: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    source_url: str
    platform: Platform = Platform.YOUTUBE_SHORTS
    num_shorts: int = 5
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    segments_found: int = 0
    output_paths: list[str] = Field(default_factory=list)
    error: Optional[str] = None

    def update_status(self, status: JobStatus, progress: float = 0.0) -> None:
        """Update job status and timestamp."""
        self.status = status
        self.progress = progress
        self.updated_at = datetime.utcnow()
