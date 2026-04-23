"""Application configuration with environment variable support."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="SHORTGEN_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Paths
    data_dir: Path = Field(default=Path("./data"))
    output_dir: Path = Field(default=Path("./output"))
    models_dir: Path = Field(default=Path("./models"))

    # Whisper settings
    whisper_model: Literal["tiny", "base", "small", "medium", "large"] = Field(default="base")
    whisper_device: str = Field(default="auto")  # auto, cpu, cuda

    # LLM settings (Ollama)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="mistral")

    # Processing settings
    min_segment_duration: float = Field(default=15.0)
    max_segment_duration: float = Field(default=60.0)
    target_segments: int = Field(default=5)
    segment_overlap: float = Field(default=0.5)  # 50% overlap between windows

    # Output settings
    default_platform: str = Field(default="youtube_shorts")
    output_resolution_width: int = Field(default=1080)
    output_resolution_height: int = Field(default=1920)
    output_fps: int = Field(default=30)
    output_bitrate: str = Field(default="8M")

    # Caption settings
    caption_font_size: int = Field(default=96)  # Increased for 1080x1920 canvas
    caption_font_color: str = Field(default="white")
    caption_bg_color: str = Field(default="black@0.7")
    caption_position: str = Field(default="bottom")
    caption_outline_width: int = Field(default=5)  # Added thick outline for readability
    caption_shadow_width: int = Field(default=2)   # Added slight shadow for depth

    # API settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # Performance
    max_concurrent_jobs: int = Field(default=2)
    use_gpu: bool = Field(default=True)

    @property
    def output_resolution(self) -> tuple[int, int]:
        return (self.output_resolution_width, self.output_resolution_height)

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()