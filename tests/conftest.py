"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_video_path() -> Path:
    """Path to sample video fixture."""
    return Path(__file__).parent / "fixtures" / "sample_video.mp4"


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
