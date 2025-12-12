"""Audio energy analysis using librosa."""

import asyncio
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from shortgen.core.exceptions import AnalysisError


class AudioAnalyzer:
    """Analyze audio energy levels for detecting engaging segments."""

    def __init__(
        self,
        hop_length: int = 512,
        sample_rate: int = 22050,
        frame_duration: float = 0.5,  # Analysis window in seconds
    ):
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration

    async def analyze(self, video_path: str) -> list[tuple[float, float]]:
        """
        Analyze audio energy from video file.

        Args:
            video_path: Path to video file

        Returns:
            List of (timestamp, energy) tuples
        """
        path = Path(video_path)
        if not path.exists():
            raise AnalysisError(f"Video file not found: {video_path}")

        logger.info(f"Analyzing audio energy: {path.name}")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._analyze_sync,
            str(path),
        )

        return result

    def _analyze_sync(self, video_path: str) -> list[tuple[float, float]]:
        """Synchronous audio analysis implementation."""
        try:
            import librosa

            # Load audio from video
            y, sr = librosa.load(video_path, sr=self.sample_rate, mono=True)

            # Calculate RMS energy
            rms = librosa.feature.rms(
                y=y,
                hop_length=self.hop_length,
            )[0]

            # Convert to time-energy pairs
            times = librosa.times_like(rms, sr=sr, hop_length=self.hop_length)

            # Normalize energy to 0-1 range
            if rms.max() > 0:
                rms_normalized = rms / rms.max()
            else:
                rms_normalized = rms

            # Aggregate into larger windows for analysis
            window_samples = int(self.frame_duration * sr / self.hop_length)
            energy_data = []

            for i in range(0, len(rms_normalized), window_samples):
                window_rms = rms_normalized[i:i + window_samples]
                if len(window_rms) > 0:
                    timestamp = times[i] if i < len(times) else times[-1]
                    avg_energy = float(np.mean(window_rms))
                    energy_data.append((timestamp, avg_energy))

            logger.info(f"Extracted {len(energy_data)} energy samples")
            return energy_data

        except Exception as e:
            raise AnalysisError(f"Audio analysis failed: {e}") from e

    def get_high_energy_regions(
        self,
        energy_data: list[tuple[float, float]],
        threshold: float = 0.7,
        min_duration: float = 3.0,
    ) -> list[tuple[float, float]]:
        """
        Find regions with high audio energy.

        Args:
            energy_data: List of (timestamp, energy) tuples
            threshold: Energy threshold (0-1)
            min_duration: Minimum region duration in seconds

        Returns:
            List of (start_time, end_time) tuples for high-energy regions
        """
        if not energy_data:
            return []

        regions = []
        region_start: Optional[float] = None

        for timestamp, energy in energy_data:
            if energy >= threshold:
                if region_start is None:
                    region_start = timestamp
            else:
                if region_start is not None:
                    duration = timestamp - region_start
                    if duration >= min_duration:
                        regions.append((region_start, timestamp))
                    region_start = None

        # Handle case where high energy extends to end
        if region_start is not None:
            last_timestamp = energy_data[-1][0]
            duration = last_timestamp - region_start
            if duration >= min_duration:
                regions.append((region_start, last_timestamp))

        return regions
