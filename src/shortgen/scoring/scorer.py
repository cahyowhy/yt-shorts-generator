"""Segment scoring engine."""

from typing import Optional

import numpy as np
from loguru import logger

from shortgen.core.models import ScoringWeights, Segment


class SegmentScorer:
    """Score segments based on multiple signals."""

    def __init__(self, weights: Optional[ScoringWeights] = None):
        self.weights = (weights or ScoringWeights()).normalize()

    def score_segments(self, segments: list[Segment]) -> list[Segment]:
        """
        Score all segments and update their final_score.

        Args:
            segments: List of segments with raw scores

        Returns:
            Same segments with final_score computed
        """
        if not segments:
            return segments

        # Normalize each signal across all segments
        audio_scores = self._normalize([s.audio_energy for s in segments])
        scene_scores = self._normalize([float(s.scene_changes) for s in segments])
        face_scores = self._normalize([s.face_presence for s in segments])
        highlight_scores = self._normalize([s.highlight_score for s in segments])

        # Compute weighted final scores
        for i, segment in enumerate(segments):
            segment.final_score = (
                self.weights.audio_energy * audio_scores[i]
                + self.weights.scene_activity * scene_scores[i]
                + self.weights.face_presence * face_scores[i]
                + self.weights.highlight_score * highlight_scores[i]
            )

        logger.info(
            f"Scored {len(segments)} segments, "
            f"top score: {max(s.final_score for s in segments):.3f}"
        )

        return segments

    def _normalize(self, values: list[float]) -> list[float]:
        """Normalize values to 0-1 range using min-max scaling."""
        if not values:
            return values

        arr = np.array(values)
        min_val = arr.min()
        max_val = arr.max()

        if max_val - min_val == 0:
            return [0.5] * len(values)

        normalized = (arr - min_val) / (max_val - min_val)
        return normalized.tolist()

    def score_single(self, segment: Segment) -> float:
        """
        Score a single segment.

        Note: Without other segments for normalization,
        this uses raw values which may not be ideal.
        """
        return (
            self.weights.audio_energy * segment.audio_energy
            + self.weights.scene_activity * min(1.0, segment.scene_changes / 10)
            + self.weights.face_presence * segment.face_presence
            + self.weights.highlight_score * segment.highlight_score
        )

    def explain_score(self, segment: Segment) -> dict:
        """Get breakdown of score components."""
        return {
            "final_score": segment.final_score,
            "components": {
                "audio_energy": {
                    "raw": segment.audio_energy,
                    "weight": self.weights.audio_energy,
                    "contribution": segment.audio_energy * self.weights.audio_energy,
                },
                "scene_activity": {
                    "raw": segment.scene_changes,
                    "weight": self.weights.scene_activity,
                },
                "face_presence": {
                    "raw": segment.face_presence,
                    "weight": self.weights.face_presence,
                    "contribution": segment.face_presence * self.weights.face_presence,
                },
                "highlight_score": {
                    "raw": segment.highlight_score,
                    "weight": self.weights.highlight_score,
                    "contribution": segment.highlight_score * self.weights.highlight_score,
                },
            },
            "time_range": f"{segment.start_time:.1f}s - {segment.end_time:.1f}s",
            "duration": segment.duration,
        }
