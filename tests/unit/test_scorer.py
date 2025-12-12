"""Tests for segment scorer."""

import pytest

from shortgen.core.models import ScoringWeights, Segment
from shortgen.scoring.scorer import SegmentScorer


class TestScoringWeights:
    """Tests for ScoringWeights model."""

    def test_default_weights(self):
        weights = ScoringWeights()
        assert weights.audio_energy == 0.25
        assert weights.scene_activity == 0.15
        assert weights.face_presence == 0.20
        assert weights.highlight_score == 0.40

    def test_normalize_weights(self):
        weights = ScoringWeights(
            audio_energy=1.0,
            scene_activity=1.0,
            face_presence=1.0,
            highlight_score=1.0,
        )
        normalized = weights.normalize()

        total = (
            normalized.audio_energy
            + normalized.scene_activity
            + normalized.face_presence
            + normalized.highlight_score
        )
        assert abs(total - 1.0) < 0.001


class TestSegmentScorer:
    """Tests for SegmentScorer."""

    def test_score_empty_segments(self):
        scorer = SegmentScorer()
        result = scorer.score_segments([])
        assert result == []

    def test_score_single_segment(self):
        scorer = SegmentScorer()
        segment = Segment(
            start_time=0.0,
            end_time=30.0,
            audio_energy=0.8,
            scene_changes=3,
            face_presence=0.9,
            highlight_score=0.7,
        )

        result = scorer.score_segments([segment])

        assert len(result) == 1
        assert result[0].final_score >= 0
        assert result[0].final_score <= 1

    def test_score_multiple_segments_ranking(self):
        scorer = SegmentScorer()

        segments = [
            Segment(
                start_time=0.0,
                end_time=30.0,
                audio_energy=0.2,
                scene_changes=1,
                face_presence=0.3,
                highlight_score=0.1,
            ),
            Segment(
                start_time=30.0,
                end_time=60.0,
                audio_energy=0.9,
                scene_changes=5,
                face_presence=0.95,
                highlight_score=0.9,
            ),
        ]

        result = scorer.score_segments(segments)

        # Second segment should have higher score
        assert result[1].final_score > result[0].final_score

    def test_explain_score(self):
        scorer = SegmentScorer()
        segment = Segment(
            start_time=0.0,
            end_time=30.0,
            audio_energy=0.5,
            scene_changes=2,
            face_presence=0.5,
            highlight_score=0.5,
            final_score=0.5,
        )

        explanation = scorer.explain_score(segment)

        assert "final_score" in explanation
        assert "components" in explanation
        assert "audio_energy" in explanation["components"]
        assert "time_range" in explanation
