"""Main pipeline orchestrator for video-to-shorts generation."""

import asyncio
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from shortgen.acquisition.downloader import VideoDownloader
from shortgen.analysis.audio_analyzer import AudioAnalyzer
from shortgen.analysis.face_tracker import FaceTracker
from shortgen.analysis.highlight_finder import HighlightFinder
from shortgen.analysis.scene_detector import SceneDetector
from shortgen.analysis.transcription import Transcriber
from shortgen.config import settings
from shortgen.core.models import (
    FacePosition,
    Platform,
    ScoringWeights,
    Segment,
    VideoMetadata,
)
from shortgen.output.renderer import VideoRenderer
from shortgen.processing.captioner import Captioner
from shortgen.processing.clipper import VideoClipper
from shortgen.processing.cropper import SmartCropper
from shortgen.scoring.scorer import SegmentScorer

# Type alias for progress callback
ProgressCallback = Callable[[str, float], None]


class ShortGeneratorPipeline:
    """
    Main orchestrator for the video-to-shorts pipeline.

    Coordinates all stages: download → analyze → score → process → render
    """

    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        self.weights = (weights or ScoringWeights()).normalize()
        self.progress_callback = progress_callback

        # Initialize components (lazy loading in production)
        self.downloader = VideoDownloader()
        self.transcriber = Transcriber(model_name=settings.whisper_model)
        self.audio_analyzer = AudioAnalyzer()
        self.scene_detector = SceneDetector()
        self.face_tracker = FaceTracker()
        self.highlight_finder = HighlightFinder()
        self.scorer = SegmentScorer(weights=self.weights)
        self.clipper = VideoClipper()
        self.cropper = SmartCropper()
        self.captioner = Captioner()
        self.renderer = VideoRenderer()

    def _update_progress(self, stage: str, progress: float) -> None:
        """Report progress to callback if provided."""
        if self.progress_callback:
            self.progress_callback(stage, progress)
        logger.info(f"Pipeline progress: {stage} - {progress:.1%}")

    async def process(
        self,
        url: str,
        platform: Platform = Platform.YOUTUBE_SHORTS,
        num_shorts: int = 5,
        output_dir: Optional[Path] = None,
    ) -> list[Path]:
        """
        Main processing pipeline.

        Args:
            url: YouTube video URL
            platform: Target platform for output format
            num_shorts: Number of shorts to generate
            output_dir: Output directory for generated shorts

        Returns:
            List of paths to generated short videos
        """
        output_dir = output_dir or settings.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Stage 1: Download
            self._update_progress("downloading", 0.0)
            metadata = await self.downloader.download(url)
            self._update_progress("downloading", 1.0)
            logger.info(f"Downloaded: {metadata.title} ({metadata.duration:.1f}s)")

            # Stage 2: Parallel Analysis
            self._update_progress("analyzing", 0.0)
            analysis_results = await self._run_analysis(metadata)
            self._update_progress("analyzing", 1.0)
            logger.info("Analysis complete")

            # Stage 3: Segment Generation & Scoring
            self._update_progress("scoring", 0.0)
            segments = self._generate_segments(metadata, analysis_results)
            scored_segments = self.scorer.score_segments(segments)

            # Remove overlapping segments and take top N
            top_segments = self._select_best_segments(scored_segments, num_shorts)
            self._update_progress("scoring", 1.0)
            logger.info(f"Selected {len(top_segments)} segments")

            # Stage 4: Process Each Segment
            output_paths: list[Path] = []
            for i, segment in enumerate(top_segments):
                self._update_progress("processing", i / len(top_segments))
                output_path = await self._process_segment(
                    metadata=metadata,
                    segment=segment,
                    analysis_results=analysis_results,
                    platform=platform,
                    output_dir=output_dir,
                    index=i,
                )
                output_paths.append(output_path)
                logger.info(f"Generated short {i + 1}/{len(top_segments)}: {output_path.name}")

            self._update_progress("complete", 1.0)
            logger.info(f"Pipeline complete: generated {len(output_paths)} shorts")
            return output_paths

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    async def _run_analysis(self, metadata: VideoMetadata) -> dict:
        """Run all analysis tasks in parallel."""
        video_path = metadata.file_path

        # These can run concurrently
        transcript_task = asyncio.create_task(
            self.transcriber.transcribe(video_path)
        )
        audio_task = asyncio.create_task(
            self.audio_analyzer.analyze(video_path)
        )
        scene_task = asyncio.create_task(
            self.scene_detector.detect(video_path)
        )
        face_task = asyncio.create_task(
            self.face_tracker.track(video_path)
        )

        transcript, audio_energy, scenes, face_positions = await asyncio.gather(
            transcript_task, audio_task, scene_task, face_task
        )

        # LLM analysis depends on transcript (if transcript exists)
        highlights = []
        if transcript.get("text"):
            highlights = await self.highlight_finder.find_highlights(
                transcript["text"],
                metadata.duration,
            )

        return {
            "transcript": transcript,
            "audio_energy": audio_energy,
            "scenes": scenes,
            "face_positions": face_positions,
            "highlights": highlights,
        }

    def _generate_segments(
        self,
        metadata: VideoMetadata,
        analysis: dict,
    ) -> list[Segment]:
        """Generate candidate segments from analysis results."""
        segments: list[Segment] = []
        duration = metadata.duration

        # Sliding window approach
        window_size = settings.max_segment_duration
        step_size = window_size * (1 - settings.segment_overlap)

        current_time = 0.0
        while current_time + settings.min_segment_duration <= duration:
            end_time = min(current_time + window_size, duration)

            # Skip if segment too short
            if end_time - current_time < settings.min_segment_duration:
                break

            segment = Segment(
                start_time=current_time,
                end_time=end_time,
                audio_energy=self._get_audio_energy_for_range(
                    analysis["audio_energy"], current_time, end_time
                ),
                scene_changes=self._count_scenes_in_range(
                    analysis["scenes"], current_time, end_time
                ),
                face_presence=self._get_face_presence_for_range(
                    analysis["face_positions"], current_time, end_time
                ),
                transcript=self._get_transcript_for_range(
                    analysis["transcript"], current_time, end_time
                ),
                transcript_words=self._get_transcript_words_for_range(
                    analysis["transcript"], current_time, end_time
                ),
                highlight_score=self._get_highlight_score_for_range(
                    analysis["highlights"], current_time, end_time
                ),
            )
            segments.append(segment)
            current_time += step_size

        return segments

    def _select_best_segments(
        self,
        segments: list[Segment],
        num_shorts: int,
    ) -> list[Segment]:
        """Select top segments, removing overlapping ones."""
        sorted_segments = sorted(segments, key=lambda s: s.final_score, reverse=True)

        selected: list[Segment] = []
        for segment in sorted_segments:
            if len(selected) >= num_shorts:
                break

            # Check for overlap with already selected segments
            has_overlap = any(
                segment.overlaps_with(selected_seg, threshold=0.3)
                for selected_seg in selected
            )

            if not has_overlap:
                selected.append(segment)

        return selected

    async def _process_segment(
        self,
        metadata: VideoMetadata,
        segment: Segment,
        analysis_results: dict,
        platform: Platform,
        output_dir: Path,
        index: int,
    ) -> Path:
        """Process a single segment into a short."""
        # Extract clip
        clip_path = await self.clipper.extract(
            video_path=metadata.file_path,
            start_time=segment.start_time,
            end_time=segment.end_time,
        )

        # Calculate smart crop based on face positions
        face_positions = self._filter_face_positions(
            analysis_results["face_positions"],
            segment.start_time,
            segment.end_time,
        )
        crop_windows = self.cropper.calculate_crop_windows(
            source_resolution=metadata.resolution,
            target_aspect_ratio=(9, 16),
            face_positions=face_positions,
            fps=metadata.fps,
            duration=segment.duration,
        )

        # Generate captions
        captions = self.captioner.generate(
            words=segment.transcript_words,
        )

        # Render final output
        output_filename = f"short_{index:02d}_{int(segment.start_time)}s.mp4"
        output_path = output_dir / output_filename

        await self.renderer.render(
            input_path=clip_path,
            output_path=output_path,
            crop_windows=crop_windows,
            captions=captions,
            platform=platform,
        )

        return output_path

    # ==================== Helper Methods ====================

    def _get_audio_energy_for_range(
        self,
        energy_data: list[tuple[float, float]],
        start: float,
        end: float,
    ) -> float:
        """Get average audio energy in time range."""
        if not energy_data:
            return 0.0

        values = [
            energy for timestamp, energy in energy_data
            if start <= timestamp <= end
        ]
        return sum(values) / len(values) if values else 0.0

    def _count_scenes_in_range(
        self,
        scenes: list[float],
        start: float,
        end: float,
    ) -> int:
        """Count scene changes in time range."""
        return sum(1 for scene_time in scenes if start <= scene_time <= end)

    def _get_face_presence_for_range(
        self,
        face_positions: list[FacePosition],
        start: float,
        end: float,
    ) -> float:
        """Get percentage of frames with faces in range."""
        if not face_positions:
            return 0.0

        relevant = [
            fp for fp in face_positions
            if start <= fp.timestamp <= end
        ]

        if not relevant:
            return 0.0

        # Calculate expected number of frames
        duration = end - start
        # Assuming face tracking was done at some sample rate
        # For now, return ratio of frames with confident detections
        confident = sum(1 for fp in relevant if fp.confidence > 0.5)
        return confident / len(relevant) if relevant else 0.0

    def _get_transcript_for_range(
        self,
        transcript: dict,
        start: float,
        end: float,
    ) -> str:
        """Extract transcript text for time range."""
        words = transcript.get("words", [])
        relevant_words = [
            w["word"] for w in words
            if start <= w.get("start", 0) <= end
        ]
        return " ".join(relevant_words)

    def _get_transcript_words_for_range(
        self,
        transcript: dict,
        start: float,
        end: float,
    ) -> list:
        """Extract transcript words with timing for range."""
        from shortgen.core.models import TranscriptWord

        words = transcript.get("words", [])
        return [
            TranscriptWord(
                word=w["word"],
                start=w["start"] - start,  # Normalize to segment start
                end=w["end"] - start,
                confidence=w.get("confidence", 1.0),
            )
            for w in words
            if start <= w.get("start", 0) <= end
        ]

    def _get_highlight_score_for_range(
        self,
        highlights: list[dict],
        start: float,
        end: float,
    ) -> float:
        """Get LLM highlight score for time range."""
        if not highlights:
            return 0.0

        scores = [
            h.get("score", 0.0)
            for h in highlights
            if start <= h.get("start", 0) <= end or start <= h.get("end", 0) <= end
        ]
        return max(scores) if scores else 0.0

    def _filter_face_positions(
        self,
        positions: list[FacePosition],
        start: float,
        end: float,
    ) -> list[FacePosition]:
        """Filter face positions to time range."""
        return [
            FacePosition(
                frame_number=fp.frame_number,
                timestamp=fp.timestamp - start,  # Normalize to segment start
                center_x=fp.center_x,
                center_y=fp.center_y,
                width=fp.width,
                height=fp.height,
                confidence=fp.confidence,
            )
            for fp in positions
            if start <= fp.timestamp <= end
        ]
