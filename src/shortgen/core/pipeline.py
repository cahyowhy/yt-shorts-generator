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
    TranscriptWord,
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
    Unified orchestrator for the video-to-shorts pipeline.
    Supports both LLM-based highlights and Sliding Window analysis.
    """

    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        self.weights = (weights or ScoringWeights()).normalize()
        self.progress_callback = progress_callback

        # Initialize components
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
        all_segments: bool = False,
        use_sliding_window: bool = False,
    ) -> list[Path]:
        """
        Main processing pipeline.
        """
        output_dir = output_dir or settings.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Stage 1: Download
            self._update_progress("downloading", 0.0)
            metadata = await self.downloader.download(url)
            self._update_progress("downloading", 1.0)

            # Stage 2: Parallel Analysis
            self._update_progress("analyzing", 0.0)
            analysis_results = await self._run_analysis(metadata)
            self._update_progress("analyzing", 1.0)

            # Stage 3: Segment Generation
            self._update_progress("scoring", 0.0)
            
            if use_sliding_window:
                # Use sliding window logic from pipeline_old.py
                candidate_segments = self._generate_sliding_window_segments(metadata, analysis_results)
            else:
                # Use LLM highlight logic from pipeline.py
                candidate_segments = self._generate_llm_segments(metadata, analysis_results)

            # Score and rank segments
            scored_segments = self.scorer.score_segments(candidate_segments)
            
            # Selection: If all_segments is True, we don't cap the count
            selection_limit = None if all_segments else num_shorts
            top_segments = self._select_best_segments(scored_segments, selection_limit)
            
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

            self._update_progress("complete", 1.0)
            return output_paths

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    async def _run_analysis(self, metadata: VideoMetadata) -> dict:
        """Run all analysis tasks in parallel."""
        video_path = metadata.file_path

        tasks = [
            self.transcriber.transcribe(video_path, subtitle_path=getattr(metadata, 'subtitle_path', None)),
            self.audio_analyzer.analyze(video_path),
            self.scene_detector.detect(video_path),
            self.face_tracker.track(video_path)
        ]
        
        transcript, audio_energy, scenes, face_positions = await asyncio.gather(*tasks)

        highlights = []
        if transcript.get("text"):
            highlights = await self.highlight_finder.find_highlights(transcript["text"], metadata.duration)

        return {
            "transcript": transcript,
            "audio_energy": audio_energy,
            "scenes": scenes,
            "face_positions": face_positions,
            "highlights": highlights,
        }

    def _generate_llm_segments(self, metadata: VideoMetadata, analysis: dict) -> list[Segment]:
        """Generate segments based on LLM detected highlights."""
        highlights = analysis.get("highlights", [])
        segments = []
        for h in highlights:
            start, end = float(h.get("start", 0.0)), float(h.get("end", 0.0))
            if end <= start: continue

            segments.append(self._create_segment_object(start, end, float(h.get("score", 0.0)), analysis))
        return segments

    def _generate_sliding_window_segments(self, metadata: VideoMetadata, analysis: dict) -> list[Segment]:
        """Generate segments using sliding window approach from pipeline_old.py."""
        segments = []
        duration = metadata.duration
        window_size = settings.max_segment_duration
        step_size = window_size * (1 - settings.segment_overlap)

        current_time = 0.0
        while current_time + settings.min_segment_duration <= duration:
            end_time = min(current_time + window_size, duration)
            
            highlight_score = self._get_highlight_score_for_range(analysis["highlights"], current_time, end_time)
            
            segments.append(self._create_segment_object(current_time, end_time, highlight_score, analysis))
            current_time += step_size
        return segments

    def _create_segment_object(self, start: float, end: float, h_score: float, analysis: dict) -> Segment:
        """Helper to build Segment model with common analysis data."""
        return Segment(
            start_time=start,
            end_time=end,
            audio_energy=self._get_audio_energy_for_range(analysis["audio_energy"], start, end),
            scene_changes=self._count_scenes_in_range(analysis["scenes"], start, end),
            face_presence=self._get_face_presence_for_range(analysis["face_positions"], start, end),
            transcript=self._get_transcript_for_range(analysis["transcript"], start, end),
            transcript_words=self._get_transcript_words_for_range(analysis["transcript"], start, end),
            highlight_score=h_score,
        )

    def _select_best_segments(self, segments: list[Segment], limit: Optional[int]) -> list[Segment]:
        """Select top segments while removing overlapping ones."""
        sorted_segments = sorted(segments, key=lambda s: s.final_score, reverse=True)
        selected: list[Segment] = []
        
        for segment in sorted_segments:
            if limit and len(selected) >= limit:
                break

            # Prevent rendering overlapping content
            if not any(segment.overlaps_with(s, threshold=0.3) for s in selected):
                selected.append(segment)

        return selected

    async def _process_segment(self, metadata: VideoMetadata, segment: Segment, analysis_results: dict, 
                               platform: Platform, output_dir: Path, index: int) -> Path:
        """Extracts, crops, and renders a single segment."""
        clip_path = await self.clipper.extract(metadata.file_path, segment.start_time, segment.end_time)
        
        face_positions = self._filter_face_positions(analysis_results["face_positions"], segment.start_time, segment.end_time)
        crop_windows = self.cropper.calculate_crop_windows(metadata.resolution, (9, 16), face_positions, metadata.fps, segment.duration)
        
        captions = self.captioner.generate(words=segment.transcript_words)

        # Unified naming convention
        output_filename = f"short_{metadata.video_id}_{index:02d}_{int(segment.start_time)}s.mp4"
        output_path = output_dir / output_filename

        await self.renderer.render(clip_path, output_path, crop_windows, captions, platform)
        return output_path

    # ==================== Helper Methods (Shared Logic) ====================

    def _get_audio_energy_for_range(self, energy_data, start, end):
        values = [e for t, e in energy_data if start <= t <= end]
        return sum(values) / len(values) if values else 0.0

    def _count_scenes_in_range(self, scenes, start, end):
        return sum(1 for scene_time in scenes if start <= scene_time <= end)

    def _get_face_presence_for_range(self, face_positions, start, end):
        relevant = [fp for fp in face_positions if start <= fp.timestamp <= end]
        confident = sum(1 for fp in relevant if fp.confidence > 0.5)
        return confident / len(relevant) if relevant else 0.0

    def _get_transcript_for_range(self, transcript, start, end):
        words = transcript.get("words", [])
        return " ".join([w["word"] for w in words if start <= w.get("start", 0) <= end])

    def _get_transcript_words_for_range(self, transcript, start, end):
        return [TranscriptWord(word=w["word"], start=w["start"] - start, end=w["end"] - start, confidence=w.get("confidence", 1.0))
                for w in transcript.get("words", []) if start <= w.get("start", 0) <= end]

    def _get_highlight_score_for_range(self, highlights, start, end):
        scores = [h.get("score", 0.0) for h in highlights if start <= h.get("start", 0) <= end or start <= h.get("end", 0) <= end]
        return max(scores) if scores else 0.0

    def _filter_face_positions(self, positions, start, end):
        return [FacePosition(frame_number=fp.frame_number, timestamp=fp.timestamp - start, center_x=fp.center_x, 
                             center_y=fp.center_y, width=fp.width, height=fp.height, confidence=fp.confidence)
                for fp in positions if start <= fp.timestamp <= end]