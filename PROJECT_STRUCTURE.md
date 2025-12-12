# YouTube to Shorts/Reels Generator

## Project Structure

```
yt-shorts-generator/
│
├── README.md
├── pyproject.toml
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
│
├── src/
│   └── shortgen/
│       ├── __init__.py
│       ├── main.py                    # Entry point
│       ├── config.py                  # Pydantic settings
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── pipeline.py            # Main orchestrator
│       │   ├── models.py              # Data models (Pydantic)
│       │   └── exceptions.py          # Custom exceptions
│       │
│       ├── acquisition/
│       │   ├── __init__.py
│       │   ├── downloader.py          # yt-dlp wrapper
│       │   └── metadata.py            # Video metadata extraction
│       │
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── transcription.py       # Whisper integration
│       │   ├── audio_analyzer.py      # Librosa energy detection
│       │   ├── scene_detector.py      # Scene change detection
│       │   ├── face_tracker.py        # MediaPipe face tracking
│       │   └── highlight_finder.py    # LLM-based highlight detection
│       │
│       ├── scoring/
│       │   ├── __init__.py
│       │   ├── scorer.py              # Segment scoring engine
│       │   ├── strategies.py          # Pluggable scoring strategies
│       │   └── weights.py             # Configurable weight profiles
│       │
│       ├── processing/
│       │   ├── __init__.py
│       │   ├── clipper.py             # Segment extraction
│       │   ├── cropper.py             # Smart vertical cropping
│       │   ├── captioner.py           # Caption generation/burn-in
│       │   └── effects.py             # Optional effects (zoom, transitions)
│       │
│       ├── output/
│       │   ├── __init__.py
│       │   ├── renderer.py            # Final video rendering
│       │   └── formats.py             # Platform-specific formats
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── app.py                 # FastAPI application
│       │   ├── routes/
│       │   │   ├── __init__.py
│       │   │   ├── jobs.py            # Job submission/status
│       │   │   └── health.py          # Health checks
│       │   ├── schemas.py             # API request/response models
│       │   └── dependencies.py        # FastAPI dependencies
│       │
│       ├── cli/
│       │   ├── __init__.py
│       │   └── commands.py            # Typer CLI commands
│       │
│       ├── workers/
│       │   ├── __init__.py
│       │   └── tasks.py               # Background task processing
│       │
│       └── utils/
│           ├── __init__.py
│           ├── logging.py             # Structured logging setup
│           ├── metrics.py             # Prometheus metrics
│           ├── ffmpeg.py              # FFmpeg command builder
│           └── gpu.py                 # GPU detection/management
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_scorer.py
│   │   ├── test_cropper.py
│   │   └── test_transcription.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_pipeline.py
│   │   └── test_api.py
│   └── fixtures/
│       └── sample_video.mp4           # Test fixture
│
├── scripts/
│   ├── setup_models.py                # Download Whisper/LLM models
│   └── benchmark.py                   # Performance benchmarking
│
├── configs/
│   ├── default.yaml                   # Default configuration
│   ├── fast.yaml                      # Speed-optimized preset
│   └── quality.yaml                   # Quality-optimized preset
│
├── web/                               # Optional simple frontend
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
└── docs/
    ├── architecture.md
    ├── api.md
    └── deployment.md
```

---

## Core Data Models

```python
# src/shortgen/core/models.py

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
from datetime import timedelta

class Platform(str, Enum):
    YOUTUBE_SHORTS = "youtube_shorts"
    INSTAGRAM_REELS = "instagram_reels"
    TIKTOK = "tiktok"

class VideoMetadata(BaseModel):
    """Source video information"""
    url: str
    title: str
    duration: float
    resolution: tuple[int, int]
    fps: float
    file_path: str

class Segment(BaseModel):
    """A candidate segment for short generation"""
    start_time: float
    end_time: float
    
    # Scoring signals
    audio_energy: float = 0.0
    scene_changes: int = 0
    face_presence: float = 0.0  # 0-1, percentage of frames with faces
    transcript: Optional[str] = None
    highlight_score: float = 0.0  # LLM-assigned score
    
    # Composite score (computed)
    final_score: float = 0.0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

class FacePosition(BaseModel):
    """Face bounding box for a frame"""
    frame_number: int
    timestamp: float
    center_x: float  # Normalized 0-1
    center_y: float
    width: float
    height: float
    confidence: float

class CropWindow(BaseModel):
    """Dynamic crop window for vertical format"""
    timestamp: float
    x_offset: int
    y_offset: int
    width: int
    height: int

class ProcessingJob(BaseModel):
    """Job tracking model"""
    job_id: str
    status: str  # pending, downloading, analyzing, processing, complete, failed
    progress: float = 0.0
    source_url: str
    platform: Platform
    created_at: str
    segments_found: int = 0
    output_paths: list[str] = []
    error: Optional[str] = None

class ScoringWeights(BaseModel):
    """Configurable weights for segment scoring"""
    audio_energy: float = Field(default=0.25, ge=0, le=1)
    scene_activity: float = Field(default=0.15, ge=0, le=1)
    face_presence: float = Field(default=0.20, ge=0, le=1)
    highlight_score: float = Field(default=0.40, ge=0, le=1)
    
    def normalize(self) -> "ScoringWeights":
        """Ensure weights sum to 1"""
        total = self.audio_energy + self.scene_activity + self.face_presence + self.highlight_score
        return ScoringWeights(
            audio_energy=self.audio_energy / total,
            scene_activity=self.scene_activity / total,
            face_presence=self.face_presence / total,
            highlight_score=self.highlight_score / total,
        )
```

---

## Configuration

```python
# src/shortgen/config.py

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    """Application configuration with environment variable support"""
    
    # Paths
    data_dir: Path = Field(default=Path("./data"))
    output_dir: Path = Field(default=Path("./output"))
    models_dir: Path = Field(default=Path("./models"))
    
    # Whisper settings
    whisper_model: str = Field(default="base")  # tiny, base, small, medium, large
    whisper_device: str = Field(default="auto")  # auto, cpu, cuda
    
    # LLM settings (Ollama)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="mistral")
    
    # Processing settings
    min_segment_duration: float = Field(default=15.0)
    max_segment_duration: float = Field(default=60.0)
    target_segments: int = Field(default=5)
    
    # Output settings
    default_platform: str = Field(default="youtube_shorts")
    output_resolution: tuple[int, int] = Field(default=(1080, 1920))  # 9:16
    output_fps: int = Field(default=30)
    
    # API settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    
    # Performance
    max_concurrent_jobs: int = Field(default=2)
    use_gpu: bool = Field(default=True)
    
    class Config:
        env_prefix = "SHORTGEN_"
        env_file = ".env"

settings = Settings()
```

---

## Main Pipeline Orchestrator

```python
# src/shortgen/core/pipeline.py

import asyncio
from pathlib import Path
from typing import Optional, Callable
from loguru import logger

from shortgen.config import settings
from shortgen.core.models import (
    VideoMetadata, Segment, ProcessingJob, 
    ScoringWeights, Platform
)
from shortgen.acquisition.downloader import VideoDownloader
from shortgen.analysis.transcription import Transcriber
from shortgen.analysis.audio_analyzer import AudioAnalyzer
from shortgen.analysis.scene_detector import SceneDetector
from shortgen.analysis.face_tracker import FaceTracker
from shortgen.analysis.highlight_finder import HighlightFinder
from shortgen.scoring.scorer import SegmentScorer
from shortgen.processing.clipper import VideoClipper
from shortgen.processing.cropper import SmartCropper
from shortgen.processing.captioner import Captioner
from shortgen.output.renderer import VideoRenderer


class ShortGeneratorPipeline:
    """
    Main orchestrator for the video-to-shorts pipeline.
    
    Coordinates all stages: download → analyze → score → process → render
    """
    
    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        self.weights = weights or ScoringWeights()
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
    
    def _update_progress(self, stage: str, progress: float):
        """Report progress to callback if provided"""
        if self.progress_callback:
            self.progress_callback(stage, progress)
        logger.info(f"Pipeline progress: {stage} - {progress:.1%}")
    
    async def process(
        self,
        url: str,
        platform: Platform = Platform.YOUTUBE_SHORTS,
        num_shorts: int = 5,
        output_dir: Optional[Path] = None
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
            
            # Stage 2: Parallel Analysis
            self._update_progress("analyzing", 0.0)
            analysis_results = await self._run_analysis(metadata)
            self._update_progress("analyzing", 1.0)
            
            # Stage 3: Segment Generation & Scoring
            self._update_progress("scoring", 0.0)
            segments = self._generate_segments(metadata, analysis_results)
            scored_segments = self.scorer.score_segments(segments)
            top_segments = sorted(scored_segments, key=lambda s: s.final_score, reverse=True)[:num_shorts]
            self._update_progress("scoring", 1.0)
            
            # Stage 4: Process Each Segment
            output_paths = []
            for i, segment in enumerate(top_segments):
                self._update_progress("processing", i / len(top_segments))
                output_path = await self._process_segment(
                    metadata=metadata,
                    segment=segment,
                    analysis_results=analysis_results,
                    platform=platform,
                    output_dir=output_dir,
                    index=i
                )
                output_paths.append(output_path)
            
            self._update_progress("complete", 1.0)
            logger.info(f"Generated {len(output_paths)} shorts")
            return output_paths
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    async def _run_analysis(self, metadata: VideoMetadata) -> dict:
        """Run all analysis tasks in parallel"""
        
        # These can run concurrently
        transcript_task = asyncio.create_task(
            self.transcriber.transcribe(metadata.file_path)
        )
        audio_task = asyncio.create_task(
            self.audio_analyzer.analyze(metadata.file_path)
        )
        scene_task = asyncio.create_task(
            self.scene_detector.detect(metadata.file_path)
        )
        face_task = asyncio.create_task(
            self.face_tracker.track(metadata.file_path)
        )
        
        transcript, audio_energy, scenes, face_positions = await asyncio.gather(
            transcript_task, audio_task, scene_task, face_task
        )
        
        # LLM analysis depends on transcript
        highlights = await self.highlight_finder.find_highlights(transcript)
        
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
        analysis: dict
    ) -> list[Segment]:
        """Generate candidate segments from analysis results"""
        
        segments = []
        duration = metadata.duration
        
        # Sliding window approach
        window_size = settings.max_segment_duration
        step_size = window_size / 2  # 50% overlap
        
        current_time = 0.0
        while current_time + settings.min_segment_duration <= duration:
            end_time = min(current_time + window_size, duration)
            
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
                highlight_score=self._get_highlight_score_for_range(
                    analysis["highlights"], current_time, end_time
                ),
            )
            segments.append(segment)
            current_time += step_size
        
        return segments
    
    async def _process_segment(
        self,
        metadata: VideoMetadata,
        segment: Segment,
        analysis_results: dict,
        platform: Platform,
        output_dir: Path,
        index: int
    ) -> Path:
        """Process a single segment into a short"""
        
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
            segment.end_time
        )
        crop_windows = self.cropper.calculate_crop_windows(
            source_resolution=metadata.resolution,
            target_resolution=settings.output_resolution,
            face_positions=face_positions,
        )
        
        # Generate captions
        captions = self.captioner.generate(
            transcript=segment.transcript,
            start_offset=segment.start_time,
        )
        
        # Render final output
        output_filename = f"short_{index:02d}_{segment.start_time:.0f}s.mp4"
        output_path = output_dir / output_filename
        
        await self.renderer.render(
            input_path=clip_path,
            output_path=output_path,
            crop_windows=crop_windows,
            captions=captions,
            platform=platform,
        )
        
        return output_path
    
    # Helper methods for range-based lookups
    def _get_audio_energy_for_range(self, energy_data, start, end) -> float:
        # Implementation: average energy in time range
        ...
    
    def _count_scenes_in_range(self, scenes, start, end) -> int:
        # Implementation: count scene changes in range
        ...
    
    def _get_face_presence_for_range(self, face_data, start, end) -> float:
        # Implementation: percentage of frames with faces
        ...
    
    def _get_transcript_for_range(self, transcript, start, end) -> str:
        # Implementation: extract transcript text for range
        ...
    
    def _get_highlight_score_for_range(self, highlights, start, end) -> float:
        # Implementation: get LLM highlight score for range
        ...
    
    def _filter_face_positions(self, positions, start, end) -> list:
        # Implementation: filter face positions to time range
        ...
```

---

## CLI Interface

```python
# src/shortgen/cli/commands.py

import typer
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import asyncio

from shortgen.core.pipeline import ShortGeneratorPipeline
from shortgen.core.models import Platform, ScoringWeights
from shortgen.config import settings

app = typer.Typer(
    name="shortgen",
    help="Generate YouTube Shorts/Reels from long-form videos"
)
console = Console()


@app.command()
def generate(
    url: str = typer.Argument(..., help="YouTube video URL"),
    output: Path = typer.Option(
        settings.output_dir,
        "--output", "-o",
        help="Output directory"
    ),
    platform: Platform = typer.Option(
        Platform.YOUTUBE_SHORTS,
        "--platform", "-p",
        help="Target platform"
    ),
    count: int = typer.Option(
        5,
        "--count", "-n",
        help="Number of shorts to generate"
    ),
    audio_weight: float = typer.Option(0.25, help="Audio energy weight"),
    scene_weight: float = typer.Option(0.15, help="Scene activity weight"),
    face_weight: float = typer.Option(0.20, help="Face presence weight"),
    highlight_weight: float = typer.Option(0.40, help="Highlight score weight"),
):
    """Generate shorts from a YouTube video"""
    
    weights = ScoringWeights(
        audio_energy=audio_weight,
        scene_activity=scene_weight,
        face_presence=face_weight,
        highlight_score=highlight_weight,
    ).normalize()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing...", total=100)
        
        def update_progress(stage: str, pct: float):
            progress.update(task, description=stage.capitalize(), completed=pct * 100)
        
        pipeline = ShortGeneratorPipeline(
            weights=weights,
            progress_callback=update_progress
        )
        
        output_paths = asyncio.run(
            pipeline.process(
                url=url,
                platform=platform,
                num_shorts=count,
                output_dir=output,
            )
        )
    
    console.print(f"\n[green]✓ Generated {len(output_paths)} shorts:[/green]")
    for path in output_paths:
        console.print(f"  • {path}")


@app.command()
def analyze(
    url: str = typer.Argument(..., help="YouTube video URL"),
):
    """Analyze a video without generating shorts (preview mode)"""
    # Implementation: run analysis and display results
    ...


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current config"),
):
    """Manage configuration"""
    if show:
        console.print(settings.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
```

---

## FastAPI Application

```python
# src/shortgen/api/app.py

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uuid

from shortgen.config import settings
from shortgen.api.schemas import (
    JobCreateRequest, JobResponse, JobStatus
)
from shortgen.core.pipeline import ShortGeneratorPipeline
from shortgen.core.models import ProcessingJob

# In-memory job store (use Redis for production)
jobs: dict[str, ProcessingJob] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models, warm up
    yield
    # Shutdown: cleanup


app = FastAPI(
    title="ShortGen API",
    description="Generate YouTube Shorts from long-form videos",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/jobs", response_model=JobResponse)
async def create_job(
    request: JobCreateRequest,
    background_tasks: BackgroundTasks
):
    """Submit a new video processing job"""
    job_id = str(uuid.uuid4())
    
    job = ProcessingJob(
        job_id=job_id,
        status="pending",
        source_url=request.url,
        platform=request.platform,
        created_at=datetime.utcnow().isoformat(),
    )
    jobs[job_id] = job
    
    background_tasks.add_task(process_job, job_id, request)
    
    return JobResponse(job_id=job_id, status="pending")


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get job status and results"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        output_paths=job.output_paths,
        error=job.error,
    )


@app.get("/health")
async def health():
    return {"status": "healthy"}


async def process_job(job_id: str, request: JobCreateRequest):
    """Background task to process a job"""
    job = jobs[job_id]
    
    def update_progress(stage: str, progress: float):
        job.status = stage
        job.progress = progress
    
    try:
        pipeline = ShortGeneratorPipeline(
            weights=request.weights,
            progress_callback=update_progress
        )
        
        output_paths = await pipeline.process(
            url=request.url,
            platform=request.platform,
            num_shorts=request.num_shorts,
        )
        
        job.status = "complete"
        job.progress = 1.0
        job.output_paths = [str(p) for p in output_paths]
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
```

---

## pyproject.toml

```toml
[project]
name = "shortgen"
version = "0.1.0"
description = "Generate YouTube Shorts/Reels from long-form videos"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "yt-dlp>=2024.1.0",
    "openai-whisper>=20231117",
    "moviepy>=1.0.3",
    "librosa>=0.10.0",
    "scenedetect>=0.6.0",
    "mediapipe>=0.10.0",
    "opencv-python>=4.8.0",
    "ffmpeg-python>=0.2.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "fastapi>=0.109.0",
    "uvicorn>=0.27.0",
    "typer>=0.9.0",
    "rich>=13.0.0",
    "loguru>=0.7.0",
    "httpx>=0.26.0",
    "numpy>=1.26.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
    "mypy>=1.8.0",
]

[project.scripts]
shortgen = "shortgen.cli.commands:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

---

## Makefile

```makefile
.PHONY: install dev test lint run api docker

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src/shortgen

lint:
	ruff check src/ tests/
	mypy src/

run:
	shortgen generate $(URL)

api:
	uvicorn shortgen.api.app:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t shortgen .

docker-run:
	docker-compose up -d

setup-models:
	python scripts/setup_models.py
```

---

## Key Implementation Files to Build Next

Priority order for implementation:

1. **`acquisition/downloader.py`** - Get videos downloading first
2. **`analysis/transcription.py`** - Whisper integration
3. **`analysis/audio_analyzer.py`** - Audio energy detection
4. **`analysis/face_tracker.py`** - Face position tracking
5. **`processing/cropper.py`** - Smart vertical cropping (the hardest part)
6. **`scoring/scorer.py`** - Segment scoring logic
7. **`output/renderer.py`** - Final video rendering

Would you like me to implement any of these components in detail?
