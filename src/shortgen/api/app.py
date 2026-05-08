"""FastAPI application for ShortGen."""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from shortgen.config import settings
from shortgen.core.models import JobStatus, Platform, ProcessingJob, ScoringWeights
from shortgen.core.pipeline import ShortGeneratorPipeline

# In-memory job store (use Redis/DB for production)
jobs: dict[str, ProcessingJob] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    settings.ensure_directories()
    yield
    # Shutdown
    pass


app = FastAPI(
    title="ShortGen API",
    description="Generate YouTube Shorts/Reels from long-form videos",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class JobCreateRequest(BaseModel):
    """Request to create a new processing job."""

    url: str = Field(..., description="YouTube video URL")
    platform: Platform = Field(default=Platform.YOUTUBE_SHORTS)
    num_shorts: int = Field(default=5, ge=1, le=20, description="Number of shorts to generate (maps to --count)")
    weights: Optional[ScoringWeights] = None
    output_dir: Optional[str] = Field(default=None, description="Output directory path (maps to --output)")
    overide_lang: Optional[str] = Field(default=None, description="Force using selected lang, if not found or empty will use original lang (maps to --lang)")
    watermark_title: Optional[str] = Field(default=None, description="Embed watermark (maps to --wm)")
    tell_llm_skip_analyze_from_0_until: Optional[str] = Field(default=None, description="This inform llm to skip analyze transcript from 00:00:00 until ... (example 00:00:50) (maps to --llm-skip-analyze-ts-until)")
    video_cuts: Optional[List[List[float]]] = Field(
        default=None, 
        description="List of start/end times in seconds to bypass LLM, e.g., [[0,30],[32,67]]"
    )

    @field_validator('video_cuts')
    def validate_video_cuts(cls, v):
        if v is not None:
            if not all(len(i) == 2 for i in v):
                raise ValueError("Format tidak valid untuk video_cuts. Harus berupa array dari array 2-elemen, e.g., [[0,30],[32,67]]")
        return v


class JobResponse(BaseModel):
    """Job status response."""

    job_id: str
    status: JobStatus
    progress: float = 0.0
    source_url: str
    platform: Platform
    created_at: datetime
    output_paths: list[str] = []
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str


class VideoInfoResponse(BaseModel):
    """Video info response."""
    
    title: str
    duration: float
    resolution: str
    fps: str
    channel: str
    views: int


# Mount static files for output access (optional)
import os

output_folder = Path(settings.output_dir).resolve()
app.mount("/output", StaticFiles(directory=str(output_folder)), name="output")

@app.get("/output-files")
async def list_output_files():
    output_folder = Path.cwd() / "output"
    
    if not output_folder.exists():
        return {"error": "Folder tidak ditemukan"}
        
    files_list = []
    for file_path in output_folder.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(output_folder)
            files_list.append({
                "file_name": file_path.name,
                "url": f"/output/{rel_path}",
                "mtime": file_path.stat().st_mtime  # <-- Tambahkan baris ini
            })
            
    return {
        "files": files_list
    }

app.mount("/static", StaticFiles(directory="static"), name="static")

# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")


@app.get("/info", response_model=VideoInfoResponse)
async def get_video_info(url: str = Query(..., description="YouTube video URL")):
    """Get video information without processing (equivalent to 'shortgen info')."""
    from shortgen.acquisition.downloader import VideoDownloader
    
    downloader = VideoDownloader()
    
    try:
        info = await downloader.get_info(url)
        return VideoInfoResponse(
            title=info.get("title", "Unknown"),
            duration=info.get("duration", 0),
            resolution=f"{info.get('width', '?')}x{info.get('height', '?')}",
            fps=str(info.get("fps", "?")),
            channel=info.get("uploader", "Unknown"),
            views=info.get("view_count", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/jobs", response_model=JobResponse)
async def create_job(
    request: JobCreateRequest,
    background_tasks: BackgroundTasks,
):
    """Submit a new video processing job (equivalent to 'shortgen generate')."""
    job_id = str(uuid.uuid4())

    job = ProcessingJob(
        job_id=job_id,
        status=JobStatus.PENDING,
        source_url=request.url,
        platform=request.platform,
        num_shorts=request.num_shorts,
    )
    jobs[job_id] = job

    # Start background processing
    background_tasks.add_task(
        process_job_background,
        job_id,
        request,
    )

    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        source_url=job.source_url,
        platform=job.platform,
        created_at=job.created_at,
    )


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get job status and results."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        source_url=job.source_url,
        platform=job.platform,
        created_at=job.created_at,
        output_paths=job.output_paths,
        error=job.error,
    )


@app.get("/jobs", response_model=list[JobResponse])
async def list_jobs(limit: int = 10):
    """List recent jobs."""
    recent = sorted(
        jobs.values(),
        key=lambda j: j.created_at,
        reverse=True,
    )[:limit]

    return [
        JobResponse(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            source_url=job.source_url,
            platform=job.platform,
            created_at=job.created_at,
            output_paths=job.output_paths,
            error=job.error,
        )
        for job in recent
    ]


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    del jobs[job_id]
    return {"status": "deleted"}


@app.post("/clean")
async def clean_temp_files():
    """Clean temporary files (equivalent to 'shortgen clean')."""
    from shortgen.processing.clipper import VideoClipper
    try:
        clipper = VideoClipper()
        count = clipper.cleanup_temp_files()
        return {"status": "success", "cleaned_files": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Background task
async def process_job_background(
    job_id: str,
    request: JobCreateRequest,
) -> None:
    """Background task to process a job."""
    job = jobs.get(job_id)
    if not job:
        return

    def update_progress(stage: str, progress: float) -> None:
        if job_id in jobs:
            status_map = {
                "downloading": JobStatus.DOWNLOADING,
                "analyzing": JobStatus.ANALYZING,
                "scoring": JobStatus.SCORING,
                "processing": JobStatus.PROCESSING,
                "rendering": JobStatus.RENDERING,
                "complete": JobStatus.COMPLETE,
            }
            jobs[job_id].status = status_map.get(stage, JobStatus.PROCESSING)
            jobs[job_id].progress = progress

    try:
        weights = request.weights or ScoringWeights()

        pipeline = ShortGeneratorPipeline(
            weights=weights,
            progress_callback=update_progress,
        )

        output_path_obj = Path(request.output_dir) if request.output_dir else None

        output_paths = await pipeline.process(
            url=request.url,
            platform=request.platform,
            num_shorts=request.num_shorts,
            output_dir=output_path_obj,
            watermark_title=request.watermark_title,
            video_cuts=request.video_cuts,
            overide_lang=request.overide_lang,
            tell_llm_skip_analyze_from_0_until=request.tell_llm_skip_analyze_from_0_until
        )

        job.status = JobStatus.COMPLETE
        job.progress = 1.0
        job.output_paths = [str(p) for p in output_paths]

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)

# Endpoint untuk membuka file HTML utama
@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the Vue.js frontend."""
    return FileResponse("static/index.html")