"""FastAPI application for ShortGen."""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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
    num_shorts: int = Field(default=5, ge=1, le=20)
    weights: Optional[ScoringWeights] = None


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


# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")


@app.post("/jobs", response_model=JobResponse)
async def create_job(
    request: JobCreateRequest,
    background_tasks: BackgroundTasks,
):
    """Submit a new video processing job."""
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

        output_paths = await pipeline.process(
            url=request.url,
            platform=request.platform,
            num_shorts=request.num_shorts,
        )

        job.status = JobStatus.COMPLETE
        job.progress = 1.0
        job.output_paths = [str(p) for p in output_paths]

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)


# Mount static files for output access (optional)
# app.mount("/outputs", StaticFiles(directory=settings.output_dir), name="outputs")
