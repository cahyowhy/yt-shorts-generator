"""CLI commands using Typer."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from shortgen.config import settings
from shortgen.core.models import Platform, ScoringWeights
from shortgen.core.pipeline import ShortGeneratorPipeline

app = typer.Typer(
    name="shortgen",
    help="Generate YouTube Shorts/Reels from long-form videos",
    add_completion=False,
)
console = Console()


@app.command()
def generate(
    url: str = typer.Argument(..., help="YouTube video URL"),
    output: Path = typer.Option(
        None,
        "--output", "-o",
        help="Output directory (default: ./output)",
    ),
    platform: Platform = typer.Option(
        Platform.YOUTUBE_SHORTS,
        "--platform", "-p",
        help="Target platform",
    ),
    count: int = typer.Option(
        5,
        "--count", "-n",
        help="Number of shorts to generate",
    ),
    audio_weight: float = typer.Option(
        0.25,
        "--audio-weight",
        help="Weight for audio energy (0-1)",
    ),
    scene_weight: float = typer.Option(
        0.15,
        "--scene-weight",
        help="Weight for scene activity (0-1)",
    ),
    face_weight: float = typer.Option(
        0.20,
        "--face-weight",
        help="Weight for face presence (0-1)",
    ),
    highlight_weight: float = typer.Option(
        0.40,
        "--highlight-weight",
        help="Weight for LLM highlight score (0-1)",
    ),
) -> None:
    """Generate shorts from a YouTube video."""

    output_dir = output or settings.output_dir

    weights = ScoringWeights(
        audio_energy=audio_weight,
        scene_activity=scene_weight,
        face_presence=face_weight,
        highlight_score=highlight_weight,
    ).normalize()

    console.print(f"\n[bold blue]ShortGen[/bold blue] - Generating {count} shorts")
    console.print(f"URL: {url}")
    console.print(f"Platform: {platform.value}")
    console.print(f"Output: {output_dir}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing...", total=100)

        def update_progress(stage: str, pct: float) -> None:
            description = stage.replace("_", " ").capitalize()
            progress.update(task, description=description, completed=pct * 100)

        pipeline = ShortGeneratorPipeline(
            weights=weights,
            progress_callback=update_progress,
        )

        try:
            output_paths = asyncio.run(
                pipeline.process(
                    url=url,
                    platform=platform,
                    num_shorts=count,
                    output_dir=output_dir,
                )
            )

            progress.update(task, description="Complete!", completed=100)

        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")
            raise typer.Exit(1)

    # Display results
    console.print(f"\n[green]✓ Generated {len(output_paths)} shorts:[/green]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("File")
    table.add_column("Size", justify="right")

    for i, path in enumerate(output_paths, 1):
        size_mb = path.stat().st_size / (1024 * 1024)
        table.add_row(str(i), path.name, f"{size_mb:.1f} MB")

    console.print(table)
    console.print(f"\n[dim]Output directory: {output_dir}[/dim]")


@app.command()
def info(
    url: str = typer.Argument(..., help="YouTube video URL"),
) -> None:
    """Get video information without processing."""

    from shortgen.acquisition.downloader import VideoDownloader

    console.print(f"\n[bold]Fetching video info...[/bold]\n")

    downloader = VideoDownloader()

    try:
        info = asyncio.run(downloader.get_info(url))

        table = Table(show_header=False, box=None)
        table.add_column("Field", style="bold")
        table.add_column("Value")

        table.add_row("Title", info.get("title", "Unknown"))
        table.add_row("Duration", f"{info.get('duration', 0):.0f} seconds")
        table.add_row("Resolution", f"{info.get('width', '?')}x{info.get('height', '?')}")
        table.add_row("FPS", str(info.get("fps", "?")))
        table.add_row("Channel", info.get("uploader", "Unknown"))
        table.add_row("Views", f"{info.get('view_count', 0):,}")

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def config(
    show: bool = typer.Option(
        False,
        "--show",
        help="Show current configuration",
    ),
) -> None:
    """View or manage configuration."""

    if show:
        import json

        console.print("\n[bold]Current Configuration:[/bold]\n")
        config_dict = settings.model_dump()

        # Convert Path objects to strings
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)

        console.print_json(json.dumps(config_dict, indent=2))
    else:
        console.print("Use --show to display current configuration")
        console.print("Edit .env file to change settings")


@app.command()
def clean(
    confirm: bool = typer.Option(
        False,
        "--yes", "-y",
        help="Skip confirmation",
    ),
) -> None:
    """Clean temporary files."""

    from shortgen.processing.clipper import VideoClipper

    if not confirm:
        confirm = typer.confirm("Delete all temporary files?")

    if confirm:
        clipper = VideoClipper()
        count = clipper.cleanup_temp_files()
        console.print(f"[green]Cleaned {count} temporary files[/green]")
    else:
        console.print("Cancelled")


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
