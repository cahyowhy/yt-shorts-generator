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
    url: str,
    output: Path = typer.Option(None, "--output", "-o", help="Output directory"),
    platform: Platform = typer.Option(Platform.YOUTUBE_SHORTS, "--platform", "-p"),
    count: int = typer.Option(5, "--count", "-c", help="Number of shorts to generate"),
    all_segments: bool = typer.Option(False, "--all-segments", help="Render all segments found by the LLM"),
):
    pipeline = ShortGeneratorPipeline()
    import asyncio
    asyncio.run(pipeline.process(
        url=url,
        platform=platform,
        num_shorts=count,
        output_dir=output,
        all_segments=all_segments,
    ))

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
