# ShortGen - YouTube to Shorts/Reels Generator

Generate viral-worthy short-form videos from long-form YouTube content using local AI models.

## Features

- **No External APIs** - Uses local Whisper, Ollama, and MediaPipe
- **Smart Segment Detection** - Combines audio energy, scene changes, face tracking, and LLM analysis
- **Intelligent Vertical Cropping** - Tracks faces/speakers for optimal 9:16 framing
- **Auto Captions** - Burned-in captions from Whisper transcription
- **Multi-Platform Output** - YouTube Shorts, Instagram Reels, TikTok formats

## Quick Start

### Prerequisites

- Python 3.11+
- FFmpeg installed (`brew install ffmpeg` or `apt install ffmpeg`)
- [Ollama](https://ollama.ai) installed for LLM-based highlight detection

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/yt-shorts-generator.git
cd yt-shorts-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Download Whisper model (first run will auto-download)
# Or pre-download: python scripts/setup_models.py

# Pull Ollama model
ollama pull mistral
```

### Usage

#### CLI

```bash
# Generate 5 shorts from a YouTube video
shortgen generate "https://youtube.com/watch?v=VIDEO_ID"

# Customize output
shortgen generate "https://youtube.com/watch?v=VIDEO_ID" \
    --output ./my_shorts \
    --platform instagram_reels \
    --count 3

# Adjust scoring weights
shortgen generate "https://youtube.com/watch?v=VIDEO_ID" \
    --audio-weight 0.3 \
    --highlight-weight 0.5
```

#### API

```bash
# Start the API server
make api

# Or directly:
uvicorn shortgen.api.app:app --reload
```

Then submit jobs via HTTP:

```bash
curl -X POST http://localhost:8000/jobs \
    -H "Content-Type: application/json" \
    -d '{"url": "https://youtube.com/watch?v=VIDEO_ID", "num_shorts": 5}'
```

## Architecture

```
Input URL
    ↓
yt-dlp download
    ↓
Parallel analysis:
├── Whisper transcription
├── Audio energy (librosa)
├── Scene detection
└── Face tracking (MediaPipe)
    ↓
LLM highlight scoring (Ollama)
    ↓
Segment ranking
    ↓
Smart vertical crop + captions
    ↓
Output shorts
```

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `SHORTGEN_WHISPER_MODEL` | `base` | Whisper model size |
| `SHORTGEN_OLLAMA_MODEL` | `mistral` | LLM for highlights |
| `SHORTGEN_MIN_SEGMENT_DURATION` | `15` | Minimum short length |
| `SHORTGEN_MAX_SEGMENT_DURATION` | `60` | Maximum short length |
| `SHORTGEN_USE_GPU` | `true` | Enable GPU acceleration |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
make test

# Lint
make lint

# Run specific test
pytest tests/unit/test_scorer.py -v
```

## Project Structure

```
src/shortgen/
├── core/           # Pipeline orchestration, models
├── acquisition/    # Video downloading
├── analysis/       # Transcription, audio, scene, face tracking
├── scoring/        # Segment scoring engine
├── processing/     # Clipping, cropping, captions
├── output/         # Final rendering
├── api/            # FastAPI application
└── cli/            # Typer CLI
```

## License

MIT
