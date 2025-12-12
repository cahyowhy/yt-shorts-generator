#!/usr/bin/env python3
"""Download and setup required AI models."""

import subprocess
import sys


def setup_whisper():
    """Pre-download Whisper model."""
    print("Setting up Whisper model...")
    try:
        import whisper
        model = whisper.load_model("base")
        print("✓ Whisper base model ready")
    except Exception as e:
        print(f"✗ Whisper setup failed: {e}")


def setup_ollama():
    """Pull Ollama model."""
    print("Setting up Ollama model...")
    try:
        result = subprocess.run(
            ["ollama", "pull", "mistral"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("✓ Ollama mistral model ready")
        else:
            print(f"✗ Ollama setup failed: {result.stderr}")
    except FileNotFoundError:
        print("✗ Ollama not installed. Install from: https://ollama.ai")


def check_ffmpeg():
    """Check FFmpeg installation."""
    print("Checking FFmpeg...")
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✓ FFmpeg installed: {version}")
        else:
            print("✗ FFmpeg check failed")
    except FileNotFoundError:
        print("✗ FFmpeg not installed. Install from: https://ffmpeg.org/download.html")


def main():
    print("\n=== ShortGen Model Setup ===\n")

    check_ffmpeg()
    print()

    setup_whisper()
    print()

    setup_ollama()
    print()

    print("Setup complete!\n")


if __name__ == "__main__":
    main()
