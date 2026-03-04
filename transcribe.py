#!/usr/bin/env python3
"""
transcribe.py — CLI entry point.

Usage:
    python transcribe.py path/to/audio.mp3
    python transcribe.py path/to/audio.wav --model large
    python transcribe.py path/to/audio.m4a --output result.json
"""

import sys
import os
import json
import argparse
import logging

# Add src to path when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline import TranscriptionPipeline


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe an audio file to text")
    parser.add_argument("audio_file", help="Path to the audio file (WAV, MP3, M4A, etc.)")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write JSON output (prints to stdout if omitted)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    if not os.path.exists(args.audio_file):
        print(f"Error: file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    pipeline = TranscriptionPipeline(model_size=args.model)
    segments = pipeline.run(args.audio_file)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(segments, f, indent=2)
        print(f"Transcript written to: {args.output}")
    else:
        print("\n--- Transcript ---")
        for seg in segments:
            print(f"[{seg['start']:.2f}s → {seg['end']:.2f}s]  {seg['text']}")


if __name__ == "__main__":
    main()
