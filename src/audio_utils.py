"""
audio_utils.py — Audio normalisation and chunking helpers.

Replaced pydub with direct ffmpeg subprocess calls.
pydub depends on audioop which was removed in Python 3.13,
so calling ffmpeg directly is cleaner and has no extra dependencies
beyond ffmpeg itself being installed on the system.
"""

import os
import subprocess
import logging

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mp4", ".webm", ".aac"}

# Works on both Windows (%TEMP%) and Unix (/tmp)
TEMP_DIR = os.path.join(os.environ.get("TEMP", "/tmp"), "transcription")

# Path to the local ffmpeg build installed via winget.
# We also add its bin directory to PATH so that libraries like
# openai-whisper (which invoke plain "ffmpeg"/"ffprobe") can find it.
FFMPEG = r"C:\Users\Hp\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
FFMPEG_DIR = os.path.dirname(FFMPEG)
_original_path = os.environ.get("PATH", "")
if FFMPEG_DIR not in _original_path:
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + _original_path
    logger.debug(f"Prepended ffmpeg directory to PATH: {FFMPEG_DIR}")


def _ffmpeg(args: list[str]) -> None:
    """Run an ffmpeg command, raising RuntimeError if it fails."""
    cmd = [FFMPEG, "-y"] + args
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed:\n{result.stderr.decode(errors='replace')}"
        )


def prepare_audio(input_path: str) -> str:
    """
    Convert any supported audio file to 16kHz mono WAV.
    This is what Whisper expects — doing it explicitly here
    means the model always gets consistent input.

    Raises ValueError for unsupported formats.
    Raises RuntimeError if ffmpeg conversion fails.
    """
    os.makedirs(TEMP_DIR, exist_ok=True)

    ext = os.path.splitext(input_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: '{ext}'. "
            f"Supported formats: {sorted(SUPPORTED_FORMATS)}"
        )

    out_path = os.path.join(TEMP_DIR, "prepared.wav")

    logger.info(f"Preparing audio: {input_path}")
    _ffmpeg([
        "-i", input_path,  # input file
        "-ac", "1",        # mono
        "-ar", "16000",    # 16kHz sample rate
        out_path,
    ])

    logger.info(f"Converted to 16kHz mono WAV: {out_path}")
    return out_path


def get_duration_seconds(audio_path: str) -> float:
    """Get audio duration using ffprobe (ships with ffmpeg)."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.decode(errors='replace')}")
    return float(result.stdout.decode().strip())


def chunk_audio(
    audio_path: str,
    chunk_sec: int = 60,
    overlap_sec: int = 2,
) -> list[tuple[str, float]]:
    """
    Split audio into overlapping chunks for long-file processing.

    The overlap prevents words getting cut at chunk boundaries.
    Returns list of (chunk_file_path, start_offset_seconds).
    """
    os.makedirs(TEMP_DIR, exist_ok=True)

    total_sec = get_duration_seconds(audio_path)
    chunks = []
    start = 0.0
    index = 0

    while start < total_sec:
        duration = min(chunk_sec, total_sec - start)
        chunk_path = os.path.join(TEMP_DIR, f"chunk_{index:04d}.wav")

        _ffmpeg([
            "-i", audio_path,
            "-ss", str(start),     # start time
            "-t",  str(duration),  # duration to extract
            "-ac", "1",
            "-ar", "16000",
            chunk_path,
        ])

        chunks.append((chunk_path, start))
        logger.debug(f"Chunk {index}: {start:.1f}s – {start + duration:.1f}s")

        start += chunk_sec - overlap_sec
        index += 1

    logger.info(f"Split into {len(chunks)} chunks")
    return chunks