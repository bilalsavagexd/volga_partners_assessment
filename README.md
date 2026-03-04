# Transcription Pipeline

A speech-to-text pipeline built on [OpenAI Whisper](https://github.com/openai/whisper). It accepts audio files in any common format, transcribes them with per-segment timestamps, and exposes the whole thing as a REST API.

---

## Quick Start

```bash
# Install dependencies (requires Python 3.10+)
pip install -r requirements.txt

# System dependency for audio format conversion
brew install ffmpeg         # macOS
sudo apt install ffmpeg     # Ubuntu/Debian
winget install ffmpeg       # Windows

# Run via CLI
python transcribe.py path/to/audio.mp3

# Run as API server
uvicorn src.api:app --reload --port 8000
```

---

## Part 1: Transcription Pipeline — Design Decisions

### Why Whisper?

I considered a few options: Google Speech-to-Text (accurate, but costs money per minute), Vosk (offline, lightweight, but noticeably worse on accented speech), and Whisper. Whisper wins for this use case because it's open-source, runs fully offline, handles multiple languages without configuration, and its `base` model is fast enough on CPU for most workloads. Larger models (`medium`, `large`) are available if accuracy matters more than speed.

### Audio Format Handling

Rather than writing format-specific parsers, all audio goes through `pydub` + FFmpeg before hitting Whisper. The normalisation step converts everything to **16kHz mono WAV** — which is what Whisper expects internally anyway. Doing this explicitly in `prepare_audio()` means:

1. Format errors surface early, with a clear message, instead of failing silently inside the model
2. The model always gets consistent input quality
3. Adding a new supported format is a one-line change to the `SUPPORTED_FORMATS` set

Supported formats today: `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.mp4`, `.webm`, `.aac`

### Long File Handling

Files over 25MB are automatically chunked into 60-second segments with a 2-second overlap. The overlap is important — without it, words at chunk boundaries get dropped. After transcription each chunk's timestamps are offset by the chunk's start time, so the final output has consistent absolute timestamps across the full recording.

The threshold is configurable via `long_file_threshold_mb` in `pipeline.run()`. An alternative would be using Whisper's built-in VAD, which works fine for files under ~30 minutes, but chunking is safer for longer recordings because it keeps memory usage bounded and is easy to parallelise later.

---

## Part 2: System Design (Optional API & Extras)

The core of this project is the CLI command:

```bash
python transcribe.py path/to/audio.mp3
```

Everything else in `src/api.py` and `src/retry.py` is optional scaffolding showing how you *could* wrap the pipeline in an HTTP API or add retry logic in a production system. You don't need any of that to run transcriptions from the command line.

If you only care about local CLI usage, you can ignore the API parts entirely.