# Transcription Pipeline

![Transcription Pipeline Architecture](Transcription%20Pipeline%20Architecture.png)

A speech-to-text pipeline built on [OpenAI Whisper](https://github.com/openai/whisper). It accepts audio files in any common format, transcribes them with per-segment timestamps, and can be run as a simple CLI tool or exposed as a REST API.

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

# Run as API server (optional)
uvicorn src.api:app --reload --port 8000
```

## Part 1: Transcription Pipeline

### Why Whisper?

I considered a few options: Google Speech-to-Text (accurate, but costs money per minute), Vosk (offline, lightweight, but noticeably worse on accented speech), and Whisper. Whisper wins for this use case because it's open-source, runs fully offline, handles multiple languages without configuration, and its `base` model is fast enough on CPU for most workloads. Larger models (`medium`, `large`) are available if accuracy matters more than speed.

### Audio Format Handling

Rather than writing format-specific parsers, all audio goes through FFmpeg before hitting Whisper. The normalisation step converts everything to **16kHz mono WAV** — which is what Whisper expects internally anyway. Doing this explicitly in `prepare_audio()` means:

1. Format errors surface early, with a clear message, instead of failing silently inside the model
2. The model always gets consistent input quality
3. Adding a new supported format is a one-line change to the `SUPPORTED_FORMATS` set

Supported formats: `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.mp4`, `.webm`, `.aac`

### Long File Handling

Files over 25MB are automatically chunked into 60-second segments with a 2-second overlap. The overlap is important — without it, words at chunk boundaries get dropped. After transcription, each chunk's timestamps are offset by the chunk's start time, so the final output has consistent absolute timestamps across the full recording.

The threshold is configurable via `long_file_threshold_mb` in `pipeline.run()`. An alternative would be using Whisper's built-in VAD, which works fine for files under ~30 minutes, but chunking is safer for longer recordings because it keeps memory usage bounded and is easy to parallelise later.

### Timestamped Segments

Whisper returns rich segment-level data with start/end times. The pipeline formats this into a clean structure: each segment is a dict with `start`, `end`, and `text` fields. Empty or whitespace-only segments are filtered out automatically. When processing chunked files, timestamps are adjusted to reflect absolute time in the original recording.

## Part 2: System Design Considerations

### 1. Handling Concurrent Uploads

For an API that accepts multiple uploads simultaneously, I'd use an **asynchronous job queue pattern**:

- The upload endpoint (`POST /transcribe`) accepts the file, generates a unique job ID, and returns immediately with `202 Accepted` status
- The actual transcription runs in a background thread pool (or a proper job queue like Celery + Redis in production)
- This prevents HTTP timeouts on long files and allows multiple transcriptions to run concurrently

The key insight: don't make the HTTP request wait for transcription to finish. Accept the upload, return a job ID, and let clients poll for results.

### 2. Storing Audio and Transcripts

I'd separate storage for raw audio files and transcript data:

- **Audio files**: Store in an object store (S3, Supabase Storage, etc.) organized by job ID (e.g., `{job_id}/audio.mp3`). Object stores handle large binaries well and can generate signed URLs for temporary access
- **Transcripts**: Store JSON segment data in the same object store for archival, plus lightweight job metadata (status, timestamps, segment count) in a database (Postgres) for fast querying
- **Cleanup**: Implement retention policies (e.g., delete audio after 30 days) to control storage costs

This separation lets you query job status quickly from the database while keeping heavy payloads in cheaper object storage.

### 3. Retrying Failed Transcriptions

Two layers of retry logic:

- **Immediate retries**: A retry decorator wraps the transcription function, catching exceptions and retrying up to 3 times with exponential backoff (2s, 4s, 8s). Handles transient issues like temporary disk I/O errors or model loading race conditions
- **Stall recovery**: A periodic background task scans for jobs stuck in `processing` longer than a timeout (e.g., 5 minutes). These are reset to `queued` for retry, catching cases where a worker crashed mid-job
- **After max retries**: Mark as `permanently_failed` and alert for human investigation

The goal is automatic recovery from transient issues while avoiding infinite retry loops on jobs that will never succeed.

### 4. Exposing as an API

A simple REST API with an async job pattern:

- `POST /transcribe` — Upload audio file, returns `202 Accepted` with `job_id`
- `GET /jobs/{job_id}` — Poll for status (`queued`, `processing`, `completed`, `failed`) and retrieve segments when complete
- `GET /health` — Liveness check for monitoring

Why async? Transcription can take minutes. Making clients wait would hit HTTP timeouts and waste connection resources. With async, clients upload, get a job ID immediately, then poll until completion.

Future enhancements could include webhook callbacks, pagination for very long transcripts, and authentication at the API gateway level.

## Project Structure

```
.
├── transcribe.py          # CLI entry point
├── src/
│   ├── pipeline.py        # Core transcription pipeline
│   ├── audio_utils.py     # Audio format conversion & chunking
│   ├── api.py             # Optional FastAPI wrapper
│   └── retry.py            # Optional retry logic
└── requirements.txt       # Python dependencies
```
