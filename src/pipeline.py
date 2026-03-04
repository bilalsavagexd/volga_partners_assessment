"""
pipeline.py — Core transcription pipeline using OpenAI Whisper.

Accepts a local audio file, normalises it, runs it through Whisper,
and returns a list of timestamped segments.
"""

import os
import logging
from audio_utils import prepare_audio, chunk_audio

logger = logging.getLogger(__name__)


class TranscriptionPipeline:
    def __init__(self, model_size: str = "base"):
        # Import here so the module is importable even without whisper installed
        import whisper
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
        self.model_size = model_size

    def run(self, audio_path: str, long_file_threshold_mb: int = 25) -> list[dict]:
        """
        Transcribe an audio file and return timestamped segments.

        Automatically switches to chunked processing for large files
        so we don't blow up memory on a 2-hour recording.
        """
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info(f"Processing file: {audio_path} ({file_size_mb:.1f} MB)")

        if file_size_mb > long_file_threshold_mb:
            logger.info("Large file detected — switching to chunked mode")
            return self._run_chunked(audio_path)
        else:
            return self._run_single(audio_path)

    def _run_single(self, audio_path: str) -> list[dict]:
        prepared = prepare_audio(audio_path)
        result = self.model.transcribe(prepared, verbose=False)
        return self._format_segments(result["segments"], offset=0.0)

    def _run_chunked(self, audio_path: str) -> list[dict]:
        chunks = chunk_audio(audio_path)
        all_segments = []

        for chunk_path, time_offset in chunks:
            logger.info(f"Transcribing chunk at offset {time_offset:.1f}s")
            result = self.model.transcribe(chunk_path, verbose=False)
            all_segments.extend(self._format_segments(result["segments"], offset=time_offset))

        return all_segments

    def _format_segments(self, raw_segments: list, offset: float) -> list[dict]:
        return [
            {
                "start": round(seg["start"] + offset, 2),
                "end": round(seg["end"] + offset, 2),
                "text": seg["text"].strip(),
            }
            for seg in raw_segments
            if seg["text"].strip()  # skip empty segments
        ]
