"""
tests/test_pipeline.py — Unit tests using mock audio data.

We mock out the Whisper model so tests run fast without a GPU
and without needing actual audio files.
"""

import os
import sys
import json
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ─── Mock Whisper response ───────────────────────────────────────────────────

MOCK_SEGMENTS = [
    {"start": 0.0,  "end": 3.2,  "text": "  Hey, welcome to the demo.  "},
    {"start": 3.2,  "end": 7.8,  "text": "Today we walk through the pipeline."},
    {"start": 7.8,  "end": 12.1, "text": "  Let's talk about audio chunking.  "},
]

MOCK_WHISPER_RESULT = {"segments": MOCK_SEGMENTS, "text": "full transcript here"}


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestTranscriptionPipeline:
    @patch("pipeline.prepare_audio", return_value="/tmp/prepared.wav")
    @patch("whisper.load_model")
    def test_basic_transcription_returns_segments(self, mock_load, mock_prepare):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = MOCK_WHISPER_RESULT
        mock_load.return_value = mock_model

        from pipeline import TranscriptionPipeline
        pipeline = TranscriptionPipeline()
        result = pipeline.run.__wrapped__(pipeline, "/fake/audio.wav") if hasattr(pipeline.run, '__wrapped__') else pipeline._run_single("/fake/audio.wav")

        assert len(result) == 3
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 3.2
        assert result[0]["text"] == "Hey, welcome to the demo."  # stripped

    @patch("pipeline.prepare_audio", return_value="/tmp/prepared.wav")
    @patch("whisper.load_model")
    def test_empty_segments_are_filtered(self, mock_load, mock_prepare):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "  "},  # whitespace only
                {"start": 1.0, "end": 2.5, "text": "Real content here"},
            ]
        }
        mock_load.return_value = mock_model

        from pipeline import TranscriptionPipeline
        pipeline = TranscriptionPipeline()
        result = pipeline._run_single("/fake/audio.wav")

        assert len(result) == 1
        assert result[0]["text"] == "Real content here"

    @patch("pipeline.chunk_audio")
    @patch("whisper.load_model")
    def test_chunked_mode_applies_time_offset(self, mock_load, mock_chunk):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 5.0, "text": "Hello"}]
        }
        mock_load.return_value = mock_model

        # Simulate two chunks: one at 0s and one at 60s
        mock_chunk.return_value = [("/tmp/chunk_0.wav", 0.0), ("/tmp/chunk_1.wav", 60.0)]

        from pipeline import TranscriptionPipeline
        pipeline = TranscriptionPipeline()
        result = pipeline._run_chunked("/fake/long_audio.wav")

        assert len(result) == 2
        assert result[0]["start"] == 0.0
        assert result[1]["start"] == 60.0  # offset applied


class TestAudioUtils:
    def test_unsupported_format_raises_value_error(self):
        from audio_utils import prepare_audio
        with pytest.raises(ValueError, match="Unsupported format"):
            prepare_audio("/fake/audio.xyz")

    def test_chunk_list_has_correct_offsets(self, tmp_path):
        from pydub import AudioSegment
        from audio_utils import chunk_audio

        # Create a 3-minute silent WAV for testing
        silence = AudioSegment.silent(duration=180_000)
        audio_path = str(tmp_path / "test.wav")
        silence.export(audio_path, format="wav")

        chunks = chunk_audio(audio_path, chunk_ms=60_000, overlap_ms=2_000)

        # 3 minutes with 60s chunks and 2s overlap → 4 chunks
        assert len(chunks) == 4
        assert chunks[0][1] == 0.0     # first chunk starts at 0s
        assert chunks[1][1] == 58.0    # second chunk starts at 58s (60 - 2 overlap)


class TestRetryLogic:
    def test_retry_decorator_retries_on_failure(self):
        from retry import with_retry

        call_count = 0

        @with_retry
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Transient error")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count == 3

    def test_retry_raises_after_max_attempts(self):
        from retry import with_retry, MAX_RETRIES

        @with_retry
        def always_fails():
            raise RuntimeError("Permanent failure")

        with pytest.raises(RuntimeError, match="Permanent failure"):
            always_fails()
