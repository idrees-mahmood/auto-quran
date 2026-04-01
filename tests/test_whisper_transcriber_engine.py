"""Tests for WhisperTranscriber engine parameter and whisperx pipeline."""

from unittest.mock import MagicMock

import pytest

import src.audio_processing_utils as apu


@pytest.fixture(autouse=True)
def clear_align_cache():
    apu._ALIGN_MODEL_CACHE.clear()
    yield
    apu._ALIGN_MODEL_CACHE.clear()


def _make_mock_whisperx(word_score: float = 0.95) -> MagicMock:
    mock_wx = MagicMock()
    mock_wx.load_audio.return_value = b"fake-audio"
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "segments": [{"start": 0.0, "end": 2.0, "text": "مرحبا", "words": []}],
        "language": "ar",
    }
    mock_wx.load_model.return_value = mock_model
    mock_wx.load_align_model.return_value = (MagicMock(), {"language": "ar"})
    aligned_words = [{"word": "مرحبا", "start": 0.0, "end": 0.5, "score": word_score}]
    mock_wx.align.return_value = {
        "segments": [{"start": 0.0, "end": 2.0, "text": "مرحبا", "words": aligned_words}],
        "word_segments": aligned_words,
        "language": "ar",
    }
    return mock_wx


def test_whisperx_score_normalised_to_probability(monkeypatch, tmp_path):
    mock_wx = _make_mock_whisperx(word_score=0.95)
    monkeypatch.setattr(apu, "_whisperx_lib", mock_wx)

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake-audio-bytes")

    t = apu.WhisperTranscriber(model_name="turbo", device="cpu", engine="whisperx")
    t.model = mock_wx.load_model.return_value

    result = t.transcribe(str(audio_file))

    for segment in result["segments"]:
        for word in segment.get("words", []):
            assert "probability" in word, "score should be renamed to probability"
            assert "score" not in word, "original score key should be removed"
            assert word["probability"] == 0.95


def test_whisperx_compute_type_float16_for_cuda():
    assert apu._whisperx_compute_type("cuda") == "float16"


def test_whisperx_compute_type_float16_for_mps():
    assert apu._whisperx_compute_type("mps") == "float16"


def test_whisperx_compute_type_int8_for_cpu():
    assert apu._whisperx_compute_type("cpu") == "int8"


def test_alignment_model_cached_on_second_call(monkeypatch, tmp_path):
    mock_wx = _make_mock_whisperx()
    monkeypatch.setattr(apu, "_whisperx_lib", mock_wx)

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake-audio-bytes")

    t = apu.WhisperTranscriber(model_name="turbo", device="cpu", engine="whisperx")
    t.model = mock_wx.load_model.return_value

    t.transcribe(str(audio_file))
    t.transcribe(str(audio_file))

    assert mock_wx.load_align_model.call_count == 1, \
        "load_align_model should only be called once due to cache"


def test_openai_whisper_path_does_not_call_whisperx(monkeypatch, tmp_path):
    mock_wx = MagicMock()
    monkeypatch.setattr(apu, "_whisperx_lib", mock_wx)

    mock_whisper_lib = MagicMock()
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"segments": [], "text": "", "language": "ar"}
    mock_whisper_lib.load_model.return_value = mock_model
    monkeypatch.setattr(apu, "whisper", mock_whisper_lib)

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake-audio-bytes")

    t = apu.WhisperTranscriber(model_name="turbo", device="cpu", engine="openai-whisper")
    t.transcribe(str(audio_file))

    mock_wx.load_audio.assert_not_called()
    mock_wx.align.assert_not_called()
    mock_model.transcribe.assert_called_once()
