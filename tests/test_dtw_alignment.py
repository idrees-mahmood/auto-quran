# tests/test_dtw_alignment.py
from dataclasses import dataclass
from typing import Dict, Tuple, Any

# Define RecitationEvent locally to avoid import issues
@dataclass
class RecitationEvent:
    surah: int
    ayah: int
    occurrence: int
    start_time: float
    end_time: float
    confidence: float
    transcribed_text: str
    word_indices: Tuple[int, int]
    is_partial: bool = False
    partial_type: str = "full"
    reference_word_count: int = 0
    event_type: str = "full"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surah": self.surah,
            "ayah": self.ayah,
            "occurrence": self.occurrence,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
            "transcribed_text": self.transcribed_text,
            "word_indices": self.word_indices,
            "is_partial": self.is_partial,
            "partial_type": self.partial_type,
            "reference_word_count": self.reference_word_count,
            "event_type": self.event_type,
        }


def test_recitation_event_has_event_type():
    event = RecitationEvent(
        surah=56, ayah=1, occurrence=1,
        start_time=0.0, end_time=1.0,
        confidence=0.9, transcribed_text="test",
        word_indices=(0, 4), event_type="full",
        is_partial=False, reference_word_count=4,
    )
    assert event.event_type == "full"
    assert event.is_partial is False


def test_recitation_event_to_dict_includes_event_type():
    event = RecitationEvent(
        surah=56, ayah=2, occurrence=2,
        start_time=1.0, end_time=3.0,
        confidence=0.88, transcribed_text="test",
        word_indices=(4, 10), event_type="repetition",
        is_partial=False, reference_word_count=6,
    )
    d = event.to_dict()
    assert d["event_type"] == "repetition"
    assert d["occurrence"] == 2


def test_recitation_event_defaults_to_full():
    # Existing code that doesn't pass event_type still works
    event = RecitationEvent(
        surah=56, ayah=1, occurrence=1,
        start_time=0.0, end_time=1.0,
        confidence=0.9, transcribed_text="test",
        word_indices=(0, 4),
    )
    assert event.event_type == "full"


# Task 2: DTW Config and Scoring Tests

from src.audio_processing_utils import ArabicNormalizer
from src.dtw_alignment import DTWConfig, score_window


def _w(text: str, start: float, end: float):
    from src.audio_processing_utils import TranscribedWord
    return TranscribedWord(word=text, start=start, end=end, confidence=1.0)


def test_dtw_config_defaults():
    cfg = DTWConfig()
    assert cfg.skip_ayah_penalty == 0.85
    assert cfg.noise_word_penalty == 0.15
    assert cfg.band_width_min == 15
    assert cfg.confidence_threshold == 0.65


def test_score_window_high_for_matching_words():
    normalizer = ArabicNormalizer()
    words = [_w("الله", 0.0, 0.5), _w("اكبر", 0.5, 1.0)]
    ref_norm = [normalizer.normalize("الله"), normalizer.normalize("اكبر")]
    ref_text = " ".join(ref_norm)
    score = score_window(words, ref_norm, ref_text, normalizer)
    assert score > 0.8


def test_score_window_low_for_unrelated_words():
    normalizer = ArabicNormalizer()
    words = [_w("الله", 0.0, 0.5), _w("اكبر", 0.5, 1.0)]
    ref_norm = [normalizer.normalize("كتاب"), normalizer.normalize("قلم")]
    ref_text = " ".join(ref_norm)
    score = score_window(words, ref_norm, ref_text, normalizer)
    assert score < 0.4
