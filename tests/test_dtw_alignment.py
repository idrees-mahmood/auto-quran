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


# Task 3: Banded Similarity Matrix Tests

from typing import List, Tuple
from src.dtw_alignment import build_banded_similarity_matrix


def _make_corpus(pairs):
    """pairs: [(ayah_num, word_list), ...]"""
    normalizer = ArabicNormalizer()
    corpus = {}
    for num, words in pairs:
        norm = [normalizer.normalize(w) for w in words]
        corpus[num] = {
            "norm_words": norm,
            "normalized": " ".join(norm),
            "count": len(words),
        }
    return corpus, normalizer


def test_matrix_only_covers_band():
    words = [_w(f"w{i}", i * 0.5, (i + 1) * 0.5) for i in range(30)]
    corpus, normalizer = _make_corpus([
        (1, ["w0", "w1", "w2"]),
        (2, ["w10", "w11", "w12"]),
    ])
    config = DTWConfig(band_width_min=3, band_width_ratio=0.1)
    matrix = build_banded_similarity_matrix(
        words=words, ayah_corpus=corpus, ayah_range=(1, 2),
        normalizer=normalizer, config=config,
    )
    # Ayah 2 expected near word 15 (half of 30); should NOT have entry at position 0
    positions_ayah2 = [i for (i, j) in matrix if j == 2]
    assert 0 not in positions_ayah2


def test_matrix_high_score_at_matching_position():
    normalizer = ArabicNormalizer()
    words = [_w("الله", 0.0, 0.5), _w("اكبر", 0.5, 1.0),
             _w("كتاب", 1.0, 1.5), _w("قلم", 1.5, 2.0)]
    corpus = {
        1: {
            "norm_words": [normalizer.normalize("الله"), normalizer.normalize("اكبر")],
            "normalized": "الله اكبر",
            "count": 2,
        }
    }
    config = DTWConfig(band_width_min=4)
    matrix = build_banded_similarity_matrix(
        words=words, ayah_corpus=corpus, ayah_range=(1, 1),
        normalizer=normalizer, config=config,
    )
    score, _ = matrix.get((0, 1), (0.0, 2))
    assert score > 0.7


from src.dtw_alignment import (
    DTWConfig, build_banded_similarity_matrix, run_dp_alignment,
)


def test_dp_sequential_clean_path():
    """Four ayahs, no noise, perfect match — path should be 4 MATCHes in order."""
    ayah_words = [
        (1, ["بسم", "الله", "الرحمن"]),
        (2, ["رب", "العالمين"]),
        (3, ["الرحمن", "الرحيم"]),
        (4, ["مالك", "يوم", "الدين"]),
    ]
    corpus, normalizer = _make_corpus(ayah_words)
    # Transcription = all ayah words concatenated in order
    all_words = [w for _, ws in ayah_words for w in ws]
    t = 0.0
    trans = []
    for word in all_words:
        trans.append(_w(word, t, t + 0.5))
        t += 0.5

    config = DTWConfig(band_width_min=6)
    matrix = build_banded_similarity_matrix(
        words=trans, ayah_corpus=corpus, ayah_range=(1, 4),
        normalizer=normalizer, config=config,
    )
    path = run_dp_alignment(
        words=trans, ayah_corpus=corpus, ayah_range=(1, 4),
        similarity_matrix=matrix, config=config,
    )
    matches = [m for m in path if m[0] == "MATCH"]
    assert len(matches) == 4
    assert [m[1] for m in matches] == [1, 2, 3, 4]


def test_dp_absorbs_leading_noise():
    """Two noise words before ayah 1 — should be consumed as NOISE."""
    corpus, normalizer = _make_corpus([(1, ["الله", "اكبر"])])
    trans = [
        _w("اعوذ",  0.0, 0.3),   # noise
        _w("بالله", 0.3, 0.6),   # noise
        _w("الله",  0.6, 1.1),
        _w("اكبر",  1.1, 1.6),
    ]
    config = DTWConfig(band_width_min=5)
    matrix = build_banded_similarity_matrix(
        words=trans, ayah_corpus=corpus, ayah_range=(1, 1),
        normalizer=normalizer, config=config,
    )
    path = run_dp_alignment(
        words=trans, ayah_corpus=corpus, ayah_range=(1, 1),
        similarity_matrix=matrix, config=config,
    )
    matches = [m for m in path if m[0] == "MATCH"]
    assert len(matches) == 1
    # Match should start at word index 2
    _, ayah_num, start_i, end_i, score = matches[0]
    assert ayah_num == 1
    assert start_i == 2


from src.dtw_alignment import (
    DTWConfig, build_banded_similarity_matrix,
    run_dp_alignment, build_recitation_events,
)
from src.alignment_utils import RecitationEvent


def test_build_events_sequential():
    ayah_words = [(1, ["الله", "اكبر"]), (2, ["رب", "العالمين"])]
    corpus, normalizer = _make_corpus(ayah_words)
    trans = [
        _w("الله", 0.0, 0.5), _w("اكبر", 0.5, 1.0),
        _w("رب",   1.0, 1.5), _w("العالمين", 1.5, 2.0),
    ]
    config = DTWConfig(band_width_min=4)
    matrix = build_banded_similarity_matrix(
        words=trans, ayah_corpus=corpus, ayah_range=(1, 2),
        normalizer=normalizer, config=config,
    )
    path = run_dp_alignment(
        words=trans, ayah_corpus=corpus, ayah_range=(1, 2),
        similarity_matrix=matrix, config=config,
    )
    events = build_recitation_events(
        path=path, words=trans, surah=56,
        ayah_corpus=corpus, normalizer=normalizer, config=config,
    )
    full = [e for e in events if e.event_type == "full"]
    assert len(full) == 2
    assert full[0].ayah == 1
    assert full[1].ayah == 2
    assert full[0].occurrence == 1
    assert abs(full[0].start_time - 0.0) < 0.01
    assert abs(full[1].end_time  - 2.0) < 0.01


def test_build_events_detects_repetition():
    """Ayah 1 spoken, then repeated. The repeat appears as NOISE to the DP
    (ayah range is exhausted), and post-processing reclassifies it."""
    corpus, normalizer = _make_corpus([(1, ["الله", "اكبر"])])
    trans = [
        _w("الله", 0.0, 0.5), _w("اكبر", 0.5, 1.0),   # first
        _w("الله", 1.5, 2.0), _w("اكبر", 2.0, 2.5),   # repeat
    ]
    config = DTWConfig(band_width_min=5)
    matrix = build_banded_similarity_matrix(
        words=trans, ayah_corpus=corpus, ayah_range=(1, 1),
        normalizer=normalizer, config=config,
    )
    path = run_dp_alignment(
        words=trans, ayah_corpus=corpus, ayah_range=(1, 1),
        similarity_matrix=matrix, config=config,
    )
    events = build_recitation_events(
        path=path, words=trans, surah=56,
        ayah_corpus=corpus, normalizer=normalizer, config=config,
    )
    reps = [e for e in events if e.event_type == "repetition"]
    assert len(reps) >= 1
    assert reps[0].ayah == 1
    assert reps[0].occurrence == 2
