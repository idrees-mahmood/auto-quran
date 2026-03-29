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


# Task 6: Integration smoke test

import json
from pathlib import Path
from src.audio_processing_utils import load_quran_text
from src.alignment_utils import AyahDetector


def test_dtw_mode_smoke_surah56():
    """DTW mode runs on the Surah 56 fixture and detects >=35 full matches."""
    base = Path(__file__).parent.parent
    trans_path = (
        base / "data/transcriptions/"
               "95ce00c87a6e56f5_Surah Al-Waqi_ah Verses 1 - 40  Shaykh Ali Salah O.json"
    )
    quran_path = base / "data/quran/quran.json"

    if not trans_path.exists():
        import pytest; pytest.skip("Transcription fixture not available")

    with open(trans_path) as f:
        data = json.load(f)

    from src.audio_processing_utils import TranscribedWord
    words = []
    for seg in data.get("transcription", data).get("segments", []):
        for w in seg.get("words", []):
            words.append(TranscribedWord(
                word=w.get("word", "").strip(),
                start=w.get("start", 0.0),
                end=w.get("end", 0.0),
                confidence=w.get("probability", w.get("confidence", 0.0)),
            ))

    quran_data = load_quran_text(str(quran_path))
    detector = AyahDetector(quran_data=quran_data, confidence_threshold=0.65)

    results = detector.detect_ayahs_from_transcription(
        transcribed_words=words,
        surah_hint=56,
        start_ayah=1,
        end_ayah=40,
        mode="dtw",
    )

    full_matches = [r for r in results if r.get("event_type") == "full"]
    assert len(full_matches) >= 35, (
        f"Expected >=35 full matches, got {len(full_matches)}. "
        f"All events: {[(r['ayah'], r.get('event_type')) for r in results]}"
    )


def _load_surah56_words():
    """Helper: load word list from the Surah 56 fixture."""
    base = Path(__file__).parent.parent
    trans_path = (
        base / "data/transcriptions/"
               "95ce00c87a6e56f5_Surah Al-Waqi_ah Verses 1 - 40  Shaykh Ali Salah O.json"
    )
    if not trans_path.exists():
        return None
    with open(trans_path) as f:
        data = json.load(f)
    from src.audio_processing_utils import TranscribedWord
    words = []
    for seg in data.get("transcription", data).get("segments", []):
        for w in seg.get("words", []):
            words.append(TranscribedWord(
                word=w.get("word", "").strip(),
                start=w.get("start", 0.0),
                end=w.get("end", 0.0),
                confidence=w.get("probability", w.get("confidence", 0.0)),
            ))
    return words


def test_preamble_skip_does_not_consume_ayah1():
    """Regression: _skip_opening_formulas must skip exactly 9 words (5 isti'adha
    + 4 basmallah) and must NOT consume ayah-1 words 'إذا وقعت الواقعة'."""
    words = _load_surah56_words()
    if words is None:
        import pytest; pytest.skip("Transcription fixture not available")

    quran_data = load_quran_text(
        str(Path(__file__).parent.parent / "data/quran/quran.json")
    )
    detector = AyahDetector(quran_data=quran_data)
    skip = detector._skip_opening_formulas(words)

    # Isti'adha (5) + basmallah (4) = 9. Must never exceed that.
    assert skip == 9, (
        f"Expected 9 preamble words skipped, got {skip}. "
        f"Word at skip pos: '{words[skip].word}'"
    )
    # The first non-preamble word must be the start of ayah 1
    first_content = words[skip].word
    assert "إذا" in first_content or "اذا" in first_content, (
        f"Expected ayah-1 start word 'إذا', got '{first_content}'"
    )


def test_dtw_no_end_ayah_finds_reasonable_matches():
    """Regression: DTW with end_ayah=None (full surah) must still find most of
    the recited ayahs via auto-range inference — no SKIP events returned."""
    words = _load_surah56_words()
    if words is None:
        import pytest; pytest.skip("Transcription fixture not available")

    quran_data = load_quran_text(
        str(Path(__file__).parent.parent / "data/quran/quran.json")
    )
    detector = AyahDetector(quran_data=quran_data, confidence_threshold=0.65)

    results = detector.detect_ayahs_from_transcription(
        transcribed_words=words,
        surah_hint=56,
        start_ayah=1,
        end_ayah=None,  # most common user case — must not break scoring
        mode="dtw",
    )

    # No SKIP events must leak out to callers
    skip_events = [r for r in results if r.get("event_type") == "skip"]
    assert skip_events == [], f"SKIP events must not be returned: {skip_events}"

    # Auto-range should still find a substantial portion of the recitation
    full_matches = [r for r in results if r.get("event_type") == "full"]
    assert len(full_matches) >= 25, (
        f"Expected >=25 full matches with end_ayah=None, got {len(full_matches)}. "
        f"Events: {[(r['ayah'], r.get('event_type')) for r in results]}"
    )


# Task 1 regression: wider window covers intra-ayah phrase repetition
def test_wider_window_covers_intra_repeat():
    """
    Ayah 2 has 6 reference words. The transcription doubles every word (each
    word repeated twice consecutively), giving 12 transcription words for ayah 2.

    With max_w = ref_count + 3 = 9 the matrix can only evaluate windows up to
    width 9. The best score reachable within that cap is ~0.776, which falls
    below confidence_threshold=0.85, so no MATCH event is produced for ayah 2.

    With max_w = ref_count * 2 = 12 the matrix scores the full 12-word window
    (~0.906 >= 0.85) and the DP produces a full MATCH event for ayah 2.

    This test verifies that the fix (ref_count * 2) is required to detect the
    ayah when the reciter doubles each word.
    """
    from src.dtw_alignment import (
        build_banded_similarity_matrix, run_dp_alignment,
        build_recitation_events, DTWConfig,
    )
    from src.audio_processing_utils import ArabicNormalizer

    normalizer = ArabicNormalizer()

    # Ayah 2 reference: 6 distinct words
    ayah2_ref = ["ا", "ب", "ت", "ث", "ج", "ح"]

    corpus = {
        1: {"norm_words": ["خ", "د", "ذ"],
            "normalized": "خ د ذ", "count": 3},
        2: {"norm_words": ayah2_ref,
            "normalized": " ".join(ayah2_ref), "count": 6},
    }

    # Transcription: ayah1 clean (3 words) | ayah2 with each word doubled (12 words)
    # Total: 3 + 12 = 15 words
    raw = ["خ", "د", "ذ"]  # ayah 1 (clean)
    for word in ayah2_ref:
        raw.extend([word, word])  # each ayah2 word doubled

    t = 0.0
    trans = []
    for word in raw:
        trans.append(_w(word, t, t + 0.5))
        t += 0.5

    # confidence_threshold=0.85: old cap (ref_count+3=9) yields max score ~0.776
    # which is below 0.85 -> no MATCH for ayah 2.
    # New cap (ref_count*2=12) yields ~0.906 >= 0.85 -> MATCH produced.
    config = DTWConfig(band_width_min=15, confidence_threshold=0.85)
    matrix = build_banded_similarity_matrix(
        words=trans, ayah_corpus=corpus, ayah_range=(1, 2),
        normalizer=normalizer, config=config,
    )
    path = run_dp_alignment(
        words=trans, ayah_corpus=corpus, ayah_range=(1, 2),
        similarity_matrix=matrix, config=config,
    )
    events = build_recitation_events(
        path=path, words=trans, surah=99, ayah_corpus=corpus,
        normalizer=normalizer, config=config,
    )

    ayah2_events = [e for e in events if e.ayah == 2 and e.event_type == "full"]
    assert ayah2_events, (
        "Expected at least one full event for ayah 2. "
        "The wider window (ref_count*2) is required to score the doubled-word recitation."
    )


# Task 2 regression: noise second-pass emits multiple repetition events
def test_noise_second_pass_splits_block_repetition():
    """
    Ayahs 1 and 2 are matched cleanly in the first pass. A trailing noise
    region contains first ayah-1 content then ayah-2 content.
    The greedy second-pass must emit exactly 2 repetition events, one for
    each previously-seen ayah, with non-overlapping word_indices.
    """
    from src.dtw_alignment import (
        build_banded_similarity_matrix, run_dp_alignment,
        build_recitation_events, DTWConfig,
    )
    from src.audio_processing_utils import ArabicNormalizer

    normalizer = ArabicNormalizer()

    corpus = {
        1: {"norm_words": ["ا", "ب", "ت"],
            "normalized": "ا ب ت", "count": 3},
        2: {"norm_words": ["ج", "د", "ه"],
            "normalized": "ج د ه", "count": 3},
    }

    # Transcription: ayah1 | ayah2 | ayah1 again | ayah2 again
    raw = ["ا", "ب", "ت",   # ayah 1 first pass
           "ج", "د", "ه",   # ayah 2 first pass
           "ا", "ب", "ت",   # ayah 1 repeated (noise region)
           "ج", "د", "ه"]   # ayah 2 repeated (noise region)
    t = 0.0
    trans = []
    for word in raw:
        trans.append(_w(word, t, t + 0.5))
        t += 0.5

    config = DTWConfig(band_width_min=12, confidence_threshold=0.55)
    matrix = build_banded_similarity_matrix(
        words=trans, ayah_corpus=corpus, ayah_range=(1, 2),
        normalizer=normalizer, config=config,
    )
    path = run_dp_alignment(
        words=trans, ayah_corpus=corpus, ayah_range=(1, 2),
        similarity_matrix=matrix, config=config,
    )
    events = build_recitation_events(
        path=path, words=trans, surah=99, ayah_corpus=corpus,
        normalizer=normalizer, config=config,
    )

    rep_events = [e for e in events if e.event_type == "repetition"]
    assert len(rep_events) == 2, (
        f"Expected exactly 2 repetition events, got {len(rep_events)}. "
        f"All events: {[(e.ayah, e.event_type, e.word_indices) for e in events]}"
    )

    # Both ayahs should appear as repetitions
    rep_ayahs = {e.ayah for e in rep_events}
    assert 1 in rep_ayahs, f"Ayah 1 not in repetitions: {rep_ayahs}"
    assert 2 in rep_ayahs, f"Ayah 2 not in repetitions: {rep_ayahs}"

    # word_indices must not overlap
    sorted_reps = sorted(rep_events, key=lambda e: e.word_indices[0])
    for a, b in zip(sorted_reps, sorted_reps[1:]):
        assert a.word_indices[1] <= b.word_indices[0], (
            f"Overlapping repetition events: {a.word_indices} and {b.word_indices}"
        )
