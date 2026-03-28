# DTW-Based Ayah Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the greedy sequential alignment algorithm with a globally-optimal DTW-based engine that handles stumbles, repetitions, unusual pauses, and skips without cascade failure, and surface annotated results in an improved Streamlit UI.

**Architecture:** Build a new standalone module `src/dtw_alignment.py` containing the similarity matrix, DP, and event builder. Wire it into the existing `AyahDetector` class as a new `mode="dtw"` option on the existing `detect_ayahs_from_transcription()` entry point. Replace the detected-ayahs flat table in `app.py` with a timeline bar + annotated card list.

**Tech Stack:** Python, rapidfuzz (already in requirements), Streamlit `st.components.v1.html` for the timeline, existing `ArabicNormalizer` and word-scoring functions from `src/alignment_utils.py`.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/dtw_alignment.py` | **Create** | DTWConfig, score_window, similarity matrix, DP, traceback, event builder |
| `src/alignment_utils.py` | **Modify** | Add `event_type` to `RecitationEvent`; add `detect_ayahs_dtw()`; add `mode` param |
| `app.py` | **Modify** | Timeline + card list UI, DTW mode selector, exclusion filtering |
| `tests/test_dtw_alignment.py` | **Create** | Unit tests for DTW engine |

---

## Task 1: Add `event_type` to `RecitationEvent`

**Files:**
- Modify: `src/alignment_utils.py:35-68`
- Create: `tests/test_dtw_alignment.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dtw_alignment.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alignment_utils import RecitationEvent


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_dtw_alignment.py::test_recitation_event_has_event_type -v
```

Expected: `FAIL — TypeError: RecitationEvent.__init__() got an unexpected keyword argument 'event_type'`

- [ ] **Step 3: Add `event_type` field to RecitationEvent in `src/alignment_utils.py`**

The current dataclass ends around line 68. Replace the full dataclass body with:

```python
@dataclass
class RecitationEvent:
    """
    A single occurrence of an ayah being recited.
    """
    surah: int
    ayah: int
    occurrence: int          # 1 = first time, 2+ = repetition
    start_time: float
    end_time: float
    confidence: float
    transcribed_text: str
    word_indices: Tuple[int, int]
    is_partial: bool = False
    partial_type: str = "full"          # kept for backward compat
    reference_word_count: int = 0
    event_type: str = "full"            # "full" | "partial" | "repetition" | "skip"

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
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_dtw_alignment.py -v
python -m pytest tests/ -v
```

Expected: All 3 new tests PASS. All existing tests PASS (`event_type` defaults to `"full"` so no existing code breaks).

- [ ] **Step 5: Commit**

```bash
git add src/alignment_utils.py tests/test_dtw_alignment.py
git commit -m "feat: add event_type field to RecitationEvent (backward-compatible)"
```

---

## Task 2: Create `src/dtw_alignment.py` — config and scoring

**Files:**
- Create: `src/dtw_alignment.py`
- Modify: `tests/test_dtw_alignment.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_dtw_alignment.py`:

```python
from src.audio_processing_utils import TranscribedWord, ArabicNormalizer
from src.dtw_alignment import DTWConfig, score_window


def _w(text: str, start: float, end: float) -> TranscribedWord:
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
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_dtw_alignment.py::test_dtw_config_defaults -v
```

Expected: `FAIL — ModuleNotFoundError: No module named 'src.dtw_alignment'`

- [ ] **Step 3: Create `src/dtw_alignment.py`**

```python
"""
DTW-based globally optimal ayah alignment.

Replaces the greedy sequential algorithm with dynamic programming that
finds the globally optimal mapping from transcription word positions to
Quran ayahs. Immune to cascade failure because all decisions are made
simultaneously over the full matrix rather than greedily left-to-right.
"""

import difflib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.audio_processing_utils import ArabicNormalizer, TranscribedWord

try:
    from rapidfuzz import fuzz
    _RAPIDFUZZ = True
except ImportError:
    _RAPIDFUZZ = False

logger = logging.getLogger(__name__)

# DP move type constants
_MATCH = "MATCH"
_SKIP_AYAH = "SKIP_AYAH"
_NOISE = "NOISE"

INF = float("inf")


@dataclass
class DTWConfig:
    """Tunable parameters for DTW alignment."""
    band_width_ratio: float = 0.15       # Band width as fraction of total expected words
    band_width_min: int = 15             # Minimum band width in words
    skip_ayah_penalty: float = 0.85     # DP cost for skipping an ayah (high — last resort)
    noise_word_penalty: float = 0.15    # DP cost per noise word consumed
    max_noise_run: int = 8              # Max consecutive noise words before MATCH required
    partial_confidence_threshold: float = 0.55   # Below this = partial event
    confidence_threshold: float = 0.65  # Minimum score for a MATCH to be accepted


def _fuzzy_pair(a: str, b: str) -> float:
    """Similarity between two normalised Arabic words (0–1)."""
    if _RAPIDFUZZ:
        return fuzz.ratio(a, b) / 100.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def score_window(
    window_words: List[TranscribedWord],
    ref_norm_words: List[str],
    ref_norm_text: str,
    normalizer: ArabicNormalizer,
) -> float:
    """
    Score how well a window of transcribed words matches a reference ayah.

    Uses 70 % word-level fuzzy alignment + 30 % string-level fuzzy match,
    matching the existing scoring approach in alignment_utils.py.

    Args:
        window_words:   Transcribed words in this window.
        ref_norm_words: Normalised reference words for the ayah.
        ref_norm_text:  Full normalised reference string (for string-level match).
        normalizer:     ArabicNormalizer instance.

    Returns:
        Score in [0, 1]. Higher = better match.
    """
    if not window_words or not ref_norm_words:
        return 0.0

    window_norm = [normalizer.normalize(w.word) for w in window_words]

    # Word-level alignment (order-insensitive fuzzy matching)
    matched, used = 0, set()
    for tw in window_norm:
        best, best_idx = 0.0, -1
        for ri, rw in enumerate(ref_norm_words):
            if ri in used:
                continue
            s = _fuzzy_pair(tw, rw)
            if s > best:
                best, best_idx = s, ri
        if best > 0.6 and best_idx >= 0:
            matched += 1
            used.add(best_idx)
    word_score = matched / len(ref_norm_words)

    # String-level fuzzy score
    window_text = " ".join(window_norm)
    if _RAPIDFUZZ:
        str_score = fuzz.ratio(window_text, ref_norm_text) / 100.0
    else:
        str_score = difflib.SequenceMatcher(None, window_text, ref_norm_text).ratio()

    return 0.7 * word_score + 0.3 * str_score
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_dtw_alignment.py::test_dtw_config_defaults \
                 tests/test_dtw_alignment.py::test_score_window_high_for_matching_words \
                 tests/test_dtw_alignment.py::test_score_window_low_for_unrelated_words -v
```

Expected: All 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dtw_alignment.py tests/test_dtw_alignment.py
git commit -m "feat: create dtw_alignment.py with DTWConfig and score_window"
```

---

## Task 3: Banded similarity matrix

**Files:**
- Modify: `src/dtw_alignment.py`
- Modify: `tests/test_dtw_alignment.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_dtw_alignment.py`:

```python
from src.dtw_alignment import DTWConfig, score_window, build_banded_similarity_matrix


def _make_corpus(pairs: List[Tuple[int, List[str]]]):
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
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_dtw_alignment.py::test_matrix_only_covers_band -v
```

Expected: `FAIL — ImportError: cannot import name 'build_banded_similarity_matrix'`

- [ ] **Step 3: Add `build_banded_similarity_matrix` to `src/dtw_alignment.py`**

Append to `src/dtw_alignment.py`:

```python
def build_banded_similarity_matrix(
    words: List[TranscribedWord],
    ayah_corpus: Dict[int, Dict],
    ayah_range: Tuple[int, int],
    normalizer: ArabicNormalizer,
    config: DTWConfig,
) -> Dict[Tuple[int, int], Tuple[float, int]]:
    """
    Build a sparse similarity matrix for (word_position, ayah_num) pairs.

    Only fills cells within a diagonal band around each ayah's expected
    position (derived from the ayah range word counts). Cells outside the
    band are never evaluated, ruling out long-range implausible alignments.

    Args:
        words:       Full list of transcribed words.
        ayah_corpus: {ayah_num: {"norm_words": [...], "normalized": str, "count": int}}
        ayah_range:  (start_ayah, end_ayah) inclusive.
        normalizer:  ArabicNormalizer instance.
        config:      DTWConfig instance.

    Returns:
        Dict mapping (word_pos, ayah_num) -> (best_score, best_window_size).
        Only in-band cells are present.
    """
    start_ayah, end_ayah = ayah_range
    ayahs = [j for j in range(start_ayah, end_ayah + 1) if j in ayah_corpus]
    M = len(words)
    if not ayahs or M == 0:
        return {}

    total_ref_words = sum(ayah_corpus[j]["count"] for j in ayahs)
    if total_ref_words == 0:
        return {}

    band_width = max(config.band_width_min,
                     int(total_ref_words * config.band_width_ratio))
    logger.debug(f"Similarity matrix: band_width={band_width}, "
                 f"total_ref={total_ref_words}, M={M}")

    matrix: Dict[Tuple[int, int], Tuple[float, int]] = {}
    cumulative_words = 0

    for j in ayahs:
        ref = ayah_corpus[j]
        ref_count = ref["count"]
        ref_norm_words = ref["norm_words"]
        ref_norm_text = ref["normalized"]

        expected_pos = int((cumulative_words / total_ref_words) * M)
        cumulative_words += ref_count

        band_start = max(0, expected_pos - band_width)
        band_end = min(M - 1, expected_pos + band_width)

        for i in range(band_start, band_end + 1):
            best_score, best_w = 0.0, ref_count
            min_w = max(1, ref_count - 2)
            max_w = min(M - i, ref_count + 3)
            for w_size in range(min_w, max_w + 1):
                s = score_window(words[i: i + w_size],
                                 ref_norm_words, ref_norm_text, normalizer)
                if s > best_score:
                    best_score, best_w = s, w_size
            matrix[(i, j)] = (best_score, best_w)

    logger.debug(f"Matrix built: {len(matrix)} cells for {len(ayahs)} ayahs")
    return matrix
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_dtw_alignment.py::test_matrix_only_covers_band \
                 tests/test_dtw_alignment.py::test_matrix_high_score_at_matching_position -v
```

Expected: Both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dtw_alignment.py tests/test_dtw_alignment.py
git commit -m "feat: add build_banded_similarity_matrix"
```

---

## Task 4: DP alignment

**Files:**
- Modify: `src/dtw_alignment.py`
- Modify: `tests/test_dtw_alignment.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_dtw_alignment.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_dtw_alignment.py::test_dp_sequential_clean_path -v
```

Expected: `FAIL — ImportError: cannot import name 'run_dp_alignment'`

- [ ] **Step 3: Add `run_dp_alignment` to `src/dtw_alignment.py`**

Append to `src/dtw_alignment.py`:

```python
def run_dp_alignment(
    words: List[TranscribedWord],
    ayah_corpus: Dict[int, Dict],
    ayah_range: Tuple[int, int],
    similarity_matrix: Dict[Tuple[int, int], Tuple[float, int]],
    config: DTWConfig,
) -> List[Tuple]:
    """
    Find the globally optimal alignment path via dynamic programming.

    States: (word_position i, ayah_column k)
    Transitions:
      MATCH(i→i+w, k→k+1)   consume w words, advance ayah (w from matrix)
      SKIP_AYAH(i→i, k→k+1) skip ayah without consuming words (high penalty)
      NOISE(i→i+n, k→k)     consume n words without advancing ayah

    Args:
        words:             Transcribed words.
        ayah_corpus:       {ayah_num: {"count": int, ...}}
        ayah_range:        (start_ayah, end_ayah) inclusive.
        similarity_matrix: Output of build_banded_similarity_matrix.
        config:            DTWConfig.

    Returns:
        List of move tuples (in chronological order):
          ("MATCH",    ayah_num, start_idx, end_idx, score)
          ("SKIP_AYAH", ayah_num)
          ("NOISE",    start_idx, end_idx)
    """
    start_ayah, end_ayah = ayah_range
    ayahs = [j for j in range(start_ayah, end_ayah + 1) if j in ayah_corpus]
    if not ayahs:
        return []

    M = len(words)
    N = len(ayahs)

    # dp[i][k]     = minimum cost to reach (word_pos=i, ayah_col=k)
    # parent[i][k] = (prev_i, prev_k, move_tuple)
    dp = [[INF] * (N + 1) for _ in range(M + 1)]
    parent: List[List[Optional[Tuple]]] = [[None] * (N + 1) for _ in range(M + 1)]
    dp[0][0] = 0.0

    for i in range(M + 1):
        for k in range(N + 1):
            if dp[i][k] == INF:
                continue
            cost = dp[i][k]
            j = ayahs[k] if k < N else None  # current ayah (None if past end)

            # --- MATCH ---
            if j is not None:
                cell = similarity_matrix.get((i, j))
                if cell is not None:
                    score, w_size = cell
                    if score >= config.confidence_threshold:
                        ni, nk = i + w_size, k + 1
                        if ni <= M:
                            c = cost + (1.0 - score)
                            if c < dp[ni][nk]:
                                dp[ni][nk] = c
                                parent[ni][nk] = (
                                    i, k, (_MATCH, j, i, ni, score)
                                )

            # --- SKIP_AYAH ---
            if j is not None:
                nk = k + 1
                c = cost + config.skip_ayah_penalty
                if c < dp[i][nk]:
                    dp[i][nk] = c
                    parent[i][nk] = (i, k, (_SKIP_AYAH, j))

            # --- NOISE (consume 1..max_noise_run words) ---
            if i < M:
                for n in range(1, config.max_noise_run + 1):
                    ni = i + n
                    if ni > M:
                        break
                    c = cost + config.noise_word_penalty * n
                    if c < dp[ni][k]:
                        dp[ni][k] = c
                        parent[ni][k] = (i, k, (_NOISE, i, ni))

    # Find best terminal: prefer (M, N) but accept trailing skips/noise
    best_cost, best_end = INF, (M, N)
    for ek in range(max(0, N - 3), N + 1):
        for ei in range(max(0, M - 10), M + 1):
            if dp[ei][ek] < best_cost:
                best_cost, best_end = dp[ei][ek], (ei, ek)

    # Traceback
    moves: List[Tuple] = []
    ci, ck = best_end
    while parent[ci][ck] is not None:
        pi, pk, move = parent[ci][ck]
        moves.append(move)
        ci, ck = pi, pk
    moves.reverse()

    n_match = sum(1 for m in moves if m[0] == _MATCH)
    n_skip = sum(1 for m in moves if m[0] == _SKIP_AYAH)
    n_noise = sum(1 for m in moves if m[0] == _NOISE)
    logger.info(f"DP: cost={best_cost:.3f}, "
                f"{n_match} MATCHes, {n_skip} SKIPs, {n_noise} NOISE moves")
    return moves
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_dtw_alignment.py::test_dp_sequential_clean_path \
                 tests/test_dtw_alignment.py::test_dp_absorbs_leading_noise -v
```

Expected: Both PASS.

- [ ] **Step 5: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/dtw_alignment.py tests/test_dtw_alignment.py
git commit -m "feat: add DP alignment engine (MATCH/SKIP_AYAH/NOISE transitions)"
```

---

## Task 5: Build `RecitationEvent` objects from DP path

**Files:**
- Modify: `src/dtw_alignment.py`
- Modify: `tests/test_dtw_alignment.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_dtw_alignment.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_dtw_alignment.py::test_build_events_sequential -v
```

Expected: `FAIL — ImportError: cannot import name 'build_recitation_events'`

- [ ] **Step 3: Add `build_recitation_events` to `src/dtw_alignment.py`**

Append to `src/dtw_alignment.py`:

```python
def build_recitation_events(
    path: List[Tuple],
    words: List[TranscribedWord],
    surah: int,
    ayah_corpus: Dict[int, Dict],
    normalizer: ArabicNormalizer,
    config: DTWConfig,
) -> List:
    """
    Convert a DP path into annotated RecitationEvent objects.

    MATCH moves  → "full" or "partial" events.
    SKIP_AYAH    → "skip" events (no timing).
    NOISE regions→ checked against already-matched ayahs; strong match
                   reclassified as "repetition" with incremented occurrence.

    Args:
        path:        Output of run_dp_alignment.
        words:       Transcribed words (same list passed to run_dp_alignment).
        surah:       Surah number.
        ayah_corpus: {ayah_num: {"norm_words", "normalized", "count"}}
        normalizer:  ArabicNormalizer instance.
        config:      DTWConfig.

    Returns:
        List of RecitationEvent objects in time order.
    """
    # Import here to avoid circular import at module load time
    from src.alignment_utils import RecitationEvent

    events: List[RecitationEvent] = []
    occurrence: Dict[int, int] = {}
    matched_ayahs: List[int] = []   # ayahs already matched, in order

    # ---- First pass: process MATCH and SKIP_AYAH moves ----
    noise_regions: List[Tuple[int, int]] = []   # (start_i, end_i) of NOISE runs
    prev_was_noise = False
    noise_run_start = 0

    for move in path:
        mtype = move[0]

        if mtype == _MATCH:
            if prev_was_noise:
                noise_regions.append((noise_run_start, move[2]))  # move[2] = start_i
                prev_was_noise = False

            _, ayah_num, start_i, end_i, score = move
            ref_count = ayah_corpus[ayah_num]["count"]
            occurrence[ayah_num] = occurrence.get(ayah_num, 0) + 1
            occ = occurrence[ayah_num]

            consumed = end_i - start_i
            is_partial = (
                score < config.partial_confidence_threshold
                or consumed < ref_count - 3
            )

            if occ > 1:
                evt_type = "repetition"
            elif is_partial:
                evt_type = "partial"
            else:
                evt_type = "full"

            event_words = words[start_i:end_i]
            events.append(RecitationEvent(
                surah=surah,
                ayah=ayah_num,
                occurrence=occ,
                start_time=event_words[0].start if event_words else 0.0,
                end_time=event_words[-1].end if event_words else 0.0,
                confidence=score,
                transcribed_text=" ".join(w.word for w in event_words),
                word_indices=(start_i, end_i),
                is_partial=is_partial,
                partial_type="partial" if is_partial else "full",
                reference_word_count=ref_count,
                event_type=evt_type,
            ))
            matched_ayahs.append(ayah_num)

        elif mtype == _SKIP_AYAH:
            if prev_was_noise:
                noise_regions.append((noise_run_start, move[0]))
                prev_was_noise = False
            _, ayah_num = move
            ref_count = ayah_corpus[ayah_num]["count"]
            events.append(RecitationEvent(
                surah=surah, ayah=ayah_num, occurrence=1,
                start_time=0.0, end_time=0.0, confidence=0.0,
                transcribed_text="", word_indices=(0, 0),
                is_partial=True, partial_type="skip",
                reference_word_count=ref_count, event_type="skip",
            ))

        elif mtype == _NOISE:
            _, noise_s, noise_e = move
            if not prev_was_noise:
                noise_run_start = noise_s
                prev_was_noise = True
            # noise_e is the running end; will be captured when next non-NOISE arrives

    # Capture trailing noise
    if prev_was_noise and path:
        last_noise = [m for m in path if m[0] == _NOISE]
        if last_noise:
            noise_regions.append((noise_run_start, last_noise[-1][2]))

    # ---- Second pass: check noise regions for repetitions ----
    if matched_ayahs:
        for noise_start, noise_end in noise_regions:
            noise_words = words[noise_start:noise_end]
            if len(noise_words) < 2:
                continue
            best_score, best_ayah = 0.0, None
            for prev_ayah in set(matched_ayahs):
                ref = ayah_corpus.get(prev_ayah)
                if not ref:
                    continue
                s = score_window(
                    noise_words, ref["norm_words"], ref["normalized"], normalizer
                )
                if s > best_score:
                    best_score, best_ayah = s, prev_ayah

            if best_ayah is not None and best_score >= config.confidence_threshold:
                ref_count = ayah_corpus[best_ayah]["count"]
                occurrence[best_ayah] = occurrence.get(best_ayah, 0) + 1
                events.append(RecitationEvent(
                    surah=surah, ayah=best_ayah,
                    occurrence=occurrence[best_ayah],
                    start_time=noise_words[0].start,
                    end_time=noise_words[-1].end,
                    confidence=best_score,
                    transcribed_text=" ".join(w.word for w in noise_words),
                    word_indices=(noise_start, noise_end),
                    is_partial=False, partial_type="full",
                    reference_word_count=ref_count,
                    event_type="repetition",
                ))

    # Sort by start_time (skip events have 0.0; keep them stable)
    events.sort(key=lambda e: (e.start_time, e.ayah))
    return events
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_dtw_alignment.py::test_build_events_sequential \
                 tests/test_dtw_alignment.py::test_build_events_detects_repetition -v
```

Expected: Both PASS.

- [ ] **Step 5: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/dtw_alignment.py tests/test_dtw_alignment.py
git commit -m "feat: add build_recitation_events with repetition post-processing"
```

---

## Task 6: Wire DTW into `AyahDetector`

**Files:**
- Modify: `src/alignment_utils.py`
- Modify: `tests/test_dtw_alignment.py`

- [ ] **Step 1: Write integration test**

Add to `tests/test_dtw_alignment.py`:

```python
import json
from src.audio_processing_utils import load_quran_text
from src.alignment_utils import AyahDetector


def test_dtw_mode_smoke_surah56():
    """DTW mode runs on the Surah 56 fixture and detects ≥35 full matches."""
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
        f"Expected ≥35 full matches, got {len(full_matches)}. "
        f"All events: {[(r['ayah'], r.get('event_type')) for r in results]}"
    )
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest tests/test_dtw_alignment.py::test_dtw_mode_smoke_surah56 -v
```

Expected: `FAIL — detect_ayahs_from_transcription() got an unexpected keyword argument 'mode'`

- [ ] **Step 3: Add import to `src/alignment_utils.py`**

At the top of `src/alignment_utils.py`, after the existing imports, add:

```python
from src.dtw_alignment import (
    DTWConfig,
    build_banded_similarity_matrix,
    run_dp_alignment,
    build_recitation_events,
)
```

- [ ] **Step 4: Add `mode` parameter to `detect_ayahs_from_transcription`**

Update the method signature (around line 301 in `src/alignment_utils.py`):

```python
def detect_ayahs_from_transcription(
    self,
    transcribed_words: List[TranscribedWord],
    window_size: int = 10,
    overlap: int = 3,
    surah_hint: Optional[int] = None,
    sequential_mode: bool = True,
    start_ayah: int = 1,
    end_ayah: Optional[int] = None,
    skip_preamble: bool = True,
    allow_repetition: bool = False,
    mode: str = "sequential",   # "sequential" | "dtw"
) -> List[Dict[str, Any]]:
```

Add this block immediately after the logging lines at the start of the method body:

```python
        if mode == "dtw" and surah_hint:
            events = self.detect_ayahs_dtw(
                transcribed_words=transcribed_words,
                surah=surah_hint,
                start_ayah=start_ayah,
                end_ayah=end_ayah,
                skip_preamble=skip_preamble,
            )
            return [e.to_dict() for e in events]
```

- [ ] **Step 5: Add `detect_ayahs_dtw` method to `AyahDetector`**

Add this method inside the `AyahDetector` class, after `_detect_with_repetition`:

```python
    def detect_ayahs_dtw(
        self,
        transcribed_words: List[TranscribedWord],
        surah: int,
        start_ayah: int = 1,
        end_ayah: Optional[int] = None,
        skip_preamble: bool = True,
        config: Optional[DTWConfig] = None,
    ) -> List:
        """
        DTW-based globally optimal ayah alignment.

        Builds a banded similarity matrix then finds the minimum-cost
        alignment path via DP. Immune to cascade failure. Handles
        repetitions, unusual pauses, and skipped ayahs.

        Args:
            transcribed_words: Words from Whisper with timestamps.
            surah:             Surah number.
            start_ayah:        First ayah in recording.
            end_ayah:          Last ayah (None = end of surah).
            skip_preamble:     Skip isti'adha/basmallah at start.
            config:            DTWConfig (defaults used if None).

        Returns:
            List of RecitationEvent objects.
        """
        if config is None:
            config = DTWConfig(confidence_threshold=self.confidence_threshold)

        max_ayah = max(
            (a for (s, a) in self.corpus if s == surah),
            default=286,
        )
        actual_end = end_ayah if end_ayah else max_ayah

        logger.info(
            f"DTW alignment: Surah {surah}, Ayahs {start_ayah}–{actual_end}, "
            f"{len(transcribed_words)} words"
        )

        # Skip preamble
        word_start = 0
        if skip_preamble:
            word_start = self._skip_opening_formulas(transcribed_words)
            if word_start > 0:
                logger.info(f"Skipped {word_start} preamble words")

        working_words = transcribed_words[word_start:]

        # Build per-ayah corpus for this range
        ayah_corpus: Dict[int, Dict] = {}
        for ayah_num in range(start_ayah, actual_end + 1):
            data = self.corpus.get((surah, ayah_num))
            if data:
                norm_words = [self.normalizer.normalize(w) for w in data["words"]]
                ayah_corpus[ayah_num] = {
                    "norm_words": norm_words,
                    "normalized": data["normalized"],
                    "count": len(data["words"]),
                    "display": data["display"],
                }

        if not ayah_corpus:
            logger.warning(
                f"No corpus data for Surah {surah} "
                f"Ayahs {start_ayah}–{actual_end}"
            )
            return []

        matrix = build_banded_similarity_matrix(
            words=working_words,
            ayah_corpus=ayah_corpus,
            ayah_range=(start_ayah, actual_end),
            normalizer=self.normalizer,
            config=config,
        )
        path = run_dp_alignment(
            words=working_words,
            ayah_corpus=ayah_corpus,
            ayah_range=(start_ayah, actual_end),
            similarity_matrix=matrix,
            config=config,
        )
        events = build_recitation_events(
            path=path,
            words=working_words,
            surah=surah,
            ayah_corpus=ayah_corpus,
            normalizer=self.normalizer,
            config=config,
        )

        # Adjust word_indices back to original transcribed_words offsets
        for e in events:
            e.word_indices = (
                e.word_indices[0] + word_start,
                e.word_indices[1] + word_start,
            )

        n_full = sum(1 for e in events if e.event_type == "full")
        n_rep  = sum(1 for e in events if e.event_type == "repetition")
        n_part = sum(1 for e in events if e.event_type == "partial")
        logger.info(
            f"DTW complete: {len(events)} events "
            f"({n_full} full, {n_rep} repetitions, {n_part} partial)"
        )
        return events
```

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/test_dtw_alignment.py::test_dtw_mode_smoke_surah56 -v
python -m pytest tests/ -v
```

Expected: Smoke test PASS (≥35 full matches). All existing tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/alignment_utils.py src/dtw_alignment.py tests/test_dtw_alignment.py
git commit -m "feat: wire detect_ayahs_dtw into AyahDetector via mode='dtw'"
```

---

## Task 7: Regression checks

**Files:** No code changes — validation only.

- [ ] **Step 1: Run existing regression suite (sequential mode — must stay green)**

```bash
python regression_tests.py run
```

Expected: 39/40 (97.5%) — same as before. If this drops, stop and investigate before proceeding.

- [ ] **Step 2: Run DTW against Surah 56 fixture and log results**

```bash
python - <<'EOF'
import json
from pathlib import Path
from src.audio_processing_utils import TranscribedWord, load_quran_text
from src.alignment_utils import AyahDetector

base = Path(".")
with open("data/transcriptions/95ce00c87a6e56f5_Surah Al-Waqi_ah Verses 1 - 40  Shaykh Ali Salah O.json") as f:
    data = json.load(f)

words = []
for seg in data.get("transcription", data).get("segments", []):
    for w in seg.get("words", []):
        words.append(TranscribedWord(word=w["word"].strip(),
            start=w["start"], end=w["end"],
            confidence=w.get("probability", 0.0)))

quran_data = load_quran_text("data/quran/quran.json")
detector = AyahDetector(quran_data=quran_data)

results = detector.detect_ayahs_from_transcription(
    transcribed_words=words, surah_hint=56,
    start_ayah=1, end_ayah=40, mode="dtw")

print(f"\nTotal events: {len(results)}")
for r in results:
    print(f"  {r['ayah']:3}  {r.get('event_type','?'):12}  conf={r['confidence']:.0%}  "
          f"{r['start_time']:.1f}s–{r['end_time']:.1f}s")
EOF
```

Expected: 39–40 ayahs detected, most as "full".

- [ ] **Step 3: Run DTW against Anfal repetition fixture**

```bash
python - <<'EOF'
import json
from pathlib import Path
from src.audio_processing_utils import TranscribedWord, load_quran_text
from src.alignment_utils import AyahDetector

base = Path(".")
with open("data/transcriptions/7c82281ffe342bba_Sheikh Musa Anfal_processed.wav.json") as f:
    data = json.load(f)

words = []
for seg in data.get("transcription", data).get("segments", []):
    for w in seg.get("words", []):
        words.append(TranscribedWord(word=w["word"].strip(),
            start=w["start"], end=w["end"],
            confidence=w.get("probability", 0.0)))

quran_data = load_quran_text("data/quran/quran.json")
detector = AyahDetector(quran_data=quran_data)

results = detector.detect_ayahs_from_transcription(
    transcribed_words=words, surah_hint=8,
    start_ayah=1, end_ayah=75, mode="dtw")

reps = [r for r in results if r.get("event_type") == "repetition"]
print(f"Total: {len(results)} events, {len(reps)} repetitions")
for r in reps:
    print(f"  Ayah {r['ayah']} occ#{r['occurrence']}  {r['start_time']:.1f}s–{r['end_time']:.1f}s")
EOF
```

Expected: repetition events visible for the ayahs the sheikh re-reads.

- [ ] **Step 4: If DTW Surah 56 result is worse than sequential, tune band_width_min**

If full matches < 35, try increasing `band_width_min` to 25:

```python
# In detect_ayahs_dtw(), replace:
config = DTWConfig(confidence_threshold=self.confidence_threshold)
# with:
config = DTWConfig(confidence_threshold=self.confidence_threshold, band_width_min=25)
```

Re-run Step 2 until ≥37 full matches.

- [ ] **Step 5: Commit tuning changes if any**

```bash
git add src/alignment_utils.py
git commit -m "tune: adjust DTW band_width_min based on regression results"
```

---

## Task 8: Timeline bar UI component

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add `render_alignment_timeline` helper to `app.py`**

Find `def detect_ayahs_workflow(` (around line 555 in `app.py`). Insert this function immediately before it:

```python
def render_alignment_timeline(detected_ayahs: List[Dict], total_duration: float) -> None:
    """
    Render a horizontal timeline bar showing ayah alignment results.
    Clicking a block sets st.session_state.selected_event_idx.

    Args:
        detected_ayahs: List of ayah dicts from detect_ayahs_from_transcription.
        total_duration:  Total audio duration in seconds.
    """
    if not detected_ayahs or total_duration <= 0:
        return

    TYPE_COLOURS = {
        "full":       "#2ecc71",
        "repetition": "#f39c12",
        "partial":    "#9b59b6",
        "skip":       "#888888",
    }

    segments_html = ""
    for idx, ayah in enumerate(detected_ayahs):
        start = ayah.get("start_time", 0.0)
        end   = ayah.get("end_time", start + 0.1)
        if end <= start:
            continue
        evt_type = ayah.get("event_type", "full")
        colour   = TYPE_COLOURS.get(evt_type, "#2ecc71")
        left_pct  = (start / total_duration) * 100
        width_pct = max(0.3, ((end - start) / total_duration) * 100)
        label = str(ayah.get("ayah", "?"))
        if ayah.get("occurrence", 1) > 1:
            label += f"×{ayah['occurrence']}"
        segments_html += (
            f'<div onclick="selectEvt({idx})"'
            f' title="Ayah {ayah.get(\"ayah\",\"?\")} ({evt_type}) '
            f'{start:.1f}s\u2013{end:.1f}s"'
            f' style="position:absolute;left:{left_pct:.2f}%;width:{width_pct:.2f}%;'
            f'height:100%;background:{colour};cursor:pointer;'
            f'border-right:1px solid #111;box-sizing:border-box;'
            f'font-size:9px;color:#fff;overflow:hidden;'
            f'padding-left:2px;line-height:28px;">{label}</div>'
        )

    legend = "".join(
        f'<span style="margin-right:12px;font-size:11px;">'
        f'<span style="color:{c}">&#9632;</span>&nbsp;{t}</span>'
        for t, c in TYPE_COLOURS.items()
    )

    html = f"""
    <div style="font-family:sans-serif;padding:4px 0;">
      <div style="position:relative;height:28px;background:#1a1a1a;
                  border-radius:4px;overflow:hidden;margin-bottom:6px;">
        {segments_html}
      </div>
      <div style="color:#aaa;">{legend}</div>
    </div>
    <script>
    function selectEvt(idx) {{
      window.parent.postMessage(
        {{isStreamlitMessage: true, type: 'streamlit:setComponentValue', value: idx}},
        '*'
      );
    }}
    </script>
    """
    import streamlit.components.v1 as components
    components.html(html, height=65)
```

- [ ] **Step 2: Verify no import errors**

```bash
python -c "import app; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add render_alignment_timeline to app.py"
```

---

## Task 9: Card list UI, mode selector, and wire into Detect Ayahs tab

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add `render_event_cards` helper immediately after `render_alignment_timeline`**

```python
def render_event_cards(detected_ayahs: List[Dict]) -> None:
    """
    Render annotated cards for each recitation event.
    Repetition cards have an Include/Exclude toggle persisted in session state.
    """
    if not detected_ayahs:
        return

    BADGE = {
        "full":       "background:#27ae60;color:#fff",
        "repetition": "background:#f39c12;color:#000",
        "partial":    "background:#8e44ad;color:#fff",
        "skip":       "background:#555;color:#fff",
    }

    if "excluded_event_indices" not in ss:
        ss.excluded_event_indices = set()

    for idx, ayah in enumerate(detected_ayahs):
        evt_type  = ayah.get("event_type", "full")
        ayah_num  = ayah.get("ayah", "?")
        surah_num = ayah.get("surah", "?")
        start     = ayah.get("start_time", 0.0)
        end       = ayah.get("end_time", 0.0)
        conf      = ayah.get("confidence", 0.0)
        occ       = ayah.get("occurrence", 1)

        badge     = BADGE.get(evt_type, BADGE["full"])
        excluded  = idx in ss.excluded_event_indices
        opacity   = "0.45" if excluded else "1.0"
        is_sel    = (ss.get("selected_event_idx") == idx)
        border    = "#2ecc71" if is_sel else ("#555" if excluded else "#333")

        occ_str   = f" <em style='color:#888;font-size:0.85em;'>(#{occ})</em>" if occ > 1 else ""
        conf_col  = "#2ecc71" if conf >= 0.85 else ("#f39c12" if conf >= 0.65 else "#e74c3c")

        st.markdown(
            f'<div style="border:1px solid {border};border-radius:6px;'
            f'padding:8px 12px;margin-bottom:6px;opacity:{opacity};">'
            f'<strong>{surah_num}:{ayah_num}</strong>{occ_str}&nbsp;&nbsp;'
            f'<span style="padding:2px 7px;border-radius:10px;font-size:0.8em;{badge}">'
            f'{evt_type}</span>'
            f'&nbsp;&nbsp;<span style="color:#aaa;font-size:0.82em;">'
            f'{start:.1f}s\u2013{end:.1f}s</span>'
            f'&nbsp;&nbsp;<span style="color:{conf_col};font-size:0.82em;">'
            f'conf:&nbsp;{conf:.0%}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if evt_type == "repetition":
            include = idx not in ss.excluded_event_indices
            new_val = st.checkbox(
                "Include in video",
                value=include,
                key=f"evt_include_{idx}",
                help="Uncheck to exclude this repetition from the final video.",
            )
            if not new_val:
                ss.excluded_event_indices.add(idx)
            else:
                ss.excluded_event_indices.discard(idx)
```

- [ ] **Step 2: Add DTW mode selector to Advanced Options in `app.py`**

Find `with st.expander("⚙️ Advanced Options", expanded=False):` (around line 1165). Inside that expander, after the existing options, add:

```python
                        st.markdown("---")
                        alignment_mode = st.selectbox(
                            "Alignment mode",
                            options=["sequential", "dtw"],
                            index=0,
                            key="alignment_mode",
                            help=(
                                "**sequential** — proven, fast, best for clean recordings. "
                                "**dtw** — globally optimal, handles stumbles, repetitions, "
                                "unusual pauses (slower)."
                            ),
                        )
```

- [ ] **Step 3: Pass `mode` through `detect_ayahs_workflow`**

Update the signature of `detect_ayahs_workflow` (around line 555):

```python
def detect_ayahs_workflow(
    transcribed_words,
    confidence_threshold=0.70,
    surah_hint=None,
    progress_bar=None,
    status_text=None,
    sequential_mode=True,
    start_ayah=1,
    end_ayah=None,
    skip_preamble=True,
    allow_repetition=False,
    mode="sequential",          # NEW
):
```

Inside the function, pass `mode` to `detect_ayahs_from_transcription`:

```python
        detected_ayahs = ayah_detector.detect_ayahs_from_transcription(
            transcribed_words=transcribed_words,
            surah_hint=surah_hint,
            sequential_mode=sequential_mode,
            start_ayah=start_ayah,
            end_ayah=end_ayah,
            skip_preamble=skip_preamble,
            allow_repetition=allow_repetition,
            mode=mode,
        )
```

- [ ] **Step 4: Update the `detect_ayahs_workflow` call site (~line 1246)**

Replace the existing call:

```python
                        detected_ayahs = detect_ayahs_workflow(
                            transcribed_words=ss.transcribed_words,
                            confidence_threshold=confidence_threshold,
                            surah_hint=surah_hint,
                            progress_bar=progress_bar,
                            status_text=status_text,
                            start_ayah=start_ayah,
                            end_ayah=end_ayah,
                            skip_preamble=skip_preamble,
                            allow_repetition=allow_repetition,
                            mode=ss.get("alignment_mode", "sequential"),
                        )
```

- [ ] **Step 5: Replace the detected-ayahs results block in the Detect Ayahs tab**

Find and replace the block starting at `if ss.get('detected_ayahs'):` around line 1267 (the `st.dataframe(detection_data, ...)` section) with:

```python
                if ss.get('detected_ayahs'):
                    st.divider()
                    st.subheader("🎯 Alignment Results")

                    total_dur = 0.0
                    if ss.get("transcribed_words"):
                        total_dur = ss.transcribed_words[-1].end

                    render_alignment_timeline(ss.detected_ayahs, total_dur)
                    st.markdown("---")
                    render_event_cards(ss.detected_ayahs)

                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    full_n  = sum(1 for d in ss.detected_ayahs if d.get("event_type") == "full")
                    rep_n   = sum(1 for d in ss.detected_ayahs if d.get("event_type") == "repetition")
                    part_n  = sum(1 for d in ss.detected_ayahs if d.get("event_type") == "partial")
                    avg_c   = (sum(d["confidence"] for d in ss.detected_ayahs)
                               / len(ss.detected_ayahs))
                    col1.metric("Full matches", full_n)
                    col2.metric("Repetitions",  rep_n)
                    col3.metric("Partial",       part_n)
                    col4.metric("Avg confidence", f"{avg_c:.0%}")

                    low_conf = [d for d in ss.detected_ayahs
                                if d["confidence"] < confidence_threshold]
                    if low_conf:
                        st.warning(
                            f"⚠️ {len(low_conf)} ayahs below confidence threshold "
                            f"({confidence_threshold:.0%}) — review before aligning."
                        )
```

- [ ] **Step 6: Launch app and verify**

```bash
streamlit run app.py
```

1. Upload audio, transcribe.
2. In Advanced Options set mode = "dtw".
3. Run detection.
4. Verify: timeline bar with coloured blocks appears.
5. Verify: card list with badges appears.
6. Verify: any repetition cards show Include/Exclude checkbox.
7. Verify: summary metrics (full / repetitions / partial / avg confidence) display.

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "feat: add timeline + card list UI, DTW mode selector in Advanced Options"
```

---

## Task 10: Respect excluded events in alignment and export

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Filter excluded events before alignment**

Find the alignment call site in the Review & Align tab (~line 1335):

```python
                    aligned_ayahs = align_words_workflow(
                        detected_ayahs=ss.detected_ayahs,
```

Replace with:

```python
                    _excluded = ss.get("excluded_event_indices", set())
                    _ayahs_to_align = [
                        a for i, a in enumerate(ss.detected_ayahs)
                        if i not in _excluded
                    ]
                    aligned_ayahs = align_words_workflow(
                        detected_ayahs=_ayahs_to_align,
```

- [ ] **Step 2: Reset excluded set when new detection runs**

Find where `ss.detected_ayahs` is set after detection (around line 1259):

```python
                        ss.detected_ayahs = detected_ayahs
```

Add the reset immediately after:

```python
                        ss.detected_ayahs = detected_ayahs
                        ss.excluded_event_indices = set()   # reset on new detection
```

- [ ] **Step 3: Verify exclusions work end-to-end**

1. Detect ayahs on a recording that produces repetition events.
2. Uncheck "Include in video" on a repetition card.
3. Click "Align Words".
4. Verify the excluded ayah does not appear in the aligned output dropdown.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: filter excluded_event_indices before alignment and reset on new detection"
```
