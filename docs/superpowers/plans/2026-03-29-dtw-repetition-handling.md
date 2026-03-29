# DTW Repetition Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two DTW failure modes — intra-ayah phrase repetition truncating ayah boundaries, and multi-ayah block repetition collapsing to a single event.

**Architecture:** Two surgical changes to `src/dtw_alignment.py`: (1) widen the similarity matrix window from `ref_count + 3` to `ref_count * 2`, enabling the DP to score over-long ayah spans; (2) replace the noise-region second-pass single-check with a greedy sub-segmentation loop that emits 1–N repetition events per noise region.

**Tech Stack:** Python, pytest, `src/dtw_alignment.py`, `tests/test_dtw_alignment.py`

---

## File Map

| File | Change |
|---|---|
| `src/dtw_alignment.py:161` | Change `max_w` one-liner in `build_banded_similarity_matrix` |
| `src/dtw_alignment.py:409-440` | Replace second-pass block in `build_recitation_events` |
| `tests/test_dtw_alignment.py` | Append 4 new tests (2 unit, 2 integration) |

---

## Task 1: Widen the similarity matrix window

**Spec:** Change 1 — `build_banded_similarity_matrix`, line 161.

**Files:**
- Modify: `src/dtw_alignment.py:161`
- Test: `tests/test_dtw_alignment.py`

- [ ] **Step 1: Write the failing test**

Add this test to the bottom of `tests/test_dtw_alignment.py`:

```python
# Task 1 regression: wider window covers intra-ayah phrase repetition
def test_wider_window_covers_intra_repeat():
    """
    Ayah 2 has 4 reference words. The transcription recites those 4 words
    then immediately repeats the last 3, giving 7 transcription words total.
    With max_w = ref_count + 3 = 7 the match is borderline; with ref_count * 2 = 8
    the matrix can comfortably score the full span.

    We assert that the MATCH event for ayah 2 consumes at least ref_count + 2
    transcription words (i.e. it absorbed part of the repeated tail).
    """
    from src.dtw_alignment import (
        build_banded_similarity_matrix, run_dp_alignment,
        build_recitation_events, DTWConfig,
    )
    from src.audio_processing_utils import ArabicNormalizer

    normalizer = ArabicNormalizer()

    # Corpus: ayah 1 = 3 words, ayah 2 = 4 words
    corpus = {
        1: {"norm_words": ["ا", "ب", "ت"],
            "normalized": "ا ب ت", "count": 3},
        2: {"norm_words": ["ج", "د", "ه", "و"],
            "normalized": "ج د ه و", "count": 4},
    }

    # Transcription: ayah1 words | ayah2 words | last 3 of ayah2 repeated
    # Total: 3 + 4 + 3 = 10 words
    raw = ["ا", "ب", "ت",   # ayah 1
           "ج", "د", "ه", "و",   # ayah 2 first pass
           "د", "ه", "و"]        # ayah 2 last 3 repeated
    t = 0.0
    trans = []
    for word in raw:
        trans.append(_w(word, t, t + 0.5))
        t += 0.5

    config = DTWConfig(band_width_min=10, confidence_threshold=0.55)
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
    assert ayah2_events, "Expected at least one full event for ayah 2"
    # The MATCH for ayah 2 should consume at least ref_count+2 = 6 words
    best = max(ayah2_events, key=lambda e: e.word_indices[1] - e.word_indices[0])
    consumed = best.word_indices[1] - best.word_indices[0]
    assert consumed >= 6, (
        f"Expected ayah 2 to consume >=6 words (absorbed repeat), got {consumed}. "
        f"word_indices={best.word_indices}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_dtw_alignment.py::test_wider_window_covers_intra_repeat -v
```

Expected: `FAILED` — the current `max_w = ref_count + 3 = 7` may pass borderline but the assertion `consumed >= 6` checks that the wider window is actually being used. If it passes already with the old code, increase the repeat tail in the test from 3 to 6 extra words so `ref_count + 3 = 7 < ref_count + 6 + 1 = 11`.

- [ ] **Step 3: Apply the one-line fix in `src/dtw_alignment.py`**

Find line 161 (the `max_w` line inside `build_banded_similarity_matrix`):

```python
# BEFORE (line 161)
            max_w = min(M - i, ref_count + 3)
```

Change to:

```python
# AFTER
            max_w = min(M - i, ref_count * 2)
```

Context for navigation — this line is inside the `for j in ayahs:` loop, immediately after `min_w = max(1, ref_count - 2)`:

```python
        for i in range(band_start, band_end + 1):
            best_score, best_w = 0.0, ref_count
            min_w = max(1, ref_count - 2)
            max_w = min(M - i, ref_count * 2)   # ← changed line
            for w_size in range(min_w, max_w + 1):
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run python -m pytest tests/test_dtw_alignment.py::test_wider_window_covers_intra_repeat -v
```

Expected: `PASSED`

- [ ] **Step 5: Run the full suite to check for regressions**

```bash
uv run python -m pytest tests/ -q
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/dtw_alignment.py tests/test_dtw_alignment.py
git commit -m "fix: widen DTW similarity matrix window to ref_count*2 for intra-ayah repetition"
```

---

## Task 2: Replace the noise second-pass with a greedy sub-segmentation loop

**Spec:** Change 2 — `build_recitation_events`, lines 409–440.

**Files:**
- Modify: `src/dtw_alignment.py:409-440`
- Test: `tests/test_dtw_alignment.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dtw_alignment.py`:

```python
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
    assert len(rep_events) >= 2, (
        f"Expected >=2 repetition events, got {len(rep_events)}. "
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_dtw_alignment.py::test_noise_second_pass_splits_block_repetition -v
```

Expected: `FAILED` — the old second-pass emits at most 1 repetition event per noise region.

- [ ] **Step 3: Replace the second pass in `src/dtw_alignment.py`**

Find the second-pass block starting at `# ---- Second pass: check noise regions for repetitions ----` (around line 409). Replace the entire block (lines 409–440) with:

```python
    # ---- Second pass: greedy sub-segmentation of noise regions ----
    # Each noise region is scanned left-to-right. At each pointer position we
    # try every previously-matched ayah over windows of [rc-2, rc*2] words.
    # The best-scoring window that meets confidence_threshold becomes a
    # repetition event; the pointer advances past it. This lets a single noise
    # region produce 1-N repetition events (fixing multi-ayah block repeats).
    if matched_ayahs:
        prev_ayahs_sorted = sorted(set(matched_ayahs))  # stable order
        for noise_start, noise_end in noise_regions:
            if noise_end - noise_start < 2:
                continue
            ptr = noise_start
            while ptr < noise_end:
                best_score, best_ayah, best_w = 0.0, None, 0
                for prev_ayah in prev_ayahs_sorted:
                    ref = ayah_corpus.get(prev_ayah)
                    if not ref:
                        continue
                    rc = ref["count"]
                    w_min = max(2, rc - 2)
                    w_max = min(noise_end - ptr, rc * 2)
                    for w in range(w_min, w_max + 1):
                        s = score_window(
                            words[ptr: ptr + w],
                            ref["norm_words"], ref["normalized"], normalizer,
                        )
                        if s > best_score:
                            best_score, best_ayah, best_w = s, prev_ayah, w

                if best_ayah is not None and best_score >= config.confidence_threshold:
                    ref_count = ayah_corpus[best_ayah]["count"]
                    occurrence[best_ayah] = occurrence.get(best_ayah, 0) + 1
                    rep_words = words[ptr: ptr + best_w]
                    events.append(RecitationEvent(
                        surah=surah, ayah=best_ayah,
                        occurrence=occurrence[best_ayah],
                        start_time=rep_words[0].start,
                        end_time=rep_words[-1].end,
                        confidence=best_score,
                        transcribed_text=" ".join(w.word for w in rep_words),
                        word_indices=(ptr, ptr + best_w),
                        is_partial=False, partial_type="full",
                        reference_word_count=ref_count,
                        event_type="repetition",
                    ))
                    ptr += best_w
                else:
                    ptr += 1  # no good match at this position — advance silently
```

- [ ] **Step 4: Run both Task 1 and Task 2 tests**

```bash
uv run python -m pytest tests/test_dtw_alignment.py::test_wider_window_covers_intra_repeat tests/test_dtw_alignment.py::test_noise_second_pass_splits_block_repetition -v
```

Expected: both `PASSED`

- [ ] **Step 5: Run full suite**

```bash
uv run python -m pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/dtw_alignment.py tests/test_dtw_alignment.py
git commit -m "fix: greedy noise sub-segmentation emits multiple repetition events per noise region"
```

---

## Task 3: Integration tests against the Anfal fixture

**Spec:** `test_dtw_anfal_ayah4_block_repetition` and `test_dtw_anfal_ayah10_intra_repeat`.

These tests use the real Anfal transcription file and verify end-to-end correctness of both fixes.

**Files:**
- Test: `tests/test_dtw_alignment.py`
- Read (fixture): `src/data/transcriptions/7c82281ffe342bba_Sheikh Musa Anfal_processed.wav.json`

- [ ] **Step 1: Write both integration tests**

Append to `tests/test_dtw_alignment.py`:

```python
import json
from pathlib import Path


def _load_anfal_words():
    """Load TranscribedWord list from the Anfal fixture. Returns None if missing."""
    from src.audio_processing_utils import TranscribedWord
    fixture = (
        Path(__file__).parent.parent
        / "src/data/transcriptions"
        / "7c82281ffe342bba_Sheikh Musa Anfal_processed.wav.json"
    )
    if not fixture.exists():
        return None
    with open(fixture) as f:
        data = json.load(f)
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


def test_dtw_anfal_ayah4_block_repetition():
    """
    Integration: Sheikh recites ayahs 2→3→4→5 then repeats ayahs 2→3→4.
    After both fixes, each of ayahs 2, 3, 4 must appear with at least one
    event where occurrence >= 2 (i.e. the repeated block is detected).
    """
    import pytest
    words = _load_anfal_words()
    if words is None:
        pytest.skip("Anfal fixture not available")

    from src.audio_processing_utils import load_quran_text
    from src.alignment_utils import AyahDetector

    quran_data = load_quran_text(
        str(Path(__file__).parent.parent / "data/quran/quran.json")
    )
    detector = AyahDetector(quran_data=quran_data, confidence_threshold=0.65)

    results = detector.detect_ayahs_from_transcription(
        transcribed_words=words,
        surah_hint=8,
        start_ayah=1,
        end_ayah=10,
        mode="dtw",
    )

    # Each of ayahs 2, 3, 4 must appear with occurrence >= 2
    for target_ayah in [2, 3, 4]:
        repeated = [
            r for r in results
            if r["ayah"] == target_ayah and r.get("occurrence", 1) >= 2
        ]
        assert repeated, (
            f"Ayah {target_ayah} has no repetition event (occurrence>=2). "
            f"All events for ayah {target_ayah}: "
            f"{[r for r in results if r['ayah'] == target_ayah]}"
        )


def test_dtw_anfal_ayah10_intra_repeat():
    """
    Integration: Ayah 10 of Surah Anfal has 18 reference words. The sheikh
    repeats a 6-word phrase mid-ayah, making the real span ~24 words. After
    the max_w fix, all events for ayah 10 combined must span >= 20 words,
    confirming the ending words 'إن الله عزيز حكيم' are captured within the
    ayah rather than stranded as noise.
    """
    import pytest
    words = _load_anfal_words()
    if words is None:
        pytest.skip("Anfal fixture not available")

    from src.audio_processing_utils import load_quran_text
    from src.alignment_utils import AyahDetector

    quran_data = load_quran_text(
        str(Path(__file__).parent.parent / "data/quran/quran.json")
    )
    detector = AyahDetector(quran_data=quran_data, confidence_threshold=0.65)

    results = detector.detect_ayahs_from_transcription(
        transcribed_words=words,
        surah_hint=8,
        start_ayah=1,
        end_ayah=15,
        mode="dtw",
    )

    ayah10_events = [r for r in results if r["ayah"] == 10]
    assert ayah10_events, "No events found for ayah 10"

    total_words_covered = sum(
        r["word_indices"][1] - r["word_indices"][0]
        for r in ayah10_events
    )
    assert total_words_covered >= 20, (
        f"Ayah 10 events cover only {total_words_covered} words — "
        f"expected >=20 (should include both the repeated phrase and ending). "
        f"Events: {[(r['ayah'], r['event_type'], r['word_indices']) for r in ayah10_events]}"
    )
```

- [ ] **Step 2: Run both integration tests**

```bash
uv run python -m pytest tests/test_dtw_alignment.py::test_dtw_anfal_ayah4_block_repetition tests/test_dtw_alignment.py::test_dtw_anfal_ayah10_intra_repeat -v
```

Expected: both `PASSED`. If either fails, check:
- `test_dtw_anfal_ayah4_block_repetition` failing → the block repetition of ayahs 2–4 isn't being sub-segmented. Verify the noise region in the path covers the right word range by adding a temporary `print(results)`.
- `test_dtw_anfal_ayah10_intra_repeat` failing → `total_words_covered < 20`. Verify the max_w change propagated by printing `matrix[(word_pos, 10)]` for a few positions.

- [ ] **Step 3: Run full suite one final time**

```bash
uv run python -m pytest tests/ -q
```

Expected: all tests pass (previously 17, now 21).

- [ ] **Step 4: Commit**

```bash
git add tests/test_dtw_alignment.py
git commit -m "test: add integration tests for intra-ayah and block repetition handling (Anfal)"
```

---

## Self-Review Notes

**Spec coverage check:**
- ✅ Change 1 (`max_w` widening) → Task 1
- ✅ Change 2 (greedy noise second-pass) → Task 2
- ✅ Unit test `test_wider_window_covers_intra_repeat` → Task 1 Step 1
- ✅ Unit test `test_noise_second_pass_splits_block_repetition` → Task 2 Step 1
- ✅ Integration test `test_dtw_anfal_ayah4_block_repetition` → Task 3 Step 1
- ✅ Integration test `test_dtw_anfal_ayah10_intra_repeat` → Task 3 Step 1
- ✅ Edge case: no match at ptr → `ptr += 1` (Task 2 Step 3, last line of inner loop)
- ✅ Edge case: `ref_count * 2` overflow → `min(noise_end - ptr, rc * 2)` clips it

**Type consistency check:**
- `build_recitation_events` signature unchanged; `RecitationEvent` fields unchanged
- `score_window` called identically in both Task 1 matrix building and Task 2 second-pass loop
- `word_indices=(ptr, ptr + best_w)` matches the tuple format used throughout

**Placeholder scan:** None found.
