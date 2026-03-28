# DTW Repetition Handling — Design Spec

**Date:** 2026-03-28
**Status:** Approved
**Scope:** `src/dtw_alignment.py`, `src/alignment_utils.py`, `tests/test_dtw_alignment.py`

---

## Problem

Two failure modes observed in the Anfal recitation (Sheikh Musa, Surah 8):

### Issue 1 — Multi-ayah block repetition collapses to one event

The sheikh recites ayahs 2→3→4→5, then repeats ayahs 2→3→4. The DTW first-pass
correctly matches the initial sequence. The repeated block falls into a noise region.
The current second-pass scores the entire noise region (≈40 words) against each
previously-seen ayah and emits a single repetition event for the best match —
collapsing three distinct ayahs into one label.

### Issue 2 — Intra-ayah phrase repetition truncates the ayah

In ayah 10, the sheikh repeats the phrase "ومن نصر إلا من عند الله" (6 words), then
continues with the ayah ending "إن الله عزيز حكيم". Total transcription span: 24
words. The reference has 18 words.

`build_banded_similarity_matrix` hard-codes `max_w = ref_count + 3 = 21`, so the
matrix can never score the 24-word window. The last 3 words of the ayah ("إن الله
عزيز حكيم") are stranded as noise, and subsequent ayahs misalign.

---

## Design

### Change 1 — Widen the similarity matrix window (`dtw_alignment.py`)

**Location:** `build_banded_similarity_matrix`, inner loop over `w_size`

**Before:**
```python
max_w = min(M - i, ref_count + 3)
```

**After:**
```python
max_w = min(M - i, ref_count * 2)
```

Allows windows up to twice the reference word count at each band position. The
existing `score_window` function is already robust to over-long windows:
`word_score = matched / len(ref_norm_words)` — extra repeated words still match
reference words, keeping the score high. The DP `size_penalty` (proportional to
`|w_size - ref_count|`) naturally discourages unnecessarily wide matches when a
narrower window scores equally well.

No other changes to matrix building or DP.

### Change 2 — Greedy sub-segmentation in the noise second pass (`dtw_alignment.py`)

**Location:** `build_recitation_events`, second pass (noise → repetition check)

**Current behaviour:** For each noise region, score the entire block against each
previously-seen ayah, emit at most one repetition event.

**New behaviour:** Replace the single-block check with a greedy scan loop:

```
For each noise_region (noise_start, noise_end):
    ptr = noise_start
    while ptr < noise_end:
        best_score, best_ayah, best_w = 0.0, None, 0
        for prev_ayah in set(matched_ayahs):        # deduplicated
            ref = ayah_corpus[prev_ayah]
            rc  = ref["count"]
            for w in range(max(2, rc - 2), min(noise_end - ptr, rc * 2) + 1):
                s = score_window(words[ptr:ptr+w], ref["norm_words"],
                                 ref["normalized"], normalizer)
                if s > best_score:
                    best_score, best_ayah, best_w = s, prev_ayah, w
        if best_score >= config.confidence_threshold:
            occurrence[best_ayah] += 1
            emit RepetitionEvent(ayah=best_ayah, words[ptr:ptr+best_w], ...)
            ptr += best_w
        else:
            ptr += 1   # unmatched word — skip silently
```

**Key properties:**
- Emits 1–N repetition events from a single noise region (fixes multi-ayah block)
- Window range `[rc-2, rc*2]` mirrors the widened matrix (consistent tolerance)
- Unmatched words are skipped without error; existing first-pass events are unaffected
- `matched_ayahs` is already de-duplicated via `set()`; loop order is
  deterministic (sorted by ayah number for stable output)

---

## Data Flow

```
transcribed_words
    │
    ▼
build_banded_similarity_matrix
    max_w = min(M-i, ref_count * 2)   ← widened
    store (best_score, best_w) per (word_pos, ayah_num)
    │
    ▼
run_dp_alignment           (unchanged)
    │
    ▼
build_recitation_events — first pass   (unchanged)
    MATCH  → full / partial / repetition events
    NOISE  → noise_regions list
    │
    ▼
build_recitation_events — second pass  ← rewritten
    greedy sub-segmentation of each noise region
    → 0–N additional repetition events
```

---

## Edge Cases

| Scenario | Handled by |
|---|---|
| Intra-repeat at ayah end (ayah 10) | Wider max_w in matrix; MATCH absorbs full span |
| Multi-ayah block repetition (ayahs 2–4) | Second-pass greedy loop emits 3 separate events |
| Noise region with no good sub-match | ptr advances word-by-word; no crash, no event |
| Wider window merges two adjacent ayahs | DP size_penalty discourages it; globally optimal path |
| `ref_count * 2` overflows `M - i` | Clipped by existing `min(M - i, …)` |
| Single-ayah noise region (current behaviour) | Loop runs once; same result as before |

---

## Testing

### New unit tests — `tests/test_dtw_alignment.py`

**`test_wider_window_covers_intra_repeat`**
Build a 3-ayah corpus. Construct transcription where ayah 2 is recited correctly
then its last phrase is repeated (word count = ref_count + 6). Assert the MATCH
event for ayah 2 covers all words including the repeated tail (`end_idx - start_idx
>= ref_count + 5`).

**`test_noise_second_pass_splits_block_repetition`**
Recite ayahs 1+2 cleanly (MATCH events), then add a noise region containing ayah 1
content followed by ayah 2 content. Assert second pass emits exactly 2 repetition
events with `occurrence == 2` and non-overlapping `word_indices`.

### New integration tests — `tests/test_dtw_alignment.py`

**`test_dtw_anfal_ayah4_block_repetition`**
Run DTW on the Anfal fixture (`start_ayah=1, end_ayah=10`). Assert each of ayahs 2,
3, 4 appears with at least one event having `occurrence >= 2`.

**`test_dtw_anfal_ayah10_intra_repeat`**
Same fixture. Assert the event(s) for ayah 10 together span at least 20 words
(`sum(e.word_indices[1] - e.word_indices[0] for e in ayah10_events) >= 20`),
confirming both the repeated phrase and the ayah ending are captured.

---

## Files Changed

| File | Change |
|---|---|
| `src/dtw_alignment.py` | `max_w` one-liner; second-pass loop replacement |
| `tests/test_dtw_alignment.py` | 4 new tests (2 unit, 2 integration) |
| `src/alignment_utils.py` | No changes needed |
