# DTW-Based Ayah Alignment Redesign

**Date:** 2026-03-27
**Status:** Approved for implementation
**Scope:** Replace `_detect_sequential` and `_detect_with_repetition` in `alignment_utils.py` with a single DTW-based global alignment engine. Add a new results visualisation to the Streamlit UI.

---

## Problem Statement

The current alignment algorithm makes greedy local decisions — it processes one segment at a time and commits immediately. A single bad match causes every subsequent ayah to be misidentified (cascade failure). It also depends on pause-based segmentation, which breaks when a sheikh pauses mid-ayah or flows across ayah boundaries without a gap.

Difficult recitations that cause failure today:
- Sheikh makes a mistake and self-corrects (stumbles, restarts the ayah)
- Sheikh deliberately repeats an ayah
- Mid-ayah pauses breaking the pause segmentation assumption
- Sheikh skips an ayah entirely
- Cascade failure from one bad match propagating forward

---

## Goals

1. Replace greedy local matching with a globally optimal alignment that is structurally immune to cascade failure.
2. Remove dependence on pause-based segmentation as a correctness requirement.
3. Detect and annotate repetitions and partial matches as first-class events; leave stumble vs deliberate repeat classification to the user.
4. Surface annotated results clearly in the Streamlit UI.

### Out of Scope

- Word-level timestamp precision within a segment (ayah-level timing is sufficient for difficult recitations).
- Real-time processing — this is a batch offline step, seconds-level latency is acceptable.
- Multi-surah recordings in a single session.

---

## Algorithm Design

### Overview

```
Whisper words
    │
    ▼
Preamble skip (isti'adha / basmallah)
    │
    ▼
Build banded similarity matrix
(word positions × ayahs, constrained by positional prior)
    │
    ▼
DP alignment — find globally optimal path
(MATCH / SKIP_AYAH / CONSUME_NOISE transitions)
    │
    ▼
Traceback → raw event sequence
    │
    ▼
Repetition & stumble post-processing
    │
    ▼
Annotated RecitationEvent list
```

### Phase 1 — Preamble Skip

Same logic as current `_skip_opening_formulas()`. Words that don't match any ayah in the range at the start are consumed as preamble. The DP also naturally handles unmatched leading words via CONSUME_NOISE transitions, so this phase is a fast-path optimisation only.

### Phase 2 — Banded Similarity Matrix

The user always specifies a surah and ayah range (`start_ayah` to `end_ayah`). This gives us a strong positional prior:

1. Compute `total_ref_words` = sum of reference word counts for all ayahs in range.
2. For each ayah `j`, compute:
   ```
   expected_word_pos(j) = (words_before_j / total_ref_words) × len(transcribed_words)
   ```
3. Only score similarity for word positions `i` within `±band_width` of `expected_word_pos(j)`.

This creates a sparse (banded) matrix. Default `band_width` = `max(15, total_ref_words * 0.15)` — 15% of total expected words, minimum 15. Wide enough to absorb skips and repetitions, narrow enough to rule out implausible long-range misalignments.

**Similarity scoring for cell (i, j):**

- Window size floats around `ref_word_count(j)` with tolerance ±2.
- Best score across window sizes is used.
- Scoring uses existing `_calculate_word_alignment_score()` (word-level fuzzy matching) blended with `_quick_fuzzy_score()` (string-level), weighted 70/30 as today.
- Score is normalised to [0, 1]. Cost = `1 − score`.

### Phase 3 — Dynamic Programming Alignment

**State:** `(i, j)` — transcription word position `i`, ayah index `j`
**Goal:** minimum-cost path from `(0, start_ayah)` to any `(M, near_end_ayah)`

**Transitions:**

| Move | Effect | Cost |
|------|--------|------|
| **MATCH** | Consume `ref_count(j)` words (±2 flex), advance to ayah `j+1` | `1 − similarity(i, j)` |
| **SKIP_AYAH** | Advance to ayah `j+1` without consuming words | `0.85` (high penalty — last resort only) |
| **CONSUME_NOISE** | Consume 1 word, stay on ayah `j` | `0.15` per word |

SKIP_AYAH is considered unlikely in practice (a sheikh skipping an ayah is unusual and usually a transcription/processing error). The high penalty of 0.85 means the DP will only take this move when no plausible MATCH or CONSUME_NOISE path exists. It exists to prevent the DP from getting permanently stuck rather than to model intentional skips.

CONSUME_NOISE is allowed for up to `max_noise_run = min(8, ref_count(j))` consecutive words before MATCH must be attempted. This prevents the DP from treating an entire ayah as noise.

**Termination:** The DP is complete when either all ayahs in range have been processed or all transcription words are consumed. Remaining unmatched ayahs at the end are flagged as skipped.

**Complexity:** O(M × band_width × N) where M = transcription words, N = ayahs in range. For a 500-word recording over 40 ayahs with band_width=75, this is ~1.5M operations — well within acceptable latency.

### Phase 4 — Traceback

Walk backwards from the optimal terminal state. Each MATCH transition produces a raw event:

```python
RawEvent(
    ayah=j,
    start_word_idx=i,
    end_word_idx=i + consumed,
    start_time=words[i].start,
    end_time=words[i + consumed - 1].end,
    similarity_score=score,
    preceded_by_noise_words=n   # count of CONSUME_NOISE moves before this MATCH
)
```

SKIP_AYAH transitions produce a `SkippedAyah` record (no timing, just an annotation).

### Phase 5 — Repetition Post-Processing

After traceback, scan for these patterns in the raw event sequence:

**Repetition detection:**
If a MATCH event for ayah `j` appears when ayah `j` has already been matched earlier in the sequence, it is a repetition. Increment `occurrence` counter. The algorithm does not attempt to classify whether a repetition was a stumble (mistake) or deliberate — that distinction is left to the user in the UI.

**Partial detection:**
A MATCH event is flagged as partial if `similarity_score < 0.55` or `consumed_words < ref_count(j) - 3`.

### Phase 6 — Annotated Output

Each event is a `RecitationEvent` (extending the existing dataclass):

```python
@dataclass
class RecitationEvent:
    surah: int
    ayah: int
    occurrence: int          # 1 = first time, 2+ = repetition
    start_time: float
    end_time: float
    confidence: float        # = similarity_score
    transcribed_text: str
    word_indices: Tuple[int, int]
    event_type: str          # "full" | "partial" | "repetition" | "skip"
    is_partial: bool         # True if event_type is "partial" — kept for backward compat
    reference_word_count: int
```

`event_type` replaces the current `is_partial` / `partial_type` split with a single unified field.

---

## Tunable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `band_width_ratio` | 0.15 | Band width as fraction of total expected words |
| `band_width_min` | 15 | Minimum band width in words |
| `skip_ayah_penalty` | 0.85 | DP cost for skipping an ayah (high — last resort only) |
| `noise_word_penalty` | 0.15 | DP cost per noise word consumed |
| `max_noise_run` | 8 | Max consecutive noise words before forced MATCH attempt |
| `partial_confidence_threshold` | 0.55 | Below this = partial event |
| `confidence_threshold` | 0.65 | Minimum score for a MATCH to be accepted at all |

---

## Streamlit UI Changes

### Results Visualisation (Detect Ayahs tab)

Replace the current flat table with two components stacked vertically:

**1. Timeline bar**

A horizontal bar spanning the full audio duration. Each ayah is a coloured block proportional to its duration. Colour encodes event type:
- Green (`#2ecc71`) — full match
- Orange (`#f39c12`) — repetition (user decides if stumble or deliberate)
- Purple (`#9b59b6`) — partial
- Grey — skipped

Clicking a block sets `st.session_state.selected_event_idx`, which the card list uses to highlight the corresponding card (Streamlit re-renders on state change — no native DOM scrolling needed).

**2. Annotated card list**

A vertical list of cards, one per event. Each card shows:
- Ayah reference (surah:ayah, occurrence number if >1)
- Time range (start → end)
- Confidence score
- Event type badge (colour-coded)

Repetition cards include a user toggle: **Include in video / Exclude**. The user decides whether a given repetition was a stumble (exclude) or deliberate (include). This decision is persisted in session state and respected during the alignment and export steps.

Low-confidence cards (< `confidence_threshold`) are visually distinct (dimmed border, warning icon).

### No Changes To

- Export format (Tarteel JSON) — unchanged
- Word alignment step — unchanged (runs after detection, same as today)
- Pre-processing and transcription tabs

---

## Implementation Notes

### What Changes in `alignment_utils.py`

- Add `detect_ayahs_dtw()` as the new primary entry point on `AyahDetector`.
- Keep `_detect_sequential()` and `_detect_with_repetition()` intact — they remain accessible via the existing `allow_repetition` flag for regression testing and fallback.
- `detect_ayahs_from_transcription()` gets a new `mode` parameter: `"sequential"` (current default), `"dtw"` (new). Default stays `"sequential"` until DTW passes regression tests.
- Reuse existing `_calculate_word_alignment_score()`, `_quick_fuzzy_score()`, `_skip_opening_formulas()`, `ArabicNormalizer`.
- `RecitationEvent` gains `event_type` field (backwards compatible — existing code that only reads `is_partial` still works).

### What Changes in `app.py`

- Replace the detected ayahs table in the Detect Ayahs tab with the timeline + card list components.
- Add per-event include/exclude toggles for stumble and repetition events.
- Pass include/exclude decisions through to the alignment and export steps.
- Add a mode selector (sequential / DTW) in Advanced Options — hidden by default, DTW becomes default once validated.

### Regression Testing

- All existing fixtures in `data/fixtures/` must pass at the same accuracy (currently 97.5%) with `mode="sequential"`.
- DTW mode targets ≥97.5% on existing fixtures plus correct handling of the Anfal (Surah 8) repetition test.
- A new fixture should be captured for a difficult recitation once the algorithm is working.

---

## Success Criteria

1. Existing regression tests pass unchanged with `mode="sequential"`.
2. DTW mode matches all 40 ayahs in the Surah 56 fixture (currently 39/40).
3. DTW mode correctly identifies and annotates repetitions in the Anfal fixture.
4. A recitation where a sheikh re-reads an ayah produces: repetition event flagged with `occurrence=2`, no downstream misalignment, user can include or exclude it.
5. A recitation where one ayah is skipped produces: skip annotation for the missing ayah, all subsequent ayahs correctly aligned.
