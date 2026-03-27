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
                            ref_count = ayah_corpus[j]["count"]
                            size_penalty = (
                                abs(w_size - ref_count) * config.noise_word_penalty
                            )
                            c = cost + (1.0 - score) + size_penalty
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

    # Find best terminal: prefer (M, N) but accept trailing skips/noise.
    # Penalise partial ayah coverage so that a full path always beats a
    # partial one even if the partial path has lower raw DP cost.
    best_cost, best_end = INF, (M, N)
    for ek in range(max(0, N - 3), N + 1):
        skip_penalty = (N - ek) * config.skip_ayah_penalty
        for ei in range(max(0, M - 10), M + 1):
            effective = dp[ei][ek] + skip_penalty
            if effective < best_cost:
                best_cost, best_end = effective, (ei, ek)

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

    MATCH moves  -> "full" or "partial" events.
    SKIP_AYAH    -> "skip" events (no timing).
    NOISE regions-> checked against already-matched ayahs; strong match
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
                # Close the noise region at the current position (no word index change)
                noise_regions.append((noise_run_start, noise_run_start))
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
            # noise_e is the running end; captured when next non-NOISE arrives

    # Capture trailing noise from explicit NOISE moves
    if prev_was_noise and path:
        last_noise = [m for m in path if m[0] == _NOISE]
        if last_noise:
            noise_regions.append((noise_run_start, last_noise[-1][2]))

    # Capture words that come after all matched/noise regions (not in path at all)
    last_covered = 0
    for move in path:
        if move[0] == _MATCH:
            last_covered = max(last_covered, move[3])  # end_i
        elif move[0] == _NOISE:
            last_covered = max(last_covered, move[2])  # end_i
    if last_covered < len(words):
        noise_regions.append((last_covered, len(words)))

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
