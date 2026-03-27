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
