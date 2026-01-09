#!/usr/bin/env python3
"""
Deep Debug Script for Segment Matching

This script provides detailed analysis of:
1. How transcription is segmented
2. What each segment contains
3. Why segments fail to match
4. Where ayah boundaries should be
"""

import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_processing_utils import TranscribedWord, load_quran_text
from alignment_utils import (
    AyahDetector, ArabicNormalizer, segment_by_pauses,
    WordClassification, reconstruct_ayahs
)

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)


def load_transcription(trans_path: str) -> List[TranscribedWord]:
    """Load transcription and extract words."""
    with open(trans_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    words = []
    trans_data = data.get("transcription", data)
    
    if "segments" in trans_data:
        for segment in trans_data["segments"]:
            if "words" in segment:
                for w in segment["words"]:
                    words.append(TranscribedWord(
                        word=w.get("word", "").strip(),
                        start=w.get("start", 0.0),
                        end=w.get("end", 0.0),
                        confidence=w.get("probability", w.get("confidence", 0.0))
                    ))
    
    return words


def analyze_segments(
    words: List[TranscribedWord],
    min_gap: float = 0.4,
    min_words: int = 4
) -> List[Tuple[int, int, str, float, float]]:
    """
    Analyze segmentation and return detailed info.
    
    Returns: [(start_idx, end_idx, text, start_time, end_time), ...]
    """
    segments = segment_by_pauses(words, min_gap_seconds=min_gap, min_segment_words=min_words)
    
    result = []
    for start_idx, end_idx in segments:
        seg_words = words[start_idx:end_idx]
        text = " ".join(w.word for w in seg_words)
        start_time = seg_words[0].start
        end_time = seg_words[-1].end
        result.append((start_idx, end_idx, text, start_time, end_time))
    
    return result


def find_ayah_in_corpus(
    detector: AyahDetector,
    text: str,
    surah: int,
    start_ayah: int,
    end_ayah: int
) -> List[Dict]:
    """
    Score a text segment against all ayahs in range.
    Returns sorted list of matches with scores.
    """
    normalizer = detector.normalizer
    text_norm = normalizer.normalize(text)
    text_words = text_norm.split()
    
    matches = []
    
    for ayah_num in range(start_ayah, min(end_ayah + 1, end_ayah + 20)):
        ayah_data = detector.corpus.get((surah, ayah_num))
        if not ayah_data:
            continue
        
        ref_norm = ayah_data["normalized"]
        ref_words = ref_norm.split()
        
        # Calculate similarity
        from rapidfuzz import fuzz
        fuzzy_score = fuzz.ratio(text_norm, ref_norm) / 100.0
        
        # Word overlap score
        matched_words = 0
        for t_word in text_words:
            best_match = max((fuzz.ratio(t_word, r_word) for r_word in ref_words), default=0)
            if best_match > 60:
                matched_words += 1
        
        word_coverage = matched_words / len(ref_words) if ref_words else 0
        
        matches.append({
            "ayah": ayah_num,
            "fuzzy_score": fuzzy_score,
            "word_coverage": word_coverage,
            "ref_text": ayah_data["display"][:50] + "...",
            "ref_words": len(ref_words),
            "matched_words": matched_words
        })
    
    # Sort by fuzzy score
    matches.sort(key=lambda x: x["fuzzy_score"], reverse=True)
    return matches[:5]  # Top 5


def analyze_gap_between_segments(
    words: List[TranscribedWord],
    segments: List[Tuple[int, int, str, float, float]],
    seg_idx: int
) -> Dict:
    """Analyze the gap between segment seg_idx and seg_idx+1."""
    if seg_idx >= len(segments) - 1:
        return {}
    
    seg1 = segments[seg_idx]
    seg2 = segments[seg_idx + 1]
    
    gap_start = seg1[1]  # End of seg1
    gap_end = seg2[0]    # Start of seg2
    
    if gap_end <= gap_start:
        return {"gap_words": 0}
    
    gap_words = words[gap_start:gap_end]
    gap_text = " ".join(w.word for w in gap_words)
    
    return {
        "gap_words": len(gap_words),
        "gap_text": gap_text,
        "gap_time": seg2[3] - seg1[4]  # Time gap
    }


def main():
    base_dir = Path(__file__).parent.parent
    trans_path = base_dir / "data/transcriptions/7c82281ffe342bba_Sheikh Musa Anfal_processed.wav.json"
    quran_path = base_dir / "data/quran/quran.json"
    
    print("=" * 80)
    print("DEEP DEBUG: ANFAL SEGMENT ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\n📂 Loading data...")
    words = load_transcription(str(trans_path))
    quran_data = load_quran_text(str(quran_path))
    
    print(f"   Loaded {len(words)} transcribed words")
    
    # Create detector
    normalizer = ArabicNormalizer()
    detector = AyahDetector(quran_data=quran_data, normalizer=normalizer, confidence_threshold=0.65)
    
    # Get Ayah 9 reference
    ayah_9_data = detector.corpus.get((8, 9))
    ayah_10_data = detector.corpus.get((8, 10))
    
    print("\n📖 Reference Ayahs:")
    print(f"   Ayah 9 ({len(ayah_9_data['display'].split())} words): {ayah_9_data['display'][:80]}...")
    print(f"   Ayah 10 ({len(ayah_10_data['display'].split())} words): {ayah_10_data['display'][:80]}...")
    
    # Analyze segments
    print("\n📊 Analyzing segmentation...")
    segments = analyze_segments(words, min_gap=0.4, min_words=4)
    print(f"   Found {len(segments)} segments")
    
    # Focus on segments 9-20 (around Ayah 9)
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS: SEGMENTS 9-25 (AYAHS 9-15 REGION)")
    print("=" * 80)
    
    for i in range(9, min(25, len(segments))):
        start_idx, end_idx, text, start_time, end_time = segments[i]
        word_count = end_idx - start_idx
        
        print(f"\n{'─' * 80}")
        print(f"SEGMENT {i}")
        print(f"{'─' * 80}")
        print(f"   Words: {word_count} (indices {start_idx}-{end_idx})")
        print(f"   Time: {start_time:.2f}s - {end_time:.2f}s ({end_time - start_time:.2f}s)")
        print(f"   Text: {text[:100]}...")
        
        # Score against ayahs
        matches = find_ayah_in_corpus(detector, text, 8, 9, 20)
        
        print(f"\n   Top Matches:")
        for m in matches[:3]:
            conf_str = f"{m['fuzzy_score']:.0%}"
            cov_str = f"{m['word_coverage']:.0%}"
            print(f"      Ayah {m['ayah']:2}: fuzzy={conf_str:>4}, coverage={cov_str:>4} ({m['matched_words']}/{m['ref_words']} words)")
            print(f"         Ref: {m['ref_text']}")
        
        # Analyze gap to next segment
        gap_info = analyze_gap_between_segments(words, segments, i)
        if gap_info.get("gap_words", 0) > 0:
            print(f"\n   ⚠️ Gap to next segment: {gap_info['gap_words']} words ({gap_info['gap_time']:.2f}s)")
            print(f"      Gap text: {gap_info['gap_text'][:60]}...")
    
    # Show where Ayah 9 ENDS and Ayah 10 STARTS
    print("\n" + "=" * 80)
    print("FINDING EXACT AYAH BOUNDARIES")
    print("=" * 80)
    
    # Find approximate location of Ayah 9 words
    ayah_9_first_word = normalizer.normalize("إذ")  # "When" - first word of Ayah 9
    ayah_9_last_word = normalizer.normalize("مردفين")  # Last word of Ayah 9
    ayah_10_first_word = normalizer.normalize("وما")  # First word of Ayah 10
    
    print(f"\n   Looking for:")
    print(f"   - Ayah 9 start: {ayah_9_first_word}")
    print(f"   - Ayah 9 end: {ayah_9_last_word}")
    print(f"   - Ayah 10 start: {ayah_10_first_word}")
    
    for i, w in enumerate(words[200:350]):  # Search in approximate region
        w_norm = normalizer.normalize(w.word)
        idx = i + 200
        
        if "استغ" in w_norm or w_norm == ayah_9_first_word:
            print(f"\n   [Word {idx}] @ {w.start:.2f}s: '{w.word}' (normalized: {w_norm})")
            # Show context
            context = " ".join(ww.word for ww in words[idx:idx+15])
            print(f"   Context: {context}")
        
        if "مردف" in w_norm:
            print(f"\n   [Word {idx}] @ {w.start:.2f}s: '{w.word}' (possible Ayah 9 end)")
            # Show context
            context = " ".join(ww.word for ww in words[idx:idx+10])
            print(f"   Next words: {context}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("""
The segment-based approach is struggling because:
1. Natural pauses don't align with ayah boundaries
2. Segments are too coarse (merging multiple ayahs)

Consider:
- Reducing min_segment_words to 2 or 3
- Using smaller min_gap (0.25s instead of 0.4s)
- OR: Use word-level classification instead
""")


if __name__ == "__main__":
    main()
