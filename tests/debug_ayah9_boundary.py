#!/usr/bin/env python3
"""
Find exact Ayah 9 boundaries and understand segment overlap.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_processing_utils import TranscribedWord, load_quran_text
from alignment_utils import AyahDetector, ArabicNormalizer, segment_by_pauses

def load_transcription(trans_path: str):
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


def main():
    base_dir = Path(__file__).parent.parent
    trans_path = base_dir / "data/transcriptions/7c82281ffe342bba_Sheikh Musa Anfal_processed.wav.json"
    quran_path = base_dir / "data/quran/quran.json"
    
    words = load_transcription(str(trans_path))
    quran_data = load_quran_text(str(quran_path))
    normalizer = ArabicNormalizer()
    detector = AyahDetector(quran_data=quran_data, normalizer=normalizer, confidence_threshold=0.65)
    
    # Get reference ayahs
    print("=" * 80)
    print("REFERENCE AYAHS")
    print("=" * 80)
    
    for ayah_num in range(9, 16):
        data = detector.corpus.get((8, ayah_num))
        if data:
            words_list = data["display"].split()
            print(f"\n  Ayah {ayah_num} ({len(words_list)} words):")
            print(f"    First 3: {' '.join(words_list[:3])}")
            print(f"    Last 3:  {' '.join(words_list[-3:])}")
    
    # Find Ayah 9 start (إِذْ تَسْتَغِيثُونَ)
    print("\n" + "=" * 80)
    print("SEARCHING FOR AYAH 9 START: إذ تستغيثون")
    print("=" * 80)
    
    ayah_9_start_pattern = ["اذ", "تستغيثون"]  # Normalized forms
    
    for i in range(len(words) - 1):
        w1_norm = normalizer.normalize(words[i].word)
        w2_norm = normalizer.normalize(words[i+1].word)
        
        if "اذ" in w1_norm and "تستغ" in w2_norm:
            print(f"\n  FOUND at word {i} @ {words[i].start:.2f}s")
            print(f"  Context (15 words):")
            for j in range(i, min(i+15, len(words))):
                print(f"    [{j:4}] {words[j].start:7.2f}s: {words[j].word}")
    
    # Find Ayah 9 end (مُرْدِفِينَ)
    print("\n" + "=" * 80)
    print("SEARCHING FOR AYAH 9 END: مردفين")
    print("=" * 80)
    
    for i in range(len(words)):
        w_norm = normalizer.normalize(words[i].word)
        if "مردف" in w_norm:
            print(f"\n  FOUND at word {i} @ {words[i].start:.2f}s: {words[i].word}")
            print(f"  Next 10 words (should be Ayah 10: وما جعله...):")
            for j in range(i+1, min(i+11, len(words))):
                print(f"    [{j:4}] {words[j].start:7.2f}s: {words[j].word}")
    
    # Now check what segment 12 contains
    print("\n" + "=" * 80)
    print("SEGMENT 12 ANALYSIS")
    print("=" * 80)
    
    segments = segment_by_pauses(words, min_gap_seconds=0.4, min_segment_words=4)
    
    if len(segments) > 12:
        start_idx, end_idx = segments[12]
        seg_words = words[start_idx:end_idx]
        
        print(f"\n  Segment 12: words {start_idx} to {end_idx} ({end_idx - start_idx} words)")
        print(f"  Time: {seg_words[0].start:.2f}s to {seg_words[-1].end:.2f}s")
        print(f"\n  Words in segment:")
        for i, w in enumerate(seg_words):
            idx = start_idx + i
            print(f"    [{idx:4}] {w.start:7.2f}s - {w.end:7.2f}s: {w.word}")
    
    # Check the gaps around segment 12
    print("\n" + "=" * 80)
    print("GAPS ANALYSIS - AROUND SEGMENT 12")
    print("=" * 80)
    
    for seg_idx in range(10, min(15, len(segments))):
        start_idx, end_idx = segments[seg_idx]
        seg_text = " ".join(w.word for w in words[start_idx:end_idx])
        
        if seg_idx + 1 < len(segments):
            next_start = segments[seg_idx + 1][0]
            gap_words = next_start - end_idx
            if gap_words > 0:
                gap_text = " ".join(w.word for w in words[end_idx:next_start])
                gap_time = words[next_start].start - words[end_idx-1].end
                print(f"\n  Segment {seg_idx}: {end_idx - start_idx} words")
                print(f"    Text: {seg_text[:60]}...")
                print(f"    GAP: {gap_words} words, {gap_time:.2f}s")
                print(f"    Gap text: {gap_text}")
            else:
                print(f"\n  Segment {seg_idx}: {end_idx - start_idx} words (no gap)")
                print(f"    Text: {seg_text[:60]}...")


if __name__ == "__main__":
    main()
