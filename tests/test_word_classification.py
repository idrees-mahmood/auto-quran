#!/usr/bin/env python3
"""
Test script for word-level classification algorithm.

Tests the new word-level classification with Anfal transcription 
which contains repeated ayahs.
"""

import json
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_processing_utils import TranscribedWord, load_quran_text
from alignment_utils import AyahDetector, WordClassification, reconstruct_ayahs

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_transcription(trans_path: str) -> list:
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


def test_word_classification():
    """Test word-level classification on Anfal."""
    
    base_dir = Path(__file__).parent.parent
    trans_path = base_dir / "data/transcriptions/7c82281ffe342bba_Sheikh Musa Anfal_processed.wav.json"
    quran_path = base_dir / "data/quran/quran.json"
    
    # Load data
    logger.info(f"Loading transcription...")
    words = load_transcription(str(trans_path))
    logger.info(f"Loaded {len(words)} words")
    
    logger.info(f"Loading Quran data...")
    quran_data = load_quran_text(str(quran_path))
    
    # Create detector
    detector = AyahDetector(quran_data=quran_data, confidence_threshold=0.5)
    
    # Run word classification
    logger.info("\n" + "="*60)
    logger.info("WORD-LEVEL CLASSIFICATION")
    logger.info("="*60)
    
    classifications = detector.classify_transcription_words(
        transcribed_words=words,
        surah=8,
        start_ayah=1,
        end_ayah=10  # Just first 10 ayahs for detailed view
    )
    
    logger.info(f"\nClassified {len(classifications)} words")
    
    # Show first 30 classifications
    logger.info("\n" + "="*60)
    logger.info("FIRST 30 WORD CLASSIFICATIONS")
    logger.info("="*60)
    
    for c in classifications[:30]:
        occ_str = f" (#{c.occurrence})" if c.occurrence > 1 else ""
        conf_str = f"{c.confidence:.0%}" if c.confidence > 0 else "LOW"
        logger.info(
            f"  Word {c.word_index:3}: Ayah {c.ayah}:{c.ayah_word_index}{occ_str} "
            f"[{conf_str}] '{c.transcribed_text}' → '{c.reference_text}'"
        )
    
    # Reconstruct ayahs
    logger.info("\n" + "="*60)
    logger.info("RECONSTRUCTED AYAHS")
    logger.info("="*60)
    
    # Load raw quran data for reconstruction
    with open(quran_path, 'r', encoding='utf-8') as f:
        raw_quran = json.load(f)
    
    ayah_events = reconstruct_ayahs(classifications, raw_quran)
    
    for event in ayah_events:
        occ_str = f" (#{event['occurrence']})" if event['occurrence'] > 1 else ""
        partial_str = f" [{event['partial_type']}]" if event['is_partial'] else ""
        
        logger.info(
            f"\n  Ayah {event['ayah']}{occ_str}{partial_str}"
            f" @ {event['start_time']:.1f}s - {event['end_time']:.1f}s"
            f" ({event['word_count']}/{event['reference_word_count']} words, "
            f"{event['confidence']:.0%} conf)"
        )
        logger.info(f"    Reference: {event['reference_text'][:60]}...")
        logger.info(f"    Transcribed: {event['transcribed_text'][:60]}...")
    
    # Count repetitions
    repetitions = [e for e in ayah_events if e['occurrence'] > 1]
    logger.info(f"\n✓ Found {len(repetitions)} repeated ayah occurrences")
    
    return True


def main():
    logger.info("="*60)
    logger.info("WORD-LEVEL CLASSIFICATION TEST")
    logger.info("="*60)
    
    test_word_classification()
    
    logger.info("\n✓ Test completed!")


if __name__ == "__main__":
    main()
