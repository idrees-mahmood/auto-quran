#!/usr/bin/env python3
"""
Test script for the repetition-aware ayah detection algorithm.

Tests the new algorithm with the Sheikh Musa Anfal transcription,
which contains repeated ayahs.
"""

import json
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio_processing_utils import TranscribedWord, load_quran_text
from alignment_utils import AyahDetector, segment_by_pauses, RecitationEvent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_transcription(trans_path: str) -> list:
    """Load transcription and extract words."""
    with open(trans_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    words = []
    trans_data = data.get("transcription", data)
    
    # Try segments first
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


def test_anfal_repetition():
    """Test the algorithm with Sheikh Musa Anfal transcription."""
    
    # Paths
    base_dir = Path(__file__).parent.parent
    trans_path = base_dir / "data/transcriptions/7c82281ffe342bba_Sheikh Musa Anfal_processed.wav.json"
    quran_path = base_dir / "data/quran/quran.json"
    
    if not trans_path.exists():
        logger.error(f"Transcription not found: {trans_path}")
        return False
    
    # Load data
    logger.info(f"Loading transcription from {trans_path}")
    words = load_transcription(str(trans_path))
    logger.info(f"Loaded {len(words)} words")
    
    logger.info(f"Loading Quran data from {quran_path}")
    quran_data = load_quran_text(str(quran_path))
    
    # Create detector
    detector = AyahDetector(quran_data=quran_data, confidence_threshold=0.65)
    
    # Test segmentation first
    logger.info("\n" + "="*60)
    logger.info("TESTING SEGMENTATION")
    logger.info("="*60)
    segments = segment_by_pauses(words)
    logger.info(f"Found {len(segments)} segments")
    
    # Show first few segments
    for i, (start, end) in enumerate(segments[:10]):
        seg_words = words[start:end]
        text_preview = " ".join(w.word for w in seg_words)[:50]
        logger.info(f"  Segment {i}: words {start}-{end} ({end-start} words): {text_preview}...")
    
    # Run detection
    logger.info("\n" + "="*60)
    logger.info("TESTING REPETITION-AWARE DETECTION")
    logger.info("="*60)
    
    # Surah Al-Anfal is chapter 8, we expect to detect ayahs 1-75
    detected = detector.detect_ayahs_from_transcription(
        transcribed_words=words,
        surah_hint=8,
        start_ayah=1,
        end_ayah=75,
        allow_repetition=True  # Enable repetition detection for this test
    )
    
    logger.info(f"\nDetected {len(detected)} recitation events")
    
    # Analyze results
    unique_ayahs = set()
    repetitions = []
    partials = []
    
    for event in detected:
        ayah = event["ayah"]
        occurrence = event.get("occurrence", 1)
        is_partial = event.get("is_partial", False)
        
        if occurrence > 1:
            repetitions.append(event)
        if is_partial:
            partials.append(event)
        unique_ayahs.add(ayah)
    
    logger.info(f"Unique ayahs detected: {len(unique_ayahs)}")
    logger.info(f"Repetitions detected: {len(repetitions)}")
    logger.info(f"Partial matches: {len(partials)}")
    
    # Show all events
    logger.info("\n" + "="*60)
    logger.info("ALL DETECTED EVENTS")
    logger.info("="*60)
    
    for i, event in enumerate(detected):
        ayah = event["ayah"]
        occ = event.get("occurrence", 1)
        conf = event.get("confidence", 0)
        is_partial = event.get("is_partial", False)
        partial_type = event.get("partial_type", "full")
        start_time = event.get("start_time", 0)
        
        occ_str = f" (#{occ})" if occ > 1 else ""
        partial_str = f" [{partial_type}]" if is_partial else ""
        
        logger.info(f"  {i+1:3}. Ayah {ayah:2}{occ_str}{partial_str} @ {start_time:.1f}s (conf: {conf:.0%})")
    
    # Show repetitions in detail
    if repetitions:
        logger.info("\n" + "="*60)
        logger.info("REPETITIONS DETECTED")
        logger.info("="*60)
        
        for event in repetitions:
            logger.info(f"  Ayah {event['ayah']} (occurrence #{event['occurrence']})")
            logger.info(f"    Time: {event['start_time']:.1f}s - {event['end_time']:.1f}s")
            logger.info(f"    Text: {event['transcribed_text'][:60]}...")
    
    return len(detected) > 0


def main():
    """Run all tests."""
    logger.info("="*60)
    logger.info("REPETITION-AWARE DETECTION TEST")
    logger.info("="*60)
    
    success = test_anfal_repetition()
    
    if success:
        logger.info("\n✓ Test completed successfully!")
    else:
        logger.error("\n✗ Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
