"""
Ayah matching and word alignment algorithms for Quran transcription.

This module implements fuzzy matching between Whisper transcription
and Quran reference text, with word-level alignment.

Supports repetition detection: handles cases where Qaris repeat ayahs
or go backwards in the recitation.
"""

import json
import difflib
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

from src.audio_processing_utils import (
    TranscribedWord, AyahMatch, ArabicNormalizer, 
    load_quran_text
)

try:
    from rapidfuzz import fuzz, process as fuzzy_process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RecitationEvent:
    """
    A single occurrence of an ayah being recited.

    Supports repetition: the same ayah can appear multiple times with
    different occurrence numbers.
    """
    surah: int
    ayah: int
    occurrence: int          # 1st, 2nd, 3rd time this ayah appears in recitation
    start_time: float
    end_time: float
    confidence: float
    transcribed_text: str
    word_indices: Tuple[int, int]
    is_partial: bool = False         # True if only part of the ayah was recited
    partial_type: str = "full"       # "start", "middle", "end", "full"
    reference_word_count: int = 0    # Total words in the reference ayah
    event_type: str = "full"         # "full" | "partial" | "repetition" | "skip"

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


@dataclass
class WordClassification:
    """
    Classification of a single transcribed word to its position in the Quran.
    
    This is the fundamental unit for building accurate ayah timings.
    Each transcribed word is mapped to:
    - Which ayah it belongs to
    - Its position within that ayah
    - The accurate Quran text (not the potentially inaccurate transcription)
    - Which occurrence if the ayah is repeated
    """
    word_index: int              # Index in the transcribed_words list
    surah: int
    ayah: int
    ayah_word_index: int         # Position within ayah (0-indexed)
    occurrence: int              # 1st, 2nd, 3rd time this ayah appears
    confidence: float            # Match confidence (0.0 - 1.0)
    transcribed_text: str        # What Whisper heard (may be inaccurate)
    reference_text: str          # The actual Quran word (accurate)
    start_time: float
    end_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "word_index": self.word_index,
            "surah": self.surah,
            "ayah": self.ayah,
            "ayah_word_index": self.ayah_word_index,
            "occurrence": self.occurrence,
            "confidence": self.confidence,
            "transcribed_text": self.transcribed_text,
            "reference_text": self.reference_text,
            "start_time": self.start_time,
            "end_time": self.end_time
        }


# ============================================================================
# Segmentation Utilities  
# ============================================================================

def segment_by_pauses(
    transcribed_words: List[TranscribedWord],
    min_gap_seconds: float = 0.4,
    min_segment_words: int = 3,
    max_segment_words: int = 50
) -> List[Tuple[int, int]]:
    """
    Split transcription into segments at natural pause points.
    
    Pauses often indicate ayah boundaries in Quran recitation.
    
    Args:
        transcribed_words: List of words with timestamps
        min_gap_seconds: Minimum gap between words to consider a pause
        min_segment_words: Minimum words per segment (avoids tiny fragments)
        max_segment_words: Maximum words per segment (forces splits for long sequences)
        
    Returns:
        List of (start_idx, end_idx) tuples representing segments
    """
    if not transcribed_words:
        return []
    
    segments = []
    current_start = 0
    
    for i in range(1, len(transcribed_words)):
        # Calculate gap between current and previous word
        gap = transcribed_words[i].start - transcribed_words[i-1].end
        words_in_segment = i - current_start
        
        # Split at pause if:
        # 1. Gap is large enough AND we have enough words, OR
        # 2. Segment is getting too long
        should_split = (
            (gap >= min_gap_seconds and words_in_segment >= min_segment_words) or
            words_in_segment >= max_segment_words
        )
        
        if should_split:
            segments.append((current_start, i))
            current_start = i
    
    # Add final segment
    if current_start < len(transcribed_words):
        segments.append((current_start, len(transcribed_words)))
    
    logger.debug(f"Segmented {len(transcribed_words)} words into {len(segments)} segments")
    return segments


# ============================================================================
# Ayah Detection
# ============================================================================



class AyahDetector:
    """
    Detects which surah:ayah corresponds to transcribed text segments.
    Uses fuzzy matching against Quran corpus.
    """
    
    def __init__(
        self,
        quran_data: Dict[str, Dict[str, Dict[str, str]]],
        normalizer: Optional[ArabicNormalizer] = None,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize ayah detector.
        
        Args:
            quran_data: Quran text dictionary from load_quran_text()
            normalizer: Arabic text normalizer (creates default if None)
            confidence_threshold: Minimum confidence score for auto-matching (0-1)
        """
        self.quran_data = quran_data
        self.normalizer = normalizer or ArabicNormalizer()
        self.confidence_threshold = confidence_threshold
        
        # Build normalized corpus for faster searching
        self._build_normalized_corpus()
        
    def _build_normalized_corpus(self):
        """Pre-compute normalized text for all ayahs."""
        logger.info("Building normalized Quran corpus...")
        
        self.corpus = {}  # {(surah, ayah): {"normalized": str, "display": str, "words": [str]}}
        
        for surah_str, surah_data in self.quran_data.items():
            surah = int(surah_str)
            for ayah_str, ayah_data in surah_data.items():
                ayah = int(ayah_str)
                
                # Get text (without diacritics) and displayText (with diacritics)
                text_clean = ayah_data.get("text", "")
                text_display = ayah_data.get("displayText", "").replace("\r", "")
                
                # Normalize for matching
                normalized = self.normalizer.normalize(text_clean)
                
                # Split into words
                words = text_display.split()
                
                self.corpus[(surah, ayah)] = {
                    "normalized": normalized,
                    "display": text_display,
                    "clean": text_clean,
                    "words": words
                }
        
        logger.info(f"Corpus built: {len(self.corpus)} ayahs indexed")
    
    def find_best_match(
        self,
        transcribed_text: str,
        surah_hint: Optional[int] = None,
        ayah_range: Optional[Tuple[int, int]] = None
    ) -> Optional[Tuple[int, int, float]]:
        """
        Find best matching ayah for transcribed text.
        
        Args:
            transcribed_text: Normalized transcribed text
            surah_hint: If provided, only search within this surah
            ayah_range: If provided, only search within (start_ayah, end_ayah) range
            
        Returns:
            Tuple of (surah, ayah, confidence_score) or None if no good match
        """
        normalized_transcription = self.normalizer.normalize(transcribed_text)
        
        if not normalized_transcription:
            return None
        
        # Filter corpus based on hints
        search_corpus = []
        for (surah, ayah), data in self.corpus.items():
            # Apply surah filter
            if surah_hint is not None and surah != surah_hint:
                continue
            
            # Apply ayah range filter
            if ayah_range is not None:
                start_ayah, end_ayah = ayah_range
                if not (start_ayah <= ayah <= end_ayah):
                    continue
            
            search_corpus.append(((surah, ayah), data["normalized"]))
        
        if not search_corpus:
            logger.warning("No ayahs in search corpus after filtering")
            return None
        
        # Use RapidFuzz if available, otherwise difflib
        if RAPIDFUZZ_AVAILABLE:
            matches = fuzzy_process.extract(
                normalized_transcription,
                [text for _, text in search_corpus],
                scorer=fuzz.ratio,
                limit=5
            )
            
            if matches:
                best_match_text, best_score, best_idx = matches[0]
                best_surah_ayah = search_corpus[best_idx][0]
                confidence = best_score / 100.0
                
                logger.info(f"Best match: {best_surah_ayah} (confidence: {confidence:.2%})")
                return (*best_surah_ayah, confidence)
        else:
            # Fallback to difflib
            best_ratio = 0.0
            best_match = None
            
            for (surah, ayah), text in search_corpus:
                ratio = difflib.SequenceMatcher(None, normalized_transcription, text).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = (surah, ayah, ratio)
            
            if best_match and best_match[2] >= self.confidence_threshold:
                logger.info(f"Best match: {best_match[0]}:{best_match[1]} (confidence: {best_match[2]:.2%})")
                return best_match
        
        return None
    
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
        allow_repetition: bool = False  # Enable for recitations with repeated ayahs
    ) -> List[Dict[str, Any]]:

        """
        Detect ayahs from a continuous transcription.
        
        Args:
            transcribed_words: List of transcribed words with timestamps
            window_size: Number of words to use for matching at a time
            overlap: Number of words to overlap between windows
            surah_hint: If provided, only search within this surah
            sequential_mode: If True, detects ayahs sequentially (fills gaps)
            start_ayah: Starting ayah number (default 1, for recitations that don't start from beginning)
            end_ayah: Ending ayah number (default None = until end of surah)
            skip_preamble: If True, skip isti'adha/basmallah at start
            allow_repetition: If True, use new repetition-aware algorithm (default)
            
        Returns:
            List of detected ayah segments with timing info.
            If allow_repetition=True, returns RecitationEvent-style dicts with
            'occurrence' and 'is_partial' fields.
        """
        if surah_hint:
            end_str = f" to {end_ayah}" if end_ayah else ""
            logger.info(f"Detecting ayahs from {len(transcribed_words)} words (Surah {surah_hint}, Ayah {start_ayah}{end_str})")
        else:
            logger.info(f"Detecting ayahs from {len(transcribed_words)} transcribed words")
        
        # Use repetition-aware algorithm when explicitly enabled
        # Note: This algorithm handles repetitions well but may merge short ayahs
        # For surahs with many short ayahs, use allow_repetition=False
        if allow_repetition and surah_hint:
            events = self._detect_with_repetition(
                transcribed_words, 
                surah_hint, 
                start_ayah, 
                end_ayah, 
                skip_preamble
            )
            # Convert RecitationEvents to dict format for backward compatibility
            return [event.to_dict() for event in events]
        
        # Default: Use proven sequential algorithm
        if sequential_mode and surah_hint:
            return self._detect_sequential(transcribed_words, surah_hint, window_size, start_ayah, end_ayah, skip_preamble)
        else:
            return self._detect_sliding_window(transcribed_words, window_size, overlap, surah_hint)
    


    def _detect_sequential(
        self,
        transcribed_words: List[TranscribedWord],
        surah: int,
        window_size: int,
        start_ayah: int = 1,
        end_ayah: Optional[int] = None,
        skip_preamble: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Sequential detection: tries each ayah in order with strict word boundaries.
        
        The key insight is that we know the EXACT word count of each ayah from
        the Quran corpus. We use this to prevent "word stealing" from adjacent ayahs.
        
        Algorithm:
        1. For each expected ayah, get exact reference word count
        2. Try windows of size [ref_count - 2, ref_count + 3]
        3. Score each window using FUZZY word-level alignment (tolerant of Whisper errors)
        4. Penalize windows that have unaligned trailing words (likely from next ayah)
        5. Validate by checking if trailing words match next ayah's start
        6. When stuck, try searching forward in transcription
        
        Args:
            transcribed_words: List of transcribed words
            surah: Surah number to match
            window_size: Window size for matching (used as max extension)
            start_ayah: Which ayah to start from (default 1)
            end_ayah: Which ayah to end at (default None = until end of surah)
            skip_preamble: Whether to skip isti'adha/basmallah at start
        """
        detected_ayahs = []
        i = 0
        current_ayah = start_ayah
        max_skips = 3  # Reduced from 5 - fail faster and search forward
        consecutive_skips = 0
        words_since_last_match = 0
        
        # Determine actual end ayah
        max_ayah_in_surah = max(ayah for (s, ayah) in self.corpus.keys() if s == surah) if any(s == surah for s, _ in self.corpus.keys()) else 286
        actual_end_ayah = end_ayah if end_ayah else max_ayah_in_surah
        
        logger.info(f"Sequential detection: Surah {surah}, Ayah {start_ayah} to {actual_end_ayah}")
        
        # Check for isti'adha/basmallah at start
        if skip_preamble:
            i = self._skip_opening_formulas(transcribed_words)
            if i > 0:
                logger.info(f"Skipped {i} words (isti'adha/basmallah)")
        
        # Pre-load ayah info for lookahead validation
        def get_ayah_info(ayah_num: int) -> Optional[Dict]:
            data = self.corpus.get((surah, ayah_num))
            if data:
                words = data["display"].split()
                norm_words = [self.normalizer.normalize(w) for w in words]
                return {"display": data["display"], "normalized": data["normalized"], 
                        "words": words, "norm_words": norm_words, "count": len(words)}
            return None
        
        while i < len(transcribed_words) and current_ayah <= actual_end_ayah:
            current_info = get_ayah_info(current_ayah)
            if not current_info:
                # No more ayahs in this surah
                logger.info(f"Reached end of surah at ayah {current_ayah}")
                break
            
            next_info = get_ayah_info(current_ayah + 1)
            ref_count = current_info["count"]
            
            # Debug logging
            remaining = len(transcribed_words) - i
            preview = " ".join(w.word for w in transcribed_words[i:i+min(8, remaining)])
            logger.debug(f"Ayah {current_ayah} ({ref_count} words): expecting '{current_info['display'][:40]}...'")
            logger.debug(f"  Transcription position {i}: '{preview}...'")
            
            # Find best match with strict word boundary validation
            best_result = self._find_strict_match(
                transcribed_words=transcribed_words,
                start_idx=i,
                current_ayah_info=current_info,
                next_ayah_info=next_info,
                surah=surah,
                ayah=current_ayah
            )
            
            if best_result:
                consumed, confidence, matched_text = best_result
                
                window_words = transcribed_words[i:i + consumed]
                detected_ayahs.append({
                    "surah": surah,
                    "ayah": current_ayah,
                    "confidence": confidence,
                    "start_time": window_words[0].start,
                    "end_time": window_words[-1].end,
                    "transcribed_text": matched_text,
                    "word_indices": (i, i + consumed),
                    "reference_word_count": ref_count
                })
                
                logger.info(f"✓ Ayah {current_ayah}: matched {consumed}/{ref_count} words @ {confidence:.0%}")
                i += consumed
                current_ayah += 1
                consecutive_skips = 0
                words_since_last_match = 0
            else:
                consecutive_skips += 1
                words_since_last_match += 1
                logger.warning(f"✗ Ayah {current_ayah}: no confident match found")
                
                # If we've skipped too many ayahs, we're probably out of sync
                if consecutive_skips >= max_skips:
                    logger.warning(f"Skipped {max_skips} ayahs - trying to resync...")
                    
                    # Strategy 1: Search forward in transcription for ANY matching ayah
                    found_sync = False
                    search_ahead = min(20, len(transcribed_words) - i)  # Look up to 20 words ahead
                    
                    for offset in range(1, search_ahead + 1):
                        test_idx = i + offset
                        # Try to find which ayah matches at this position
                        for test_ayah in range(current_ayah, min(current_ayah + 10, actual_end_ayah + 1)):
                            test_info = get_ayah_info(test_ayah)
                            if not test_info:
                                continue
                            test_next = get_ayah_info(test_ayah + 1)
                            
                            result = self._find_strict_match(
                                transcribed_words=transcribed_words,
                                start_idx=test_idx,
                                current_ayah_info=test_info,
                                next_ayah_info=test_next,
                                surah=surah,
                                ayah=test_ayah
                            )
                            
                            if result and result[1] >= 0.6:  # Good confidence
                                logger.info(f"🔄 Resync found: Ayah {test_ayah} at position {test_idx} (skipped {offset} words)")
                                i = test_idx
                                current_ayah = test_ayah
                                consecutive_skips = 0
                                found_sync = True
                                break
                        
                        if found_sync:
                            break
                    
                    if not found_sync:
                        # Strategy 2: Just advance position and continue
                        logger.warning(f"Could not resync, advancing 1 word")
                        i += 1
                        consecutive_skips = 0  # Reset to try again
                else:
                    # Haven't hit max skips yet, just move to next ayah
                    current_ayah += 1
        
        logger.info(f"Sequential detection complete: found {len(detected_ayahs)} ayahs")
        return detected_ayahs
    
    def _find_strict_match(
        self,
        transcribed_words: List[TranscribedWord],
        start_idx: int,
        current_ayah_info: Dict,
        next_ayah_info: Optional[Dict],
        surah: int,
        ayah: int
    ) -> Optional[Tuple[int, float, str]]:
        """
        Find the best match for an ayah with strict word boundary validation.
        
        Returns:
            Tuple of (consumed_word_count, confidence, matched_text) or None
        """
        ref_count = current_ayah_info["count"]
        ref_norm_words = current_ayah_info["norm_words"]
        remaining = len(transcribed_words) - start_idx
        
        if remaining == 0:
            return None
        
        candidates = []
        
        # Try window sizes centered around reference count
        # Allow slight variation: [ref_count - 2, ref_count + 1]
        # Using +1 (not +3) to avoid consuming words from next ayah
        min_window = max(1, ref_count - 2)
        max_window = min(remaining, ref_count + 1)
        
        for window_size in range(min_window, max_window + 1):
            window_words = transcribed_words[start_idx:start_idx + window_size]
            window_text = " ".join(w.word for w in window_words)
            window_norm = [self.normalizer.normalize(w.word) for w in window_words]
            
            # Calculate word-level alignment score
            alignment_score, aligned_count, extra_words = self._calculate_word_alignment_score(
                window_norm, ref_norm_words
            )
            
            # Check if extra words belong to next ayah (boundary violation)
            boundary_penalty = 0.0
            if extra_words and next_ayah_info:
                boundary_penalty = self._check_boundary_violation(
                    extra_words, next_ayah_info["norm_words"]
                )
            
            # Calculate final score with penalty
            # Prefer windows close to reference count
            size_penalty = abs(window_size - ref_count) * 0.05  # 5% per word difference
            final_score = alignment_score - boundary_penalty - size_penalty
            
            # Also do a fuzzy check for sanity
            fuzzy_score = self._quick_fuzzy_score(window_text, current_ayah_info["normalized"])
            
            # Combine scores (word alignment is more important)
            combined_score = (alignment_score * 0.7) + (fuzzy_score * 0.3) - boundary_penalty - size_penalty
            
            logger.debug(
                f"  Window[{window_size}]: align={alignment_score:.2f}, fuzzy={fuzzy_score:.2f}, "
                f"boundary_pen={boundary_penalty:.2f}, size_pen={size_penalty:.2f} -> {combined_score:.2f}"
            )
            
            if combined_score >= self.confidence_threshold:
                candidates.append((window_size, combined_score, window_text, extra_words))
        
        if not candidates:
            return None
        
        # Select best candidate
        # Prefer candidates with no boundary violations when scores are close
        candidates.sort(key=lambda x: (x[1], -len(x[3]) if x[3] else 0), reverse=True)
        best = candidates[0]
        
        return (best[0], best[1], best[2])
    
    def _calculate_word_alignment_score(
        self,
        transcribed_norm: List[str],
        reference_norm: List[str]
    ) -> Tuple[float, int, List[str]]:
        """
        Calculate word-level alignment score using fuzzy word matching.
        
        This is more tolerant of Whisper transcription errors like:
        - سجر vs سدر (phonetic confusion)
        - مخطود vs مخضود (similar sounds)
        
        Returns:
            Tuple of (score, aligned_count, extra_words_at_end)
        """
        if len(reference_norm) == 0:
            return 0.0, 0, []
        
        if len(transcribed_norm) == 0:
            return 0.0, 0, []
        
        # Use fuzzy matching for each word pair
        # This is more tolerant than exact word matching
        matched_count = 0
        used_ref_indices = set()
        
        for trans_word in transcribed_norm:
            best_match_score = 0
            best_match_idx = -1
            
            for ref_idx, ref_word in enumerate(reference_norm):
                if ref_idx in used_ref_indices:
                    continue
                    
                # Calculate similarity between words
                if RAPIDFUZZ_AVAILABLE:
                    score = fuzz.ratio(trans_word, ref_word) / 100.0
                else:
                    score = difflib.SequenceMatcher(None, trans_word, ref_word).ratio()
                
                if score > best_match_score:
                    best_match_score = score
                    best_match_idx = ref_idx
            
            # Consider it a match if similarity > 0.6 (allows for transcription errors)
            if best_match_score > 0.6 and best_match_idx >= 0:
                matched_count += 1
                used_ref_indices.add(best_match_idx)
        
        coverage = matched_count / len(reference_norm)
        
        # Find extra words at the end
        extra_words = []
        if len(transcribed_norm) > len(reference_norm):
            extra_words = transcribed_norm[len(reference_norm):]
        
        return coverage, matched_count, extra_words
    
    def _check_boundary_violation(
        self,
        extra_words: List[str],
        next_ayah_norm_words: List[str]
    ) -> float:
        """
        Check if extra words at the end match the start of next ayah.
        Returns a penalty score (0.0 to 0.5).
        """
        if not extra_words or not next_ayah_norm_words:
            return 0.0
        
        # Check how many extra words match the start of next ayah
        matches = 0
        for i, word in enumerate(extra_words):
            if i < len(next_ayah_norm_words):
                # Use fuzzy matching for individual words
                ratio = difflib.SequenceMatcher(None, word, next_ayah_norm_words[i]).ratio()
                if ratio > 0.8:
                    matches += 1
        
        if matches > 0:
            # Significant penalty if extra words clearly belong to next ayah
            penalty = min(0.4, matches * 0.15)  # Up to 0.4 penalty
            logger.debug(f"  Boundary violation: {matches}/{len(extra_words)} words match next ayah start -> penalty={penalty:.2f}")
            return penalty
        
        return 0.0
    
    def _quick_fuzzy_score(self, text1: str, text2: str) -> float:
        """Quick fuzzy score between two texts."""
        if RAPIDFUZZ_AVAILABLE:
            return fuzz.ratio(text1, text2) / 100.0
        else:
            return difflib.SequenceMatcher(None, text1, text2).ratio()
    
    def _skip_opening_formulas(self, transcribed_words: List[TranscribedWord]) -> int:
        """
        Detect and skip isti'adha (أعوذ بالله) and basmallah (بسم الله) at start.
        Returns the index to start actual ayah detection from.
        
        Only skips if the ACTUAL formulas are detected in sequence at the start.
        """
        if not transcribed_words or len(transcribed_words) < 2:
            return 0
        
        # Normalize first 15 words for matching
        first_words = [
            self.normalizer.normalize(w.word) 
            for w in transcribed_words[:min(15, len(transcribed_words))]
        ]
        first_text = " ".join(first_words)
        
        skip_count = 0
        
        # Isti'adha patterns (must appear at start)
        # "أعوذ بالله من الشيطان الرجيم"
        istiadha_pattern = "اعوذ بالله من الشيطان الرجيم"
        
        # Basmallah pattern 
        # "بسم الله الرحمن الرحيم"
        basmallah_pattern = "بسم الله الرحمن الرحيم"
        
        # Check for isti'adha at start
        if "اعوذ بالله" in first_text[:50]:
            # Find how many words make up the isti'adha
            for i, word in enumerate(first_words):
                if word in {'اعوذ', 'بالله', 'من', 'الشيطان', 'الرجيم'}:
                    skip_count = i + 1
                    # Stop after finding الرجيم or after checking first 7 words
                    if word == 'الرجيم' or i >= 6:
                        break
                elif skip_count > 0:
                    # We've moved past isti'adha
                    break
        
        # Check for basmallah after isti'adha (or at start if no isti'adha)
        remaining_words = first_words[skip_count:]
        if remaining_words and "بسم الله" in " ".join(remaining_words[:6]):
            for i, word in enumerate(remaining_words):
                if word in {'بسم', 'الله', 'الرحمن', 'الرحيم'}:
                    skip_count = skip_count + i + 1
                    # Stop after finding الرحيم or after 4 basmallah words
                    if word == 'الرحيم' or i >= 3:
                        break
                elif i > 0:
                    # We've moved past basmallah
                    break
        
        # Safety check: don't skip more than 12 words
        return min(skip_count, 12)

    
    def _find_actual_position(
        self, 
        transcribed_words: List[TranscribedWord], 
        start_idx: int,
        surah: int
    ) -> Optional[int]:
        """
        Find which ayah the transcription is actually at.
        Used when sequential matching fails repeatedly.
        """
        # Take a sample of words and find best match without ayah restriction
        sample_size = min(15, len(transcribed_words) - start_idx)
        if sample_size < 3:
            return None
        
        sample_words = transcribed_words[start_idx:start_idx + sample_size]
        sample_text = " ".join(w.word for w in sample_words)
        
        match = self.find_best_match(sample_text, surah_hint=surah)
        if match and match[2] >= 0.6:  # Require reasonable confidence
            logger.info(f"Detected actual position: Surah {match[0]} Ayah {match[1]} ({match[2]:.0%})")
            return match[1]
        
        return None
    
    def _detect_sliding_window(
        self,
        transcribed_words: List[TranscribedWord],
        window_size: int,
        overlap: int,
        surah_hint: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Original sliding window detection (for unknown surah)."""
        detected_ayahs = []
        i = 0
        
        while i < len(transcribed_words):
            window_end = min(i + window_size, len(transcribed_words))
            window_words = transcribed_words[i:window_end]
            window_text = " ".join(w.word for w in window_words)
            
            match = self.find_best_match(window_text, surah_hint=surah_hint)
            
            if match:
                surah, ayah, confidence = match
                
                detected_ayahs.append({
                    "surah": surah,
                    "ayah": ayah,
                    "confidence": confidence,
                    "start_time": window_words[0].start,
                    "end_time": window_words[-1].end,
                    "transcribed_text": window_text,
                    "word_indices": (i, window_end)
                })
                
                i += window_size - overlap
            else:
                i += 1
        
        logger.info(f"Sliding window: found {len(detected_ayahs)} ayah segments")
        return detected_ayahs
    
    # ========================================================================
    # Repetition-Aware Detection (New Algorithm)
    # ========================================================================
    
    def _detect_with_repetition(
        self,
        transcribed_words: List[TranscribedWord],
        surah: int,
        start_ayah: int = 1,
        end_ayah: Optional[int] = None,
        skip_preamble: bool = True,
        repetition_confidence_boost: float = 0.15
    ) -> List[RecitationEvent]:
        """
        Detect ayahs with support for repetitions and backwards jumps.
        
        This algorithm:
        1. Segments the transcription by natural pauses
        2. Matches each segment against ALL ayahs in range (not just next expected)
        3. Uses a bias toward "next expected" ayah but allows repetitions
        4. Requires higher confidence for repetitions (to avoid false positives)
        
        Args:
            transcribed_words: List of transcribed words with timestamps
            surah: Surah number to search within
            start_ayah: First ayah expected
            end_ayah: Last ayah expected (default: end of surah)
            skip_preamble: Whether to skip isti'adha/basmallah at start
            repetition_confidence_boost: Extra confidence required for repeats
            
        Returns:
            List of RecitationEvent objects in time order
        """
        if not transcribed_words:
            return []
        
        # Determine ayah range
        max_ayah = max(
            ayah for (s, ayah) in self.corpus.keys() if s == surah
        ) if any(s == surah for s, _ in self.corpus.keys()) else 286
        actual_end = end_ayah if end_ayah else max_ayah
        
        logger.info(f"Repetition-aware detection: Surah {surah}, Ayahs {start_ayah}-{actual_end}")
        
        # Skip opening formulas
        word_start_idx = 0
        if skip_preamble:
            word_start_idx = self._skip_opening_formulas(transcribed_words)
            if word_start_idx > 0:
                logger.info(f"Skipped {word_start_idx} words (isti'adha/basmallah)")
        
        working_words = transcribed_words[word_start_idx:]
        
        # Calculate average ayah length for this surah range to adapt segmentation
        total_words = 0
        ayah_count = 0
        for ayah_num in range(start_ayah, actual_end + 1):
            ayah_data = self.corpus.get((surah, ayah_num))
            if ayah_data:
                total_words += len(ayah_data.get("words", []))
                ayah_count += 1
        
        avg_ayah_words = total_words / ayah_count if ayah_count > 0 else 10
        
        # Adaptive segmentation: use smaller min_segment for surahs with short ayahs
        if avg_ayah_words <= 5:
            # Short ayah surah (like late Makkan surahs)
            min_seg = 2
            min_gap = 0.25
        elif avg_ayah_words <= 10:
            min_seg = 3
            min_gap = 0.3
        else:
            min_seg = 4
            min_gap = 0.4
        
        logger.debug(f"Avg ayah length: {avg_ayah_words:.1f} words, using min_segment={min_seg}")
        
        # Segment by pauses with adaptive parameters
        segments = segment_by_pauses(working_words, min_gap_seconds=min_gap, min_segment_words=min_seg)
        if not segments:
            logger.warning("No segments found after pause-based segmentation")
            return []
        
        logger.info(f"Found {len(segments)} segments from pause analysis")

        
        # Track occurrences of each ayah
        ayah_occurrences: Dict[int, int] = {}  # ayah_num -> count
        
        # Track what ayah we expect next (for bias)
        expected_next_ayah = start_ayah
        
        events: List[RecitationEvent] = []
        
        # Process segments, potentially splitting long ones
        pending_segments = list(segments)  # Segments still to process
        seg_idx = 0
        
        while pending_segments:
            seg_start, seg_end = pending_segments.pop(0)
            segment_words = working_words[seg_start:seg_end]
            segment_word_count = len(segment_words)
            
            if not segment_words:
                seg_idx += 1
                continue
            
            # Match segment against all candidate ayahs
            candidates = self._match_segment_to_ayahs(
                segment_words=segment_words,
                surah=surah,
                ayah_range=(start_ayah, actual_end),
                expected_next=expected_next_ayah,
                ayah_occurrences=ayah_occurrences,
                repetition_confidence_boost=repetition_confidence_boost
            )
            
            if not candidates:
                logger.debug(f"Segment {seg_idx}: No confident match found")
                seg_idx += 1
                continue
            
            # Take best candidate
            best = candidates[0]
            ayah_num = best["ayah"]
            confidence = best["confidence"]
            is_partial = best["is_partial"]
            partial_type = best["partial_type"]
            
            # Get reference word count
            ref_data = self.corpus.get((surah, ayah_num), {})
            ref_word_count = len(ref_data.get("words", []))
            
            # CHECK: Is segment significantly longer than the matched ayah?
            # If so, we should split the segment and only consume the ayah's words
            words_to_consume = min(segment_word_count, ref_word_count + 2)  # +2 tolerance
            
            if segment_word_count > ref_word_count + 3:
                # Segment is too long - split it!
                # Consume only the first ref_word_count words for this ayah
                words_to_consume = ref_word_count
                
                # Re-queue the remainder as a new segment to process
                remainder_start = seg_start + words_to_consume
                remainder_end = seg_end
                if remainder_start < remainder_end:
                    pending_segments.insert(0, (remainder_start, remainder_end))
                    logger.debug(
                        f"Segment {seg_idx}: Splitting - consuming {words_to_consume} words for Ayah {ayah_num}, "
                        f"re-queueing {remainder_end - remainder_start} remaining words"
                    )
                
                # Update segment_words to only include what we consume
                segment_words = working_words[seg_start:seg_start + words_to_consume]
            
            # Update occurrence count
            if ayah_num not in ayah_occurrences:
                ayah_occurrences[ayah_num] = 0
            ayah_occurrences[ayah_num] += 1
            occurrence = ayah_occurrences[ayah_num]
            
            # Create event
            # Adjust indices back to original transcribed_words
            orig_start = word_start_idx + seg_start
            orig_end = word_start_idx + seg_start + len(segment_words)
            
            event = RecitationEvent(
                surah=surah,
                ayah=ayah_num,
                occurrence=occurrence,
                start_time=segment_words[0].start,
                end_time=segment_words[-1].end,
                confidence=confidence,
                transcribed_text=" ".join(w.word for w in segment_words),
                word_indices=(orig_start, orig_end),
                is_partial=is_partial,
                partial_type=partial_type,
                reference_word_count=ref_word_count
            )
            events.append(event)
            
            # Log detection
            repeat_note = f" (occurrence #{occurrence})" if occurrence > 1 else ""
            partial_note = f" [{partial_type}]" if is_partial else ""
            logger.info(
                f"✓ Segment {seg_idx}: Ayah {ayah_num}{repeat_note}{partial_note} "
                f"@ {confidence:.0%}"
            )
            
            # Update expected next ayah
            if ayah_num == expected_next_ayah:
                expected_next_ayah = min(ayah_num + 1, actual_end)
            else:
                expected_next_ayah = min(ayah_num + 1, actual_end)
            
            seg_idx += 1
        
        logger.info(f"Repetition-aware detection complete: {len(events)} events")
        return events
    
    def _match_segment_to_ayahs(
        self,
        segment_words: List[TranscribedWord],
        surah: int,
        ayah_range: Tuple[int, int],
        expected_next: int,
        ayah_occurrences: Dict[int, int],
        repetition_confidence_boost: float = 0.15
    ) -> List[Dict[str, Any]]:
        """
        Match a segment against all ayahs in range and return ranked candidates.
        
        Args:
            segment_words: Words in this segment
            surah: Surah number
            ayah_range: (start_ayah, end_ayah) to search within
            expected_next: The ayah we expect next (gets a bias bonus)
            ayah_occurrences: How many times each ayah has been seen
            repetition_confidence_boost: Extra confidence required for repeats
            
        Returns:
            Sorted list of candidate matches with scores
        """
        segment_text = " ".join(w.word for w in segment_words)
        segment_norm = [self.normalizer.normalize(w.word) for w in segment_words]
        segment_norm_text = " ".join(segment_norm)
        
        candidates = []
        start_ayah, end_ayah = ayah_range
        
        for ayah_num in range(start_ayah, end_ayah + 1):
            ayah_data = self.corpus.get((surah, ayah_num))
            if not ayah_data:
                continue
            
            ref_words = ayah_data["words"]
            ref_norm_words = [self.normalizer.normalize(w) for w in ref_words]
            ref_norm_text = ayah_data["normalized"]
            
            # Calculate alignment score
            alignment_score, matched_count, extra_words = self._calculate_word_alignment_score(
                segment_norm, ref_norm_words
            )
            
            # Calculate fuzzy text score
            fuzzy_score = self._quick_fuzzy_score(segment_norm_text, ref_norm_text)
            
            # Combine scores
            base_score = (alignment_score * 0.7) + (fuzzy_score * 0.3)
            
            # Detect partial match type
            is_partial, partial_type, partial_coverage = self._detect_partial_match(
                segment_norm, ref_norm_words
            )
            
            # Apply biases
            final_score = base_score
            
            # Bias 1: Bonus for expected next ayah
            if ayah_num == expected_next:
                final_score += 0.10  # 10% bonus for expected next
            
            # Bias 2: Penalty for repetitions (require higher confidence)
            times_seen = ayah_occurrences.get(ayah_num, 0)
            if times_seen > 0:
                final_score -= repetition_confidence_boost  # Higher bar for repeats
            
            # Only include if above threshold
            effective_threshold = self.confidence_threshold
            if times_seen > 0:
                effective_threshold += repetition_confidence_boost
            
            if final_score >= self.confidence_threshold:
                candidates.append({
                    "ayah": ayah_num,
                    "confidence": final_score,
                    "base_score": base_score,
                    "is_partial": is_partial,
                    "partial_type": partial_type,
                    "partial_coverage": partial_coverage,
                    "is_repeat": times_seen > 0
                })
        
        # Sort by confidence (highest first)
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        
        return candidates
    
    def _detect_partial_match(
        self,
        segment_norm: List[str],
        ref_norm_words: List[str]
    ) -> Tuple[bool, str, float]:
        """
        Detect if a segment is a partial match (beginning, middle, or end of ayah).
        
        Returns:
            Tuple of (is_partial, partial_type, coverage)
            - is_partial: True if less than 80% of ayah covered
            - partial_type: "full", "start", "middle", "end"
            - coverage: fraction of reference words matched (0.0 - 1.0)
        """
        if not segment_norm or not ref_norm_words:
            return False, "full", 0.0
        
        # Use sequence matcher to find alignment
        matcher = difflib.SequenceMatcher(None, segment_norm, ref_norm_words)
        blocks = matcher.get_matching_blocks()
        
        # Calculate coverage
        matched_words = sum(block.size for block in blocks)
        coverage = matched_words / len(ref_norm_words) if ref_norm_words else 0.0
        
        # Determine if partial
        is_partial = coverage < 0.80
        
        if not is_partial:
            return False, "full", coverage
        
        # Determine partial type by looking at which part of ref is matched
        # Find the main matching region
        if not blocks or blocks[0].size == 0:
            return True, "middle", coverage
        
        # Check if matches start of reference
        first_block = blocks[0]
        matches_start = (first_block.b == 0 and first_block.size >= 2)
        
        # Check if matches end of reference
        last_significant = None
        for block in blocks:
            if block.size > 0:
                last_significant = block
        
        matches_end = (
            last_significant is not None and 
            last_significant.b + last_significant.size >= len(ref_norm_words) - 1
        )
        
        if matches_start and not matches_end:
            return True, "start", coverage
        elif matches_end and not matches_start:
            return True, "end", coverage
        else:
            return True, "middle", coverage
    
    # ========================================================================
    # Word-Level Classification (New Core Algorithm)
    # ========================================================================
    
    def classify_transcription_words(
        self,
        transcribed_words: List[TranscribedWord],
        surah: int,
        start_ayah: int = 1,
        end_ayah: Optional[int] = None,
        skip_preamble: bool = True,
        min_confidence: float = 0.4
    ) -> List[WordClassification]:
        """
        Classify each transcribed word to its position in the Quran.
        
        This is the fundamental matching operation. Each word is mapped to:
        - Which ayah it belongs to
        - Its position within that ayah  
        - The accurate Quran text (reference_text)
        - Which occurrence if the ayah is repeated
        
        Args:
            transcribed_words: List of words with timestamps from Whisper
            surah: Surah number to match against
            start_ayah: First ayah expected
            end_ayah: Last ayah expected (default: end of surah)
            skip_preamble: Skip isti'adha/basmallah at start
            min_confidence: Minimum confidence to include (low-confidence still included)
            
        Returns:
            List of WordClassification objects, one per transcribed word
        """
        if not transcribed_words:
            return []
        
        # Determine ayah range
        max_ayah = max(
            ayah for (s, ayah) in self.corpus.keys() if s == surah
        ) if any(s == surah for s, _ in self.corpus.keys()) else 286
        actual_end = end_ayah if end_ayah else max_ayah
        
        logger.info(f"Word classification: Surah {surah}, Ayahs {start_ayah}-{actual_end}")
        
        # Skip opening formulas
        word_start_idx = 0
        if skip_preamble:
            word_start_idx = self._skip_opening_formulas(transcribed_words)
            if word_start_idx > 0:
                logger.info(f"Skipped {word_start_idx} preamble words")
        
        # Build flattened reference word list for the ayah range
        # Format: [(ayah_num, word_idx, display_word, norm_word), ...]
        ref_words = []
        for ayah_num in range(start_ayah, actual_end + 1):
            ayah_data = self.corpus.get((surah, ayah_num))
            if ayah_data:
                display_words = ayah_data["display"].split()
                for word_idx, word in enumerate(display_words):
                    norm_word = self.normalizer.normalize(word)
                    ref_words.append((ayah_num, word_idx, word, norm_word))
        
        if not ref_words:
            logger.warning(f"No reference words found for Surah {surah}")
            return []
        
        logger.debug(f"Reference has {len(ref_words)} words across {actual_end - start_ayah + 1} ayahs")
        
        # Track state
        classifications: List[WordClassification] = []
        ref_position = 0  # Current position in ref_words
        ayah_occurrences: Dict[int, int] = {}  # ayah_num -> occurrence count
        last_ayah = start_ayah - 1  # Track for occurrence counting
        
        # Process each transcribed word
        for t_idx in range(word_start_idx, len(transcribed_words)):
            t_word = transcribed_words[t_idx]
            t_norm = self.normalizer.normalize(t_word.word)
            
            # Find best matching reference word
            best_match = self._find_best_word_match(
                t_norm=t_norm,
                ref_words=ref_words,
                current_pos=ref_position,
                ayah_occurrences=ayah_occurrences,
                start_ayah=start_ayah
            )
            
            if best_match:
                ref_idx, confidence, is_repetition = best_match
                ayah_num, word_idx, display_word, _ = ref_words[ref_idx]
                
                # Update occurrence tracking
                if ayah_num != last_ayah:
                    if is_repetition or ayah_num < last_ayah:
                        # Going backwards or detected repetition
                        ayah_occurrences[ayah_num] = ayah_occurrences.get(ayah_num, 0) + 1
                    elif ayah_num not in ayah_occurrences:
                        ayah_occurrences[ayah_num] = 1
                    last_ayah = ayah_num
                
                occurrence = ayah_occurrences.get(ayah_num, 1)
                
                classification = WordClassification(
                    word_index=t_idx,
                    surah=surah,
                    ayah=ayah_num,
                    ayah_word_index=word_idx,
                    occurrence=occurrence,
                    confidence=confidence,
                    transcribed_text=t_word.word.strip(),
                    reference_text=display_word,
                    start_time=t_word.start,
                    end_time=t_word.end
                )
                classifications.append(classification)
                
                # Advance reference position
                ref_position = ref_idx + 1
            else:
                # No confident match - still include with low confidence
                # Use current expected position
                if ref_position < len(ref_words):
                    ayah_num, word_idx, display_word, _ = ref_words[ref_position]
                else:
                    # Past end of reference, use last ayah
                    ayah_num, word_idx, display_word, _ = ref_words[-1]
                
                occurrence = ayah_occurrences.get(ayah_num, 1)
                
                classification = WordClassification(
                    word_index=t_idx,
                    surah=surah,
                    ayah=ayah_num,
                    ayah_word_index=word_idx,
                    occurrence=occurrence,
                    confidence=0.0,  # Low confidence
                    transcribed_text=t_word.word.strip(),
                    reference_text=display_word,
                    start_time=t_word.start,
                    end_time=t_word.end
                )
                classifications.append(classification)
        
        logger.info(f"Classified {len(classifications)} words")
        return classifications
    
    def _find_best_word_match(
        self,
        t_norm: str,
        ref_words: List[Tuple[int, int, str, str]],
        current_pos: int,
        ayah_occurrences: Dict[int, int],
        start_ayah: int,
        search_range: int = 8,
        repetition_threshold: float = 0.15
    ) -> Optional[Tuple[int, float, bool]]:
        """
        Find the best matching reference word for a transcribed word.
        
        Args:
            t_norm: Normalized transcribed word
            ref_words: List of (ayah_num, word_idx, display, norm) tuples
            current_pos: Current position in ref_words
            ayah_occurrences: How many times each ayah has been seen
            start_ayah: First ayah number
            search_range: How many positions forward to search
            repetition_threshold: Extra confidence required for backwards jumps
            
        Returns:
            (ref_idx, confidence, is_repetition) or None if no match
        """
        candidates = []
        
        # Search forward from current position
        for offset in range(search_range + 1):
            ref_idx = current_pos + offset
            if ref_idx >= len(ref_words):
                break
            
            ayah_num, word_idx, display_word, ref_norm = ref_words[ref_idx]
            score = self._word_similarity(t_norm, ref_norm)
            
            # Bias toward staying in sequence
            if offset == 0:
                score += 0.15  # Strong preference for current position
            elif offset == 1:
                score += 0.05  # Slight preference for next position
            
            if score >= 0.5:  # Minimum threshold
                candidates.append((ref_idx, score, False))
        
        # Note: Repetition detection is handled at a higher level
        # by tracking when the ayah sequence goes backwards.
        # Single-word matching to previous ayah starts is too error-prone
        # because common words like "الذين" appear at the start of many ayahs.
        
        if not candidates:
            return None
        
        # Return best match
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]
    
    def _word_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two normalized words."""
        if not word1 or not word2:
            return 0.0
        if word1 == word2:
            return 1.0
        if RAPIDFUZZ_AVAILABLE:
            return fuzz.ratio(word1, word2) / 100.0
        else:
            return difflib.SequenceMatcher(None, word1, word2).ratio()


def reconstruct_ayahs(
    classifications: List[WordClassification],
    quran_data: Dict[str, Dict[str, Dict[str, str]]]
) -> List[Dict[str, Any]]:
    """
    Reconstruct ayah events from word classifications.
    
    Groups classified words by (surah, ayah, occurrence) and outputs
    ayah-level events with accurate Quran text and timing from word boundaries.
    
    Args:
        classifications: List of WordClassification from classify_transcription_words()
        quran_data: Quran text dictionary for getting full ayah display text
        
    Returns:
        List of ayah event dicts with:
        - surah, ayah, occurrence
        - start_time, end_time (from word boundaries)
        - reference_text (accurate full ayah text)
        - transcribed_text (what Whisper heard)
        - word_count, reference_word_count
        - confidence (average of word confidences)
        - is_partial, partial_type
    """
    if not classifications:
        return []
    
    # Group by (surah, ayah, occurrence)
    groups: Dict[Tuple[int, int, int], List[WordClassification]] = {}
    for c in classifications:
        key = (c.surah, c.ayah, c.occurrence)
        if key not in groups:
            groups[key] = []
        groups[key].append(c)
    
    ayah_events = []
    
    for (surah, ayah, occurrence), words in groups.items():
        # Sort by word_index to ensure correct order
        words = sorted(words, key=lambda w: w.word_index)
        
        # Get full reference text for this ayah
        surah_data = quran_data.get(str(surah), {})
        ayah_data = surah_data.get(str(ayah), {})
        full_reference = ayah_data.get("displayText", ayah_data.get("text", "")).strip()
        ref_word_count = len(full_reference.split())
        
        # Calculate metrics
        start_time = words[0].start_time
        end_time = words[-1].end_time
        avg_confidence = sum(w.confidence for w in words) / len(words)
        transcribed = " ".join(w.transcribed_text for w in words)
        
        # Determine if partial
        word_coverage = len(words) / ref_word_count if ref_word_count > 0 else 1.0
        is_partial = word_coverage < 0.80
        
        # Determine partial type
        if is_partial:
            first_word_idx = words[0].ayah_word_index
            last_word_idx = words[-1].ayah_word_index
            
            if first_word_idx == 0 and last_word_idx < ref_word_count - 1:
                partial_type = "start"
            elif first_word_idx > 0 and last_word_idx >= ref_word_count - 1:
                partial_type = "end"
            elif first_word_idx > 0 and last_word_idx < ref_word_count - 1:
                partial_type = "middle"
            else:
                partial_type = "partial"
        else:
            partial_type = "full"
        
        ayah_events.append({
            "surah": surah,
            "ayah": ayah,
            "occurrence": occurrence,
            "start_time": start_time,
            "end_time": end_time,
            "reference_text": full_reference,
            "transcribed_text": transcribed,
            "word_count": len(words),
            "reference_word_count": ref_word_count,
            "confidence": avg_confidence,
            "is_partial": is_partial,
            "partial_type": partial_type,
            "word_classifications": [w.to_dict() for w in words]
        })
    
    # Sort by time
    ayah_events.sort(key=lambda x: x["start_time"])
    
    logger.info(f"Reconstructed {len(ayah_events)} ayah events from {len(classifications)} words")
    return ayah_events


# ============================================================================
# Word-Level Alignment
# ============================================================================

class WordAligner:
    """
    Aligns transcribed words to canonical Quran words with timestamp mapping.
    """
    
    def __init__(self, normalizer: Optional[ArabicNormalizer] = None):
        """
        Initialize word aligner.
        
        Args:
            normalizer: Arabic text normalizer
        """
        self.normalizer = normalizer or ArabicNormalizer()
    
    def align_words(
        self,
        transcribed_words: List[TranscribedWord],
        reference_words: List[str],
        surah: int,
        ayah: int
    ) -> List[Tuple[TranscribedWord, int]]:
        """
        Align transcribed words to reference Quran words.
        
        Uses sequence alignment (similar to diff) to map transcribed words
        to their corresponding positions in the canonical Quran text.
        
        Args:
            transcribed_words: Words from Whisper transcription
            reference_words: Words from Quran reference (with diacritics)
            surah: Surah number
            ayah: Ayah number
            
        Returns:
            List of tuples: (transcribed_word, quran_word_position)
            Position is 1-indexed to match Tarteel format
        """
        # Normalize both sides
        transcribed_normalized = [
            self.normalizer.normalize_word(w.word) 
            for w in transcribed_words
        ]
        reference_normalized = [
            self.normalizer.normalize_word(w) 
            for w in reference_words
        ]
        
        # Use SequenceMatcher to find alignment
        matcher = difflib.SequenceMatcher(
            None,
            transcribed_normalized,
            reference_normalized
        )
        
        alignments = []
        opcodes = matcher.get_opcodes()
        
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                # Perfect match - align words directly
                for idx, trans_idx in enumerate(range(i1, i2)):
                    ref_idx = j1 + idx
                    alignments.append((transcribed_words[trans_idx], ref_idx + 1))
            
            elif tag == 'replace':
                # Words differ - try to align proportionally
                trans_count = i2 - i1
                ref_count = j2 - j1
                
                if trans_count == ref_count:
                    # Same number of words, align one-to-one
                    for idx in range(trans_count):
                        alignments.append((transcribed_words[i1 + idx], j1 + idx + 1))
                else:
                    # Different counts - distribute transcribed words across reference
                    for idx in range(trans_count):
                        # Map proportionally
                        ref_position = j1 + int((idx / trans_count) * ref_count)
                        # Clamp to valid range
                        ref_position = min(ref_position, len(reference_words) - 1)
                        alignments.append((transcribed_words[i1 + idx], ref_position + 1))
            
            elif tag == 'insert':
                # Reference has extra words - these were not transcribed
                # We'll handle this in post-processing by interpolating timestamps
                pass
            
            elif tag == 'delete':
                # Transcription has extra words (errors or repeated words)
                # These words appear AFTER all reference words have been matched
                # Skip them - they likely belong to the next ayah
                if j1 >= len(reference_words):
                    # We're past the end of the reference - skip these words
                    logger.debug(f"Skipping {i2 - i1} extra transcribed words (past ayah end)")
                    continue
                
                # If we're in the middle, map to the last matched position
                for trans_idx in range(i1, i2):
                    ref_position = min(j1, len(reference_words) - 1)
                    alignments.append((transcribed_words[trans_idx], ref_position + 1))
        
        logger.info(f"Aligned {len(alignments)} words for {surah}:{ayah}")
        return alignments
    
    def interpolate_missing_words(
        self,
        alignments: List[Tuple[TranscribedWord, int]],
        total_reference_words: int,
        ayah_start_time: float,
        ayah_end_time: float
    ) -> List[Tuple[int, float, float]]:
        """
        Interpolate timestamps for words that were not transcribed by Whisper.
        
        Args:
            alignments: Aligned words from align_words()
            total_reference_words: Total number of words in reference ayah
            ayah_start_time: Start time of ayah
            ayah_end_time: End time of ayah
            
        Returns:
            List of tuples: (word_position, start_time, end_time) for ALL words
        """
        # Create mapping of word positions to timestamps
        position_to_times = {}
        for transcribed_word, position in alignments:
            position_to_times[position] = (transcribed_word.start, transcribed_word.end)
        
        # Fill in missing positions with interpolation
        all_word_times = []
        
        for pos in range(1, total_reference_words + 1):
            if pos in position_to_times:
                # Word has a timestamp from Whisper
                start, end = position_to_times[pos]
                all_word_times.append((pos, start, end))
            else:
                # Interpolate based on surrounding words
                # Find previous and next words with timestamps
                prev_pos = pos - 1
                next_pos = pos + 1
                
                while prev_pos > 0 and prev_pos not in position_to_times:
                    prev_pos -= 1
                
                while next_pos <= total_reference_words and next_pos not in position_to_times:
                    next_pos += 1
                
                # Interpolate
                if prev_pos > 0 and prev_pos in position_to_times:
                    prev_end = position_to_times[prev_pos][1]
                else:
                    prev_end = ayah_start_time
                
                if next_pos <= total_reference_words and next_pos in position_to_times:
                    next_start = position_to_times[next_pos][0]
                else:
                    next_start = ayah_end_time
                
                # Simple linear interpolation
                gap = next_start - prev_end
                steps = next_pos - prev_pos
                word_duration = gap / steps if steps > 0 else 0.5
                
                word_start = prev_end + (pos - prev_pos - 1) * word_duration
                word_end = word_start + word_duration
                
                all_word_times.append((pos, word_start, word_end))
        
        return all_word_times


# ============================================================================
# Tarteel Format Conversion
# ============================================================================

def convert_to_tarteel_format(
    ayah_matches: List[AyahMatch],
    audio_url: str,
    output_path: str
) -> str:
    """
    Convert aligned ayahs to Tarteel-compatible JSON format.
    
    Args:
        ayah_matches: List of matched and aligned ayahs
        audio_url: URL or path to the audio file
        output_path: Path to save JSON file
        
    Returns:
        Path to saved JSON file
    """
    tarteel_data = {}
    
    for match in ayah_matches:
        key = f"{match.surah}:{match.ayah}"
        
        # Convert word alignments to segments format: [[word_position, start_ms, end_ms]]
        segments = []
        for transcribed_word, word_position in match.word_alignments:
            segments.append([
                word_position,
                int(transcribed_word.start * 1000),  # Convert to milliseconds
                int(transcribed_word.end * 1000)
            ])
        
        # Calculate duration
        if segments:
            duration = max(seg[2] for seg in segments)
        else:
            duration = 0
        
        tarteel_data[key] = {
            "surah_number": match.surah,
            "ayah_number": match.ayah,
            "audio_url": audio_url,
            "duration": duration,
            "segments": segments
        }
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tarteel_data, f, ensure_ascii=False, indent=4)
    
    logger.info(f"Tarteel-format JSON saved to: {output_path}")
    return output_path
