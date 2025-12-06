"""
Ayah matching and word alignment algorithms for Quran transcription.

This module implements fuzzy matching between Whisper transcription
and Quran reference text, with word-level alignment.
"""

import json
import difflib
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging

from audio_processing_utils import (
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
        skip_preamble: bool = True
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
            
        Returns:
            List of detected ayah segments with timing info
        """
        if surah_hint:
            end_str = f" to {end_ayah}" if end_ayah else ""
            logger.info(f"Detecting ayahs from {len(transcribed_words)} words (Surah {surah_hint}, Ayah {start_ayah}{end_str})")
        else:
            logger.info(f"Detecting ayahs from {len(transcribed_words)} transcribed words")
        
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
        # Allow slight variation: [ref_count - 2, ref_count + 2]
        min_window = max(1, ref_count - 2)
        max_window = min(remaining, ref_count + 3)
        
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
        """
        if not transcribed_words:
            return 0
        
        # Common opening formula patterns (normalized)
        istiadha_words = {'اعوذ', 'بالله', 'من', 'الشيطان', 'الرجيم'}
        basmallah_words = {'بسم', 'الله', 'الرحمن', 'الرحيم'}
        
        skip_count = 0
        found_formula = True
        
        while found_formula and skip_count < len(transcribed_words):
            found_formula = False
            
            # Check next few words
            for j in range(skip_count, min(skip_count + 10, len(transcribed_words))):
                word = self.normalizer.normalize(transcribed_words[j].word)
                
                if word in istiadha_words or word in basmallah_words:
                    skip_count = j + 1
                    found_formula = True
                elif skip_count > 0 and j == skip_count:
                    # We've moved past the formulas
                    break
            
            # If we found formula words, continue checking
            if not found_formula and skip_count > 0:
                break
        
        return skip_count
    
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
                        alignments.append((transcribed_words[i1 + idx], ref_position + 1))
            
            elif tag == 'insert':
                # Reference has extra words - these were not transcribed
                # We'll handle this in post-processing by interpolating timestamps
                pass
            
            elif tag == 'delete':
                # Transcription has extra words (errors or repeated words)
                # Try to map to nearest reference word
                for trans_idx in range(i1, i2):
                    # Map to the reference position before the deletion
                    ref_position = j1 if j1 < len(reference_words) else len(reference_words)
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
