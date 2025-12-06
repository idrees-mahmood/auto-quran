"""
Quran Transcription & Ayah Matching API

A comprehensive API for transcribing Quran recitation audio and matching
it to the canonical Quran text with word-level timestamps.

Usage:
    from quran_transcription_api import QuranTranscriptionAPI
    
    api = QuranTranscriptionAPI()
    result = api.process(
        audio_path="recitation.mp3",
        surah_hint=56,
        start_ayah=1,
        end_ayah=40
    )
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

# Import core modules
from audio_processing_utils import (
    AudioPreprocessor,
    WhisperTranscriber,
    ArabicNormalizer,
    TranscribedWord,
    load_quran_text,
    compute_audio_hash,
    save_transcription_checkpoint,
    load_transcription_checkpoint
)

from alignment_utils import (
    AyahDetector,
    WordAligner,
    convert_to_tarteel_format
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class WhisperModel(str, Enum):
    """Available Whisper model sizes."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    TURBO = "turbo"


class DeviceType(str, Enum):
    """Available compute devices."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class MatchStatus(str, Enum):
    """Status of ayah matching."""
    MATCHED = "matched"
    SKIPPED = "skipped"
    RESYNCED = "resynced"
    LOW_CONFIDENCE = "low_confidence"


class MatchType(str, Enum):
    """Type of word alignment."""
    DIRECT = "direct"
    FUZZY = "fuzzy"
    INTERPOLATED = "interpolated"


class ErrorCode(str, Enum):
    """API error codes."""
    AUDIO_NOT_FOUND = "AUDIO_NOT_FOUND"
    AUDIO_INVALID = "AUDIO_INVALID"
    TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    DETECTION_FAILED = "DETECTION_FAILED"
    ALIGNMENT_FAILED = "ALIGNMENT_FAILED"
    INVALID_SURAH = "INVALID_SURAH"
    INVALID_AYAH = "INVALID_AYAH"
    QURAN_DATA_NOT_FOUND = "QURAN_DATA_NOT_FOUND"
    EXPORT_FAILED = "EXPORT_FAILED"


# =============================================================================
# Data Classes - API Request/Response Objects
# =============================================================================

@dataclass
class TranscriptionOptions:
    """Options for audio transcription."""
    model: str = "base"
    device: str = "auto"
    language: str = "ar"
    preprocess: bool = True
    use_checkpoint: bool = True
    checkpoint_path: str = "temp/transcription_checkpoint.json"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionOptions:
    """Options for ayah detection."""
    surah_hint: Optional[int] = None
    start_ayah: int = 1
    end_ayah: Optional[int] = None
    confidence_threshold: float = 0.7
    skip_preamble: bool = True
    sequential_mode: bool = True
    window_size: int = 15
    overlap: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingOptions:
    """Combined options for full pipeline."""
    transcription: TranscriptionOptions = field(default_factory=TranscriptionOptions)
    detection: DetectionOptions = field(default_factory=DetectionOptions)
    debug: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "transcription": self.transcription.to_dict(),
            "detection": self.detection.to_dict(),
            "debug": self.debug
        }


@dataclass
class TranscribedWordResult:
    """Result object for a transcribed word."""
    word: str
    start: float
    end: float
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_transcribed_word(cls, tw: TranscribedWord) -> "TranscribedWordResult":
        return cls(
            word=tw.word,
            start=tw.start,
            end=tw.end,
            confidence=tw.confidence
        )


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""
    text: str
    words: List[TranscribedWordResult]
    duration: float
    word_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
            "duration": self.duration,
            "word_count": self.word_count
        }


@dataclass
class TranscriptionMetadata:
    """Metadata for transcription."""
    model: str
    device: str
    audio_hash: str
    processing_time_seconds: float
    audio_duration_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetectedAyahResult:
    """Result object for a detected ayah."""
    surah: int
    ayah: int
    confidence: float
    start_time: float
    end_time: float
    transcribed_text: str
    reference_text: str
    word_indices: Tuple[int, int]
    reference_word_count: int
    status: str = "matched"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "surah": self.surah,
            "ayah": self.ayah,
            "confidence": self.confidence,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "transcribed_text": self.transcribed_text,
            "reference_text": self.reference_text,
            "word_indices": list(self.word_indices),
            "reference_word_count": self.reference_word_count,
            "status": self.status
        }


@dataclass
class DetectionStatistics:
    """Statistics from ayah detection."""
    total_ayahs_detected: int
    total_words_processed: int
    average_confidence: float
    skipped_preamble_words: int
    resync_events: int
    low_confidence_matches: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WordAlignmentResult:
    """Result object for word alignment."""
    word_position: int
    reference_word: str
    transcribed_word: str
    start: float
    end: float
    match_type: str
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class AlignedAyahResult:
    """Result object for an aligned ayah."""
    surah: int
    ayah: int
    reference_text: str
    word_alignments: List[WordAlignmentResult]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "surah": self.surah,
            "ayah": self.ayah,
            "reference_text": self.reference_text,
            "word_alignments": [w.to_dict() for w in self.word_alignments]
        }


@dataclass
class AlignmentStatistics:
    """Statistics from word alignment."""
    total_words_aligned: int
    direct_matches: int
    fuzzy_matches: int
    interpolated: int
    alignment_accuracy: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TarteelSegment:
    """A single segment in Tarteel format."""
    word_position: int
    start_ms: int
    end_ms: int
    
    def to_list(self) -> List[int]:
        return [self.word_position, self.start_ms, self.end_ms]


@dataclass
class TarteelAyah:
    """An ayah in Tarteel format."""
    surah_number: int
    ayah_number: int
    audio_url: str
    duration: int
    segments: List[TarteelSegment]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "surah_number": self.surah_number,
            "ayah_number": self.ayah_number,
            "audio_url": self.audio_url,
            "duration": self.duration,
            "segments": [s.to_list() for s in self.segments]
        }


@dataclass
class APIError:
    """API error object."""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "code": self.code,
            "message": self.message
        }
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class APIResponse:
    """Standard API response wrapper."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[APIError] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"success": self.success}
        if self.data:
            result.update(self.data)
        if self.error:
            result["error"] = self.error.to_dict()
        return result


# =============================================================================
# Main API Class
# =============================================================================

class QuranTranscriptionAPI:
    """
    Main API class for Quran transcription and ayah matching.
    
    Provides methods for:
    - Audio transcription with Whisper
    - Ayah detection and matching
    - Word-level alignment
    - Export to Tarteel format
    - Full pipeline processing
    """
    
    def __init__(
        self,
        quran_data_path: str = "data/quran/quran.json",
        checkpoint_dir: str = "temp",
        log_level: int = logging.INFO
    ):
        """
        Initialize the API.
        
        Args:
            quran_data_path: Path to Quran JSON data file
            checkpoint_dir: Directory for checkpoints
            log_level: Logging level
        """
        self.quran_data_path = quran_data_path
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Set logging level
        logging.getLogger('alignment_utils').setLevel(log_level)
        logging.getLogger('audio_processing_utils').setLevel(log_level)
        
        # Lazy-loaded components
        self._quran_data: Optional[Dict] = None
        self._normalizer: Optional[ArabicNormalizer] = None
        self._transcriber: Optional[WhisperTranscriber] = None
        self._preprocessor: Optional[AudioPreprocessor] = None
        
        logger.info("QuranTranscriptionAPI initialized")
    
    @property
    def quran_data(self) -> Dict:
        """Lazy-load Quran data."""
        if self._quran_data is None:
            if not os.path.exists(self.quran_data_path):
                raise FileNotFoundError(f"Quran data not found: {self.quran_data_path}")
            self._quran_data = load_quran_text(self.quran_data_path)
            logger.info(f"Loaded Quran data from {self.quran_data_path}")
        return self._quran_data
    
    @property
    def normalizer(self) -> ArabicNormalizer:
        """Lazy-load normalizer."""
        if self._normalizer is None:
            self._normalizer = ArabicNormalizer()
        return self._normalizer
    
    def _get_preprocessor(self) -> AudioPreprocessor:
        """Get or create audio preprocessor."""
        if self._preprocessor is None:
            self._preprocessor = AudioPreprocessor(
                output_dir=str(self.checkpoint_dir / "audio_processed")
            )
        return self._preprocessor
    
    def _get_transcriber(
        self,
        model: str = "base",
        device: str = "auto"
    ) -> WhisperTranscriber:
        """Get or create Whisper transcriber."""
        # Create new transcriber if model/device changed
        if (self._transcriber is None or 
            self._transcriber.model_name != model or
            (device != "auto" and self._transcriber.device != device)):
            self._transcriber = WhisperTranscriber(model_name=model, device=device)
        return self._transcriber
    
    # =========================================================================
    # Transcription Methods
    # =========================================================================
    
    def transcribe(
        self,
        audio_path: str,
        options: Optional[TranscriptionOptions] = None
    ) -> APIResponse:
        """
        Transcribe an audio file with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            options: Transcription options
            
        Returns:
            APIResponse with transcription results
        """
        options = options or TranscriptionOptions()
        start_time = time.time()
        
        try:
            # Validate audio file
            if not os.path.exists(audio_path):
                return APIResponse(
                    success=False,
                    error=APIError(
                        code=ErrorCode.AUDIO_NOT_FOUND.value,
                        message=f"Audio file not found: {audio_path}",
                        details={"audio_path": audio_path}
                    )
                )
            
            # Compute audio hash for caching
            audio_hash = compute_audio_hash(audio_path)
            
            # Check for cached transcription
            if options.use_checkpoint:
                cached = load_transcription_checkpoint(
                    checkpoint_path=options.checkpoint_path,
                    audio_hash=audio_hash,
                    model_name=options.model
                )
                if cached:
                    logger.info("Using cached transcription")
                    transcriber = self._get_transcriber(options.model, options.device)
                    words = transcriber.extract_word_timestamps(cached)
                    
                    word_results = [
                        TranscribedWordResult.from_transcribed_word(w) for w in words
                    ]
                    
                    return APIResponse(
                        success=True,
                        data={
                            "transcription": TranscriptionResult(
                                text=cached.get("text", ""),
                                words=word_results,
                                duration=words[-1].end if words else 0,
                                word_count=len(words)
                            ).to_dict(),
                            "metadata": {
                                "model": options.model,
                                "device": options.device,
                                "audio_hash": audio_hash,
                                "processing_time_seconds": time.time() - start_time,
                                "from_cache": True
                            }
                        }
                    )
            
            # Preprocess audio if requested
            processed_path = audio_path
            if options.preprocess:
                preprocessor = self._get_preprocessor()
                processed_path = preprocessor.preprocess(audio_path)
            
            # Transcribe
            transcriber = self._get_transcriber(options.model, options.device)
            result = transcriber.transcribe(
                audio_path=processed_path,
                language=options.language,
                word_timestamps=True
            )
            
            # Save checkpoint
            if options.use_checkpoint:
                save_transcription_checkpoint(
                    transcription_data=result,
                    checkpoint_path=options.checkpoint_path,
                    audio_hash=audio_hash,
                    model_name=options.model
                )
            
            # Extract words
            words = transcriber.extract_word_timestamps(result)
            word_results = [
                TranscribedWordResult.from_transcribed_word(w) for w in words
            ]
            
            processing_time = time.time() - start_time
            
            return APIResponse(
                success=True,
                data={
                    "transcription": TranscriptionResult(
                        text=result.get("text", ""),
                        words=word_results,
                        duration=words[-1].end if words else 0,
                        word_count=len(words)
                    ).to_dict(),
                    "metadata": TranscriptionMetadata(
                        model=options.model,
                        device=transcriber.device,
                        audio_hash=audio_hash,
                        processing_time_seconds=processing_time,
                        audio_duration_seconds=words[-1].end if words else 0
                    ).to_dict()
                }
            )
            
        except Exception as e:
            logger.exception("Transcription failed")
            return APIResponse(
                success=False,
                error=APIError(
                    code=ErrorCode.TRANSCRIPTION_FAILED.value,
                    message=f"Transcription failed: {str(e)}",
                    details={"exception": type(e).__name__}
                )
            )
    
    # =========================================================================
    # Detection Methods
    # =========================================================================
    
    def detect_ayahs(
        self,
        transcribed_words: List[Union[Dict, TranscribedWord]],
        options: Optional[DetectionOptions] = None
    ) -> APIResponse:
        """
        Detect ayahs from transcribed words.
        
        Args:
            transcribed_words: List of transcribed word objects or dicts
            options: Detection options
            
        Returns:
            APIResponse with detected ayahs
        """
        options = options or DetectionOptions()
        
        try:
            # Validate surah hint
            if options.surah_hint is not None:
                if not 1 <= options.surah_hint <= 114:
                    return APIResponse(
                        success=False,
                        error=APIError(
                            code=ErrorCode.INVALID_SURAH.value,
                            message=f"Invalid surah number: {options.surah_hint}. Must be 1-114.",
                            details={"surah": options.surah_hint}
                        )
                    )
            
            # Convert dicts to TranscribedWord objects if needed
            words = []
            for w in transcribed_words:
                if isinstance(w, dict):
                    words.append(TranscribedWord(
                        word=w["word"],
                        start=w["start"],
                        end=w["end"],
                        confidence=w.get("confidence")
                    ))
                else:
                    words.append(w)
            
            # Create detector
            detector = AyahDetector(
                quran_data=self.quran_data,
                normalizer=self.normalizer,
                confidence_threshold=options.confidence_threshold
            )
            
            # Detect ayahs
            detected = detector.detect_ayahs_from_transcription(
                transcribed_words=words,
                window_size=options.window_size,
                overlap=options.overlap,
                surah_hint=options.surah_hint,
                sequential_mode=options.sequential_mode,
                start_ayah=options.start_ayah,
                end_ayah=options.end_ayah,
                skip_preamble=options.skip_preamble
            )
            
            # Build result objects
            detected_ayahs = []
            total_confidence = 0
            low_confidence_count = 0
            
            for ayah_info in detected:
                confidence = ayah_info["confidence"]
                total_confidence += confidence
                
                if confidence < options.confidence_threshold:
                    low_confidence_count += 1
                    status = MatchStatus.LOW_CONFIDENCE.value
                else:
                    status = MatchStatus.MATCHED.value
                
                # Get reference text
                surah_str = str(ayah_info["surah"])
                ayah_str = str(ayah_info["ayah"])
                ref_text = ""
                if surah_str in self.quran_data:
                    if ayah_str in self.quran_data[surah_str]:
                        ref_text = self.quran_data[surah_str][ayah_str].get("displayText", "").replace("\r", "")
                
                detected_ayahs.append(DetectedAyahResult(
                    surah=ayah_info["surah"],
                    ayah=ayah_info["ayah"],
                    confidence=confidence,
                    start_time=ayah_info["start_time"],
                    end_time=ayah_info["end_time"],
                    transcribed_text=ayah_info["transcribed_text"],
                    reference_text=ref_text,
                    word_indices=tuple(ayah_info["word_indices"]),
                    reference_word_count=ayah_info.get("reference_word_count", 0),
                    status=status
                ))
            
            # Calculate statistics
            avg_confidence = total_confidence / len(detected) if detected else 0
            
            stats = DetectionStatistics(
                total_ayahs_detected=len(detected),
                total_words_processed=len(words),
                average_confidence=avg_confidence,
                skipped_preamble_words=0,  # Could be tracked in detector
                resync_events=0,  # Could be tracked in detector
                low_confidence_matches=low_confidence_count
            )
            
            return APIResponse(
                success=True,
                data={
                    "detected_ayahs": [a.to_dict() for a in detected_ayahs],
                    "statistics": stats.to_dict()
                }
            )
            
        except Exception as e:
            logger.exception("Ayah detection failed")
            return APIResponse(
                success=False,
                error=APIError(
                    code=ErrorCode.DETECTION_FAILED.value,
                    message=f"Ayah detection failed: {str(e)}",
                    details={"exception": type(e).__name__}
                )
            )
    
    # =========================================================================
    # Alignment Methods
    # =========================================================================
    
    def align_words(
        self,
        detected_ayahs: List[Union[Dict, DetectedAyahResult]],
        transcribed_words: List[Union[Dict, TranscribedWord]],
        interpolate_missing: bool = True
    ) -> APIResponse:
        """
        Perform word-level alignment for detected ayahs.
        
        Args:
            detected_ayahs: List of detected ayah objects or dicts
            transcribed_words: Original transcribed words
            interpolate_missing: Whether to interpolate missing word timestamps
            
        Returns:
            APIResponse with aligned ayahs
        """
        try:
            # Convert dicts to objects if needed
            words = []
            for w in transcribed_words:
                if isinstance(w, dict):
                    words.append(TranscribedWord(
                        word=w["word"],
                        start=w["start"],
                        end=w["end"],
                        confidence=w.get("confidence")
                    ))
                else:
                    words.append(w)
            
            aligner = WordAligner(normalizer=self.normalizer)
            aligned_ayahs = []
            
            total_aligned = 0
            direct_matches = 0
            fuzzy_matches = 0
            interpolated = 0
            
            for ayah_info in detected_ayahs:
                if isinstance(ayah_info, dict):
                    surah = ayah_info["surah"]
                    ayah = ayah_info["ayah"]
                    word_indices = ayah_info["word_indices"]
                else:
                    surah = ayah_info.surah
                    ayah = ayah_info.ayah
                    word_indices = ayah_info.word_indices
                
                start_idx, end_idx = word_indices
                ayah_words = words[start_idx:end_idx]
                
                # Get reference words
                surah_str = str(surah)
                ayah_str = str(ayah)
                ref_text = ""
                if surah_str in self.quran_data:
                    if ayah_str in self.quran_data[surah_str]:
                        ref_text = self.quran_data[surah_str][ayah_str].get("displayText", "").replace("\r", "")
                
                ref_words = ref_text.split() if ref_text else []
                
                if not ref_words:
                    continue
                
                # Align words
                alignments = aligner.align_words(
                    transcribed_words=ayah_words,
                    reference_words=ref_words,
                    surah=surah,
                    ayah=ayah
                )
                
                # Interpolate if needed
                all_word_times: List[Tuple[int, float, float]] = []
                if interpolate_missing and ayah_words:
                    interpolated_times = aligner.interpolate_missing_words(
                        alignments=alignments,
                        total_reference_words=len(ref_words),
                        ayah_start_time=ayah_words[0].start,
                        ayah_end_time=ayah_words[-1].end
                    )
                    # Convert to consistent format: (pos, start, end)
                    for item in interpolated_times:
                        all_word_times.append((int(item[2]), float(item[0]), float(item[1])))
                else:
                    all_word_times = [(int(pos), float(tw.start), float(tw.end)) for tw, pos in alignments]
                
                # Build word alignment results
                word_alignment_results = []
                aligned_positions = {pos for _, pos in alignments}
                
                for pos, start, end in all_word_times:
                    ref_word = ref_words[pos - 1] if pos <= len(ref_words) else ""
                    
                    # Find transcribed word for this position
                    trans_word = ""
                    confidence = 0.0
                    match_type = MatchType.INTERPOLATED.value
                    
                    for tw, aligned_pos in alignments:
                        if aligned_pos == pos:
                            trans_word = tw.word
                            confidence = tw.confidence or 0.0
                            
                            # Determine match type
                            norm_trans = self.normalizer.normalize(tw.word)
                            norm_ref = self.normalizer.normalize(ref_word)
                            if norm_trans == norm_ref:
                                match_type = MatchType.DIRECT.value
                                direct_matches += 1
                            else:
                                match_type = MatchType.FUZZY.value
                                fuzzy_matches += 1
                            break
                    
                    if match_type == MatchType.INTERPOLATED.value:
                        interpolated += 1
                    
                    word_alignment_results.append(WordAlignmentResult(
                        word_position=pos,
                        reference_word=ref_word,
                        transcribed_word=trans_word,
                        start=start,
                        end=end,
                        match_type=match_type,
                        confidence=confidence
                    ))
                    total_aligned += 1
                
                aligned_ayahs.append(AlignedAyahResult(
                    surah=surah,
                    ayah=ayah,
                    reference_text=ref_text,
                    word_alignments=word_alignment_results
                ))
            
            # Calculate statistics
            accuracy = direct_matches / total_aligned if total_aligned > 0 else 0
            
            stats = AlignmentStatistics(
                total_words_aligned=total_aligned,
                direct_matches=direct_matches,
                fuzzy_matches=fuzzy_matches,
                interpolated=interpolated,
                alignment_accuracy=accuracy
            )
            
            return APIResponse(
                success=True,
                data={
                    "aligned_ayahs": [a.to_dict() for a in aligned_ayahs],
                    "statistics": stats.to_dict()
                }
            )
            
        except Exception as e:
            logger.exception("Word alignment failed")
            return APIResponse(
                success=False,
                error=APIError(
                    code=ErrorCode.ALIGNMENT_FAILED.value,
                    message=f"Word alignment failed: {str(e)}",
                    details={"exception": type(e).__name__}
                )
            )
    
    # =========================================================================
    # Export Methods
    # =========================================================================
    
    def export_tarteel(
        self,
        aligned_ayahs: List[Union[Dict, AlignedAyahResult]],
        audio_url: str,
        output_path: Optional[str] = None,
        include_metadata: bool = True
    ) -> APIResponse:
        """
        Export aligned data to Tarteel-compatible JSON format.
        
        Args:
            aligned_ayahs: List of aligned ayah objects or dicts
            audio_url: URL or path to audio file
            output_path: Path to save JSON file (optional)
            include_metadata: Whether to include additional metadata
            
        Returns:
            APIResponse with Tarteel format data
        """
        try:
            tarteel_data = {}
            
            for ayah_info in aligned_ayahs:
                if isinstance(ayah_info, dict):
                    surah = ayah_info["surah"]
                    ayah = ayah_info["ayah"]
                    word_alignments = ayah_info["word_alignments"]
                else:
                    surah = ayah_info.surah
                    ayah = ayah_info.ayah
                    word_alignments = [w.to_dict() for w in ayah_info.word_alignments]
                
                key = f"{surah}:{ayah}"
                
                # Convert to segments format
                segments = []
                for wa in word_alignments:
                    if isinstance(wa, dict):
                        segments.append([
                            wa["word_position"],
                            int(wa["start"] * 1000),
                            int(wa["end"] * 1000)
                        ])
                    else:
                        segments.append([
                            wa.word_position,
                            int(wa.start * 1000),
                            int(wa.end * 1000)
                        ])
                
                # Calculate duration
                duration = max(seg[2] for seg in segments) if segments else 0
                
                tarteel_data[key] = {
                    "surah_number": surah,
                    "ayah_number": ayah,
                    "audio_url": audio_url,
                    "duration": duration,
                    "segments": segments
                }
            
            # Save to file if path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(tarteel_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Tarteel JSON saved to: {output_path}")
            
            return APIResponse(
                success=True,
                data={
                    "output_path": output_path,
                    "tarteel_format": tarteel_data,
                    "ayah_count": len(tarteel_data)
                }
            )
            
        except Exception as e:
            logger.exception("Export failed")
            return APIResponse(
                success=False,
                error=APIError(
                    code=ErrorCode.EXPORT_FAILED.value,
                    message=f"Export failed: {str(e)}",
                    details={"exception": type(e).__name__}
                )
            )
    
    # =========================================================================
    # Full Pipeline
    # =========================================================================
    
    def process(
        self,
        audio_path: str,
        surah_hint: Optional[int] = None,
        start_ayah: int = 1,
        end_ayah: Optional[int] = None,
        output_path: Optional[str] = None,
        audio_url: Optional[str] = None,
        options: Optional[ProcessingOptions] = None
    ) -> APIResponse:
        """
        Run the complete transcription and matching pipeline.
        
        Args:
            audio_path: Path to audio file
            surah_hint: Restrict search to specific surah
            start_ayah: Starting ayah number
            end_ayah: Ending ayah number
            output_path: Path to save Tarteel JSON
            audio_url: URL for audio in output (defaults to audio_path)
            options: Processing options
            
        Returns:
            APIResponse with complete pipeline results
        """
        options = options or ProcessingOptions()
        
        # Set debug mode
        if options.debug:
            logging.getLogger('alignment_utils').setLevel(logging.DEBUG)
        
        pipeline_start = time.time()
        
        try:
            # Step 1: Transcribe
            logger.info("=== Step 1: Transcription ===")
            transcribe_result = self.transcribe(audio_path, options.transcription)
            
            if not transcribe_result.success or transcribe_result.data is None:
                return transcribe_result
            
            transcription_data = transcribe_result.data["transcription"]
            transcribed_words = transcription_data["words"]
            
            # Step 2: Detect Ayahs
            logger.info("=== Step 2: Ayah Detection ===")
            detection_options = options.detection
            detection_options.surah_hint = surah_hint
            detection_options.start_ayah = start_ayah
            detection_options.end_ayah = end_ayah
            
            detect_result = self.detect_ayahs(transcribed_words, detection_options)
            
            if not detect_result.success or detect_result.data is None:
                return detect_result
            
            detected_ayahs = detect_result.data["detected_ayahs"]
            detection_stats = detect_result.data["statistics"]
            
            # Step 3: Align Words
            logger.info("=== Step 3: Word Alignment ===")
            align_result = self.align_words(detected_ayahs, transcribed_words)
            
            if not align_result.success or align_result.data is None:
                return align_result
            
            aligned_ayahs = align_result.data["aligned_ayahs"]
            alignment_stats = align_result.data["statistics"]
            
            # Step 4: Export
            logger.info("=== Step 4: Export ===")
            export_result = self.export_tarteel(
                aligned_ayahs=aligned_ayahs,
                audio_url=audio_url or audio_path,
                output_path=output_path
            )
            
            if not export_result.success or export_result.data is None:
                return export_result
            
            # Build final response
            pipeline_time = time.time() - pipeline_start
            
            return APIResponse(
                success=True,
                data={
                    "pipeline_results": {
                        "transcription": {
                            "word_count": transcription_data["word_count"],
                            "duration": transcription_data["duration"],
                            "processing_time": transcribe_result.data["metadata"]["processing_time_seconds"]
                        },
                        "detection": {
                            "ayahs_detected": detection_stats["total_ayahs_detected"],
                            "average_confidence": detection_stats["average_confidence"],
                            "low_confidence_matches": detection_stats["low_confidence_matches"]
                        },
                        "alignment": {
                            "words_aligned": alignment_stats["total_words_aligned"],
                            "direct_matches": alignment_stats["direct_matches"],
                            "interpolated": alignment_stats["interpolated"]
                        },
                        "export": {
                            "output_path": export_result.data["output_path"],
                            "format": "tarteel",
                            "ayah_count": export_result.data["ayah_count"]
                        },
                        "total_processing_time": pipeline_time
                    },
                    "detected_ayahs": detected_ayahs,
                    "aligned_ayahs": aligned_ayahs,
                    "tarteel_output": export_result.data["tarteel_format"]
                }
            )
            
        except Exception as e:
            logger.exception("Pipeline failed")
            return APIResponse(
                success=False,
                error=APIError(
                    code="PIPELINE_FAILED",
                    message=f"Pipeline failed: {str(e)}",
                    details={"exception": type(e).__name__}
                )
            )


# =============================================================================
# Flask/FastAPI REST Server (Optional)
# =============================================================================

def create_rest_api(api: Optional[QuranTranscriptionAPI] = None):
    """
    Create a Flask REST API server.
    
    Returns:
        Flask app instance
    """
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        raise ImportError("Flask is required for REST API. Install with: pip install flask")
    
    app = Flask(__name__)
    api_instance = api or QuranTranscriptionAPI()
    
    @app.route('/api/v1/transcribe', methods=['POST'])
    def transcribe():
        data = request.json
        options = TranscriptionOptions(**{k: v for k, v in data.items() if k != 'audio_path'})
        result = api_instance.transcribe(data['audio_path'], options)
        return jsonify(result.to_dict())
    
    @app.route('/api/v1/detect-ayahs', methods=['POST'])
    def detect_ayahs():
        data = request.json
        options = DetectionOptions(**{k: v for k, v in data.items() if k != 'transcribed_words'})
        result = api_instance.detect_ayahs(data['transcribed_words'], options)
        return jsonify(result.to_dict())
    
    @app.route('/api/v1/align-words', methods=['POST'])
    def align_words():
        data = request.json
        result = api_instance.align_words(
            data['detected_ayahs'],
            data['transcribed_words'],
            data.get('interpolate_missing', True)
        )
        return jsonify(result.to_dict())
    
    @app.route('/api/v1/export-tarteel', methods=['POST'])
    def export_tarteel():
        data = request.json
        result = api_instance.export_tarteel(
            data['aligned_ayahs'],
            data['audio_url'],
            data.get('output_path'),
            data.get('include_metadata', True)
        )
        return jsonify(result.to_dict())
    
    @app.route('/api/v1/process', methods=['POST'])
    def process():
        data = request.json
        
        # Build options
        options = ProcessingOptions()
        if 'options' in data:
            opts = data['options']
            if 'transcription' in opts:
                options.transcription = TranscriptionOptions(**opts['transcription'])
            if 'detection' in opts:
                options.detection = DetectionOptions(**opts['detection'])
            options.debug = opts.get('debug', False)
        
        result = api_instance.process(
            audio_path=data['audio_path'],
            surah_hint=data.get('surah_hint'),
            start_ayah=data.get('start_ayah', 1),
            end_ayah=data.get('end_ayah'),
            output_path=data.get('output_path'),
            audio_url=data.get('audio_url'),
            options=options
        )
        return jsonify(result.to_dict())
    
    @app.route('/api/v1/health', methods=['GET'])
    def health():
        return jsonify({"status": "healthy", "version": "1.0.0"})
    
    return app


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for the API."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Quran Transcription & Ayah Matching API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe and match a recitation
  python quran_transcription_api.py process -a recitation.mp3 -s 56 -o output.json
  
  # Run REST API server
  python quran_transcription_api.py serve --port 8000
  
  # Just transcribe (no matching)
  python quran_transcription_api.py transcribe -a recitation.mp3 -m base
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Run full pipeline')
    process_parser.add_argument('-a', '--audio', required=True, help='Audio file path')
    process_parser.add_argument('-s', '--surah', type=int, help='Surah number (1-114)')
    process_parser.add_argument('--start', type=int, default=1, help='Starting ayah')
    process_parser.add_argument('--end', type=int, help='Ending ayah')
    process_parser.add_argument('-o', '--output', help='Output JSON path')
    process_parser.add_argument('-m', '--model', default='base', help='Whisper model')
    process_parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    # Transcribe command
    transcribe_parser = subparsers.add_parser('transcribe', help='Transcribe audio only')
    transcribe_parser.add_argument('-a', '--audio', required=True, help='Audio file path')
    transcribe_parser.add_argument('-m', '--model', default='base', help='Whisper model')
    transcribe_parser.add_argument('-o', '--output', help='Output JSON path')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Run REST API server')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    serve_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    serve_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        api = QuranTranscriptionAPI()
        
        options = ProcessingOptions()
        options.transcription.model = args.model
        options.debug = args.debug
        
        result = api.process(
            audio_path=args.audio,
            surah_hint=args.surah,
            start_ayah=args.start,
            end_ayah=args.end,
            output_path=args.output,
            options=options
        )
        
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        
    elif args.command == 'transcribe':
        api = QuranTranscriptionAPI()
        
        options = TranscriptionOptions(model=args.model)
        result = api.transcribe(args.audio, options)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"Saved to {args.output}")
        else:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        
    elif args.command == 'serve':
        app = create_rest_api()
        print(f"Starting server on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=args.debug)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
