"""
Audio processing utilities for custom Quran recitation transcription and alignment.

This module handles:
1. Whisper transcription with word-level timestamps
2. Arabic text normalization for fuzzy matching
3. Ayah detection and alignment
4. Conversion to Tarteel-compatible JSON format
"""

import os
import json
import re
import ssl
import hashlib
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging

# Bypass SSL verification for Whisper model downloads
# This is safe for downloading open-source models from OpenAI's CDN
ssl._create_default_https_context = ssl._create_unverified_context

# Third-party imports
try:
    import whisper
    from whisper.utils import get_writer
except ImportError:
    whisper = None
    
try:
    from pydub import AudioSegment
    from pydub.effects import normalize
except ImportError:
    AudioSegment = None

try:
    from rapidfuzz import fuzz, process
except ImportError:
    fuzz = None
    process = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TranscribedWord:
    """Represents a word from Whisper transcription with timing."""
    word: str
    start: float  # seconds
    end: float    # seconds
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence
        }


@dataclass
class AyahMatch:
    """Represents a matched ayah with confidence score."""
    surah: int
    ayah: int
    confidence: float
    transcribed_text: str
    reference_text: str
    word_alignments: List[Tuple[TranscribedWord, int]]  # (transcribed_word, quran_word_position)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "surah": self.surah,
            "ayah": self.ayah,
            "confidence": self.confidence,
            "transcribed_text": self.transcribed_text,
            "reference_text": self.reference_text,
            "word_count": len(self.word_alignments)
        }


# ============================================================================
# Arabic Text Normalization
# ============================================================================

class ArabicNormalizer:
    """
    Normalizes Arabic text for fuzzy matching by removing diacritics
    and standardizing character variants.
    """
    
    # Arabic diacritics (tashkeel)
    DIACRITICS = re.compile(r'[\u064B-\u065F\u0670]')
    
    # Character normalization mappings
    ALIF_VARIANTS = {
        'أ': 'ا',  # Alif with hamza above
        'إ': 'ا',  # Alif with hamza below
        'آ': 'ا',  # Alif with madda
        'ٱ': 'ا',  # Alif wasla
    }
    
    YA_VARIANTS = {
        'ى': 'ي',  # Alif maksura to ya
    }
    
    TA_MARBUTA = {
        'ة': 'ه',  # Ta marbuta to ha (optional, context-dependent)
    }
    
    def __init__(self, normalize_ta_marbuta: bool = False):
        """
        Initialize normalizer.
        
        Args:
            normalize_ta_marbuta: If True, converts ة to ه. 
                                 Set False for stricter matching.
        """
        self.normalize_ta_marbuta = normalize_ta_marbuta
        
    def normalize(self, text: str) -> str:
        """
        Normalize Arabic text for comparison.
        
        Args:
            text: Input Arabic text
            
        Returns:
            Normalized text suitable for fuzzy matching
        """
        if not text:
            return ""
        
        # Remove diacritics
        normalized = self.DIACRITICS.sub('', text)
        
        # Normalize Alif variants
        for variant, standard in self.ALIF_VARIANTS.items():
            normalized = normalized.replace(variant, standard)
        
        # Normalize Ya variants
        for variant, standard in self.YA_VARIANTS.items():
            normalized = normalized.replace(variant, standard)
        
        # Normalize Ta Marbuta (optional)
        if self.normalize_ta_marbuta:
            for variant, standard in self.TA_MARBUTA.items():
                normalized = normalized.replace(variant, standard)
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove non-Arabic characters (keep spaces)
        normalized = re.sub(r'[^\u0600-\u06FF\s]', '', normalized)
        
        return normalized.strip()
    
    def normalize_word(self, word: str) -> str:
        """Normalize a single word (removes leading/trailing spaces)."""
        return self.normalize(word)


# ============================================================================
# Audio Preprocessing
# ============================================================================

class AudioPreprocessor:
    """Handles audio file preprocessing for optimal Whisper transcription."""
    
    OPTIMAL_SAMPLE_RATE = 16000  # Whisper's expected sample rate
    
    def __init__(self, output_dir: str = "data/audio_processed"):
        """
        Initialize audio preprocessor.
        
        Args:
            output_dir: Directory to save processed audio files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Validate audio file and return metadata.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio metadata
        """
        if AudioSegment is None:
            raise ImportError("pydub is required for audio preprocessing. Install with: pip install pydub")
        
        audio = AudioSegment.from_file(audio_path)
        
        metadata = {
            "duration_seconds": len(audio) / 1000.0,
            "channels": audio.channels,
            "sample_rate": audio.frame_rate,
            "sample_width": audio.sample_width,
            "frame_count": audio.frame_count(),
            "size_mb": Path(audio_path).stat().st_size / (1024 * 1024)
        }
        
        logger.info(f"Audio validation: {metadata}")
        return metadata
    
    def preprocess(self, audio_path: str, normalize_audio: bool = True) -> str:
        """
        Preprocess audio file for Whisper transcription.
        
        Steps:
        1. Convert to mono if stereo
        2. Resample to 16kHz if needed
        3. Normalize volume (optional)
        4. Export as WAV for consistency
        
        Args:
            audio_path: Path to input audio file
            normalize_audio: Whether to normalize audio volume
            
        Returns:
            Path to processed audio file
        """
        if AudioSegment is None:
            raise ImportError("pydub is required. Install with: pip install pydub")
        
        logger.info(f"Preprocessing audio: {audio_path}")
        
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        
        # Convert to mono if stereo
        if audio.channels > 1:
            logger.info("Converting to mono")
            audio = audio.set_channels(1)
        
        # Resample to 16kHz if needed
        if audio.frame_rate != self.OPTIMAL_SAMPLE_RATE:
            logger.info(f"Resampling from {audio.frame_rate}Hz to {self.OPTIMAL_SAMPLE_RATE}Hz")
            audio = audio.set_frame_rate(self.OPTIMAL_SAMPLE_RATE)
        
        # Normalize volume
        if normalize_audio:
            logger.info("Normalizing audio volume")
            audio = normalize(audio)
        
        # Export as WAV
        output_filename = Path(audio_path).stem + "_processed.wav"
        output_path = self.output_dir / output_filename
        
        audio.export(output_path, format="wav")
        logger.info(f"Processed audio saved to: {output_path}")
        
        return str(output_path)


# ============================================================================
# Whisper Transcription
# ============================================================================

class WhisperTranscriber:
    """Handles Whisper transcription with word-level timestamps."""
    
    def __init__(self, model_name: str = "turbo", device: str = "auto"):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cpu, cuda, mps, auto)
                   'auto' will detect best available device (MPS on M1/M2 Mac, CUDA on GPU, CPU fallback)
        """
        if whisper is None:
            raise ImportError(
                "OpenAI Whisper is required. Install with: pip install openai-whisper\n"
                "For better performance, consider: pip install faster-whisper"
            )
        
        self.model_name = model_name
        self.device = self._detect_device(device)
        self.model = None
        
        logger.info(f"Initializing Whisper model: {model_name} on device: {self.device}")
    
    def _detect_device(self, device: str) -> str:
        """
        Detect the best available device for Whisper.
        
        Args:
            device: Requested device ('auto', 'cpu', 'cuda', 'mps')
            
        Returns:
            Device string to use
        """
        if device != "auto":
            return device
        
        # Check for Apple Silicon (M1/M2/M3) with MPS
        try:
            import torch
            if torch.backends.mps.is_available():
                logger.info("✓ Apple Silicon detected - using MPS (Metal) acceleration")
                return "mps"
        except (ImportError, AttributeError):
            pass
        
        # Check for CUDA GPU
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("✓ CUDA GPU detected")
                return "cuda"
        except (ImportError, AttributeError):
            pass
        
        # Fallback to CPU
        logger.info("Using CPU (consider upgrading for faster processing)")
        return "cpu"
    
    def load_model(self):
        """Load the Whisper model (lazy loading with MPS fallback)."""
        if self.model is None:
            try:
                self.model = whisper.load_model(self.model_name, device=self.device)
                logger.info(f"Whisper model loaded: {self.model_name} on {self.device}")
            except (NotImplementedError, RuntimeError) as e:
                if self.device == "mps":
                    logger.warning(f"MPS not supported for this model, falling back to CPU")
                    self.device = "cpu"
                    self.model = whisper.load_model(self.model_name, device="cpu")
                    logger.info(f"Whisper model loaded: {self.model_name} on CPU")
                else:
                    raise
    
    def transcribe(
        self,
        audio_path: str,
        language: str = "ar",
        word_timestamps: bool = True,
        save_json: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio file with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            language: Language code (ar for Arabic)
            word_timestamps: Whether to extract word-level timestamps
            save_json: Optional path to save raw transcription JSON
            
        Returns:
            Transcription result with segments and word-level timing
        """
        self.load_model()
        
        logger.info(f"Transcribing audio: {audio_path}")
        logger.info(f"Language: {language}, Word timestamps: {word_timestamps}")
        
        # Transcribe with Whisper
        result = self.model.transcribe(
            audio_path,
            language=language,
            word_timestamps=word_timestamps,
            verbose=False
        )
        
        # Save raw transcription if requested
        if save_json:
            os.makedirs(os.path.dirname(save_json), exist_ok=True)
            with open(save_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Raw transcription saved to: {save_json}")
        
        return result
    
    def extract_word_timestamps(self, transcription: Dict[str, Any]) -> List[TranscribedWord]:
        """
        Extract word-level timestamps from Whisper transcription.
        
        Args:
            transcription: Raw Whisper transcription result
            
        Returns:
            List of TranscribedWord objects
        """
        words = []
        
        for segment in transcription.get("segments", []):
            segment_words = segment.get("words", [])
            
            for word_data in segment_words:
                word = TranscribedWord(
                    word=word_data.get("word", "").strip(),
                    start=word_data.get("start", 0.0),
                    end=word_data.get("end", 0.0),
                    confidence=word_data.get("probability", None)
                )
                if word.word:  # Skip empty words
                    words.append(word)
        
        logger.info(f"Extracted {len(words)} words with timestamps")
        return words


# ============================================================================
# Helper Functions
# ============================================================================

def load_quran_text(quran_json_path: str = "data/quran/quran.json") -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Load Quran text from JSON file.
    
    Returns:
        Dictionary with structure: {surah: {ayah: {"displayText": ..., "text": ...}}}
    """
    with open(quran_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_audio_hash(audio_path: str, chunk_size: int = 8192) -> str:
    """
    Compute SHA256 hash of audio file for cache validation.
    
    Args:
        audio_path: Path to audio file
        chunk_size: Size of chunks to read (default 8KB)
        
    Returns:
        Hex string of SHA256 hash
    """
    sha256 = hashlib.sha256()
    with open(audio_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


# Default transcription storage path
TRANSCRIPTIONS_DIR = os.path.join(os.path.dirname(__file__), "data", "transcriptions")


def get_transcription_path(audio_path: str, ensure_dir: bool = True) -> str:
    """
    Get the canonical transcription storage path for an audio file.
    
    Args:
        audio_path: Path to the audio file
        ensure_dir: Create directory if it doesn't exist
        
    Returns:
        Path where transcription should be stored
    """
    audio_hash = compute_audio_hash(audio_path)[:16]
    filename = os.path.basename(audio_path)
    # Sanitize filename
    safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in filename)[:50]
    
    if ensure_dir:
        os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)
    
    return os.path.join(TRANSCRIPTIONS_DIR, f"{audio_hash}_{safe_name}.json")


def save_transcription_checkpoint(
    transcription_data: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    audio_hash: Optional[str] = None,
    model_name: Optional[str] = None
):
    """
    Save transcription checkpoint with metadata for validation.
    
    Args:
        transcription_data: Transcription result to save
        checkpoint_path: Path to save checkpoint (auto-generated if audio_path provided)
        audio_path: Original audio file path (used to generate checkpoint_path)
        audio_hash: Hash of audio file for validation
        model_name: Model used for transcription
    """
    # Auto-generate path from audio file if not provided
    if checkpoint_path is None:
        if audio_path:
            checkpoint_path = get_transcription_path(audio_path)
        else:
            checkpoint_path = os.path.join(TRANSCRIPTIONS_DIR, "checkpoint.json")
    
    # Compute hash if not provided but audio_path is
    if audio_hash is None and audio_path and os.path.exists(audio_path):
        audio_hash = compute_audio_hash(audio_path)
    
    checkpoint = {
        "transcription": transcription_data,
        "metadata": {
            "audio_path": audio_path,
            "audio_hash": audio_hash,
            "model_name": model_name,
            "saved_at": str(os.path.getmtime(checkpoint_path)) if os.path.exists(checkpoint_path) else None
        }
    }
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    logger.info(f"Transcription saved: {checkpoint_path}")
    return checkpoint_path


def load_transcription_checkpoint(
    checkpoint_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    audio_hash: Optional[str] = None,
    model_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Load transcription checkpoint with validation.
    
    Args:
        checkpoint_path: Path to checkpoint file
        audio_path: Audio file path (used to auto-find checkpoint)
        audio_hash: Expected audio file hash
        model_name: Expected model name
        
    Returns:
        Transcription data if valid, None otherwise
    """
    # Auto-find path from audio file if not provided
    if checkpoint_path is None and audio_path:
        checkpoint_path = get_transcription_path(audio_path, ensure_dir=False)
    
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        return None
    
    # Compute hash from audio if not provided
    if audio_hash is None and audio_path and os.path.exists(audio_path):
        audio_hash = compute_audio_hash(audio_path)
    
    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        
        # Handle old format (no metadata)
        if "transcription" not in checkpoint:
            logger.warning("Old checkpoint format detected, ignoring")
            return None
        
        metadata = checkpoint.get("metadata", {})
        
        # Validate audio file matches
        if audio_hash and metadata.get("audio_hash") != audio_hash:
            logger.info("Checkpoint audio file mismatch, ignoring")
            return None
        
        # Validate model matches
        if model_name and metadata.get("model_name") != model_name:
            logger.info(f"Checkpoint model mismatch ({metadata.get('model_name')} vs {model_name}), ignoring")
            return None
        
        logger.info("Valid checkpoint found, loading...")
        return checkpoint["transcription"]
        
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Invalid checkpoint file: {e}")
        return None
