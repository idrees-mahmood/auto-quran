"""
Regression Test Framework for Quran Transcription & Ayah Matching

Provides automated testing to ensure algorithm changes don't break previously
working transcription-to-ayah matching pipelines.

Usage:
    # Run all regression tests
    python regression_tests.py run
    
    # Capture a new baseline from a successful run
    python regression_tests.py capture --audio "path/to/audio.mp3" --surah 56 --start 1 --end 40
    
    # List all test fixtures
    python regression_tests.py list
"""

import os
import sys
import json
import hashlib
import shutil
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TRANSCRIPTIONS_DIR = DATA_DIR / "transcriptions"
FIXTURES_DIR = DATA_DIR / "fixtures"
QURAN_JSON_PATH = DATA_DIR / "quran" / "quran.json"

CONFIDENCE_THRESHOLD = 0.80  # 80% minimum confidence for a "pass"


# =============================================================================
# Data Classes
# =============================================================================

class TestStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class TestMetadata:
    """Metadata for a regression test fixture."""
    name: str
    description: str
    audio_path: str
    audio_hash: str
    surah: int
    start_ayah: int
    end_ayah: int
    expected_ayah_count: int
    created_at: str
    transcription_file: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestMetadata':
        return cls(**data)


@dataclass
class ExpectedAyah:
    """Expected ayah match result."""
    surah: int
    ayah: int
    word_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class TestResult:
    """Result of running a single regression test."""
    fixture_name: str
    status: TestStatus
    expected_ayahs: int
    matched_ayahs: int
    high_confidence_matches: int
    low_confidence_matches: List[Tuple[int, int, float]]  # (surah, ayah, confidence)
    missing_ayahs: List[Tuple[int, int]]  # (surah, ayah)
    extra_ayahs: List[Tuple[int, int]]  # (surah, ayah)
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    
    @property
    def pass_rate(self) -> float:
        if self.expected_ayahs == 0:
            return 0.0
        return self.high_confidence_matches / self.expected_ayahs
    
    def summary(self) -> str:
        lines = [
            f"{'─' * 60}",
            f"Fixture: {self.fixture_name}",
            f"Status: {self.status.value.upper()}",
            f"{'─' * 60}",
        ]
        
        if self.status == TestStatus.ERROR:
            lines.append(f"Error: {self.error_message}")
        else:
            lines.extend([
                f"Expected Ayahs: {self.expected_ayahs}",
                f"Matched Ayahs:  {self.matched_ayahs}",
                f"High Confidence (≥{CONFIDENCE_THRESHOLD*100:.0f}%): {self.high_confidence_matches}",
                f"Pass Rate: {self.pass_rate*100:.1f}%",
            ])
            
            if self.low_confidence_matches:
                lines.append(f"\nLow Confidence Matches ({len(self.low_confidence_matches)}):")
                for surah, ayah, conf in self.low_confidence_matches[:5]:
                    lines.append(f"  - {surah}:{ayah} ({conf*100:.1f}%)")
                if len(self.low_confidence_matches) > 5:
                    lines.append(f"  ... and {len(self.low_confidence_matches) - 5} more")
            
            if self.missing_ayahs:
                lines.append(f"\nMissing Ayahs ({len(self.missing_ayahs)}):")
                for surah, ayah in self.missing_ayahs[:5]:
                    lines.append(f"  - {surah}:{ayah}")
                if len(self.missing_ayahs) > 5:
                    lines.append(f"  ... and {len(self.missing_ayahs) - 5} more")
                    
            if self.extra_ayahs:
                lines.append(f"\nUnexpected Ayahs ({len(self.extra_ayahs)}):")
                for surah, ayah in self.extra_ayahs[:5]:
                    lines.append(f"  - {surah}:{ayah}")
        
        lines.append(f"\nDuration: {self.duration_seconds:.2f}s")
        return "\n".join(lines)


# =============================================================================
# Core Functions
# =============================================================================

def extract_words_from_transcription(transcription_data: Dict[str, Any]) -> List['TranscribedWord']:
    """
    Extract TranscribedWord objects from Whisper transcription output.
    Handles both flat 'words' array and nested segments with words.
    """
    from audio_processing_utils import TranscribedWord
    
    words = []
    
    # First try top-level 'words' key
    if "words" in transcription_data:
        for w in transcription_data["words"]:
            if isinstance(w, dict):
                words.append(TranscribedWord(
                    word=w.get("word", "").strip(),
                    start=w.get("start", 0.0),
                    end=w.get("end", 0.0),
                    confidence=w.get("probability", w.get("confidence", 0.0))
                ))
    
    # Also check segments (Whisper standard format)
    if not words and "segments" in transcription_data:
        for segment in transcription_data["segments"]:
            if "words" in segment:
                for w in segment["words"]:
                    if isinstance(w, dict):
                        words.append(TranscribedWord(
                            word=w.get("word", "").strip(),
                            start=w.get("start", 0.0),
                            end=w.get("end", 0.0),
                            confidence=w.get("probability", w.get("confidence", 0.0))
                        ))
    
    return words


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def ensure_directories():
    """Create required directories if they don't exist."""
    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


def get_transcription_path(audio_path: str) -> Path:
    """Get the path where a transcription should be stored."""
    audio_hash = compute_file_hash(audio_path)
    filename = Path(audio_path).stem
    # Sanitize filename
    safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in filename)[:50]
    return TRANSCRIPTIONS_DIR / f"{audio_hash}_{safe_name}.json"


def save_transcription(audio_path: str, transcription_data: Dict[str, Any]) -> Path:
    """Save transcription to the transcriptions directory."""
    ensure_directories()
    trans_path = get_transcription_path(audio_path)
    
    # Add metadata
    transcription_data["_meta"] = {
        "audio_path": str(audio_path),
        "audio_hash": compute_file_hash(audio_path),
        "saved_at": datetime.now().isoformat()
    }
    
    with open(trans_path, "w", encoding="utf-8") as f:
        json.dump(transcription_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved transcription to: {trans_path}")
    return trans_path


def load_transcription(trans_path: Path) -> Optional[Dict[str, Any]]:
    """Load a transcription file."""
    if not trans_path.exists():
        return None
    with open(trans_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_fixture_path(surah: int, start_ayah: int, end_ayah: int, identifier: str) -> Path:
    """Get path for a test fixture directory."""
    safe_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in identifier)[:30]
    return FIXTURES_DIR / f"{surah}_{start_ayah}-{end_ayah}_{safe_id}"


# =============================================================================
# Fixture Management
# =============================================================================

def capture_fixture(
    audio_path: str,
    surah: int,
    start_ayah: int,
    end_ayah: int,
    identifier: str,
    description: str = "",
    transcription_data: Optional[Dict[str, Any]] = None,
    detected_ayahs: Optional[List[Dict[str, Any]]] = None,
    model_name: str = "turbo",
    expected_json_path: Optional[str] = None
) -> Path:
    """
    Capture a successful run as a regression test fixture.
    
    Args:
        audio_path: Path to the audio file
        surah: Surah number
        start_ayah: Starting ayah number
        end_ayah: Ending ayah number  
        identifier: Short identifier for this test (e.g., "ali_salah")
        description: Human-readable description
        transcription_data: Whisper transcription output (will run if not provided)
        detected_ayahs: Detected ayah matches (will run if not provided)
        model_name: Whisper model to use for transcription
        expected_json_path: Path to existing expected.json with ayah list (skips detection)
    
    Returns:
        Path to the created fixture directory
    """
    ensure_directories()
    
    # Validate audio exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Import processing modules
    from audio_processing_utils import WhisperTranscriber, AudioPreprocessor
    from alignment_utils import AyahDetector
    
    # Get or create transcription
    trans_path = get_transcription_path(audio_path)
    if transcription_data is None:
        existing = load_transcription(trans_path)
        if existing and existing.get("_meta", {}).get("audio_hash") == compute_file_hash(audio_path):
            logger.info(f"Using existing transcription: {trans_path}")
            transcription_data = existing
        else:
            logger.info(f"Running transcription with model '{model_name}'...")
            preprocessor = AudioPreprocessor()
            transcriber = WhisperTranscriber(model_name=model_name)
            
            processed_path = preprocessor.preprocess(audio_path)
            transcription_data = transcriber.transcribe(processed_path)
            trans_path = save_transcription(audio_path, transcription_data)
    else:
        trans_path = save_transcription(audio_path, transcription_data)
    
    # Load expected ayahs from JSON if provided (skip detection)
    if expected_json_path and os.path.exists(expected_json_path):
        logger.info(f"Loading expected ayahs from: {expected_json_path}")
        with open(expected_json_path, "r", encoding="utf-8") as f:
            expected_data = json.load(f)
        
        # Create fixture directory
        fixture_path = get_fixture_path(surah, start_ayah, end_ayah, identifier)
        fixture_path.mkdir(parents=True, exist_ok=True)
        
        # Copy the expected data (could be list of dicts or ayah numbers)
        expected_ayahs = []
        for item in expected_data:
            if isinstance(item, dict):
                expected_ayahs.append(ExpectedAyah(
                    surah=item.get("surah", surah),
                    ayah=item.get("ayah", item.get("ayah_number", 0)),
                    word_count=item.get("word_count", 0)
                ))
            elif isinstance(item, int):
                expected_ayahs.append(ExpectedAyah(surah=surah, ayah=item, word_count=0))
        
    else:
        # Run detection
        if detected_ayahs is None:
            logger.info("Running ayah detection...")
            from audio_processing_utils import load_quran_text
            
            words = extract_words_from_transcription(transcription_data)
            
            quran_data = load_quran_text(str(QURAN_JSON_PATH))
            detector = AyahDetector(quran_data=quran_data)
            detected_ayahs = detector.detect_ayahs_from_transcription(
                transcribed_words=words,
                surah_hint=surah,
                start_ayah=start_ayah,
                end_ayah=end_ayah
            )
        
        # Create fixture directory
        fixture_path = get_fixture_path(surah, start_ayah, end_ayah, identifier)
        fixture_path.mkdir(parents=True, exist_ok=True)
        
        # Build expected ayahs from detection
        expected_ayahs = []
        for match in detected_ayahs:
            if isinstance(match, dict):
                expected_ayahs.append(ExpectedAyah(
                    surah=match["surah"],
                    ayah=match["ayah"],
                    word_count=match.get("word_count", 0)
                ))
            else:
                expected_ayahs.append(ExpectedAyah(
                    surah=match.surah,
                    ayah=match.ayah,
                    word_count=getattr(match, "word_count", 0)
                ))
    
    # Create metadata
    metadata = TestMetadata(
        name=f"Surah {surah} Ayahs {start_ayah}-{end_ayah}",
        description=description or f"Regression test for Surah {surah}:{start_ayah}-{end_ayah}",
        audio_path=str(audio_path),
        audio_hash=compute_file_hash(audio_path),
        surah=surah,
        start_ayah=start_ayah,
        end_ayah=end_ayah,
        expected_ayah_count=len(expected_ayahs),
        created_at=datetime.now().isoformat(),
        transcription_file=str(trans_path.relative_to(DATA_DIR))
    )
    
    # Save metadata
    with open(fixture_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)
    
    # Save expected ayahs
    with open(fixture_path / "expected.json", "w", encoding="utf-8") as f:
        json.dump([a.to_dict() for a in expected_ayahs], f, ensure_ascii=False, indent=2)
    
    logger.info(f"Created fixture: {fixture_path}")
    logger.info(f"  Expected ayahs: {len(expected_ayahs)}")
    
    return fixture_path


def list_fixtures() -> List[Tuple[Path, TestMetadata]]:
    """List all available test fixtures."""
    fixtures = []
    
    if not FIXTURES_DIR.exists():
        return fixtures
    
    for fixture_dir in sorted(FIXTURES_DIR.iterdir()):
        if not fixture_dir.is_dir():
            continue
        
        metadata_path = fixture_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = TestMetadata.from_dict(json.load(f))
            fixtures.append((fixture_dir, metadata))
        except Exception as e:
            logger.warning(f"Failed to load fixture {fixture_dir}: {e}")
    
    return fixtures


# =============================================================================
# Test Runner
# =============================================================================

def run_fixture(fixture_path: Path) -> TestResult:
    """Run a single regression test fixture."""
    import time
    start_time = time.time()
    
    fixture_name = fixture_path.name
    
    try:
        # Load metadata
        with open(fixture_path / "metadata.json", "r", encoding="utf-8") as f:
            metadata = TestMetadata.from_dict(json.load(f))
        
        # Load expected ayahs
        with open(fixture_path / "expected.json", "r", encoding="utf-8") as f:
            expected_data = json.load(f)
        expected_set = {(e["surah"], e["ayah"]) for e in expected_data}
        
        # Load transcription
        trans_path = DATA_DIR / metadata.transcription_file
        transcription_data = load_transcription(trans_path)
        
        if not transcription_data:
            return TestResult(
                fixture_name=fixture_name,
                status=TestStatus.ERROR,
                expected_ayahs=len(expected_data),
                matched_ayahs=0,
                high_confidence_matches=0,
                low_confidence_matches=[],
                missing_ayahs=[],
                extra_ayahs=[],
                error_message=f"Transcription not found: {trans_path}",
                duration_seconds=time.time() - start_time
            )
        
        # Run detection
        from audio_processing_utils import load_quran_text
        from alignment_utils import AyahDetector
        
        words = extract_words_from_transcription(transcription_data)
        
        quran_data = load_quran_text(str(QURAN_JSON_PATH))
        detector = AyahDetector(quran_data=quran_data)
        detected = detector.detect_ayahs_from_transcription(
            transcribed_words=words,
            surah_hint=metadata.surah,
            start_ayah=metadata.start_ayah,
            end_ayah=metadata.end_ayah
        )
        
        # Analyze results
        detected_set = set()
        high_confidence = 0
        low_confidence = []
        
        for match in detected:
            if isinstance(match, dict):
                surah, ayah = match["surah"], match["ayah"]
                confidence = match.get("confidence", 1.0)
            else:
                surah, ayah = match.surah, match.ayah
                confidence = getattr(match, "confidence", 1.0)
            
            detected_set.add((surah, ayah))
            
            if confidence >= CONFIDENCE_THRESHOLD:
                high_confidence += 1
            else:
                low_confidence.append((surah, ayah, confidence))
        
        missing = sorted(expected_set - detected_set)
        extra = sorted(detected_set - expected_set)
        
        # Determine status
        all_expected_found = len(missing) == 0
        all_high_confidence = len(low_confidence) == 0
        
        if all_expected_found and all_high_confidence:
            status = TestStatus.PASSED
        else:
            status = TestStatus.FAILED
        
        return TestResult(
            fixture_name=fixture_name,
            status=status,
            expected_ayahs=len(expected_data),
            matched_ayahs=len(detected_set & expected_set),
            high_confidence_matches=high_confidence,
            low_confidence_matches=low_confidence,
            missing_ayahs=missing,
            extra_ayahs=extra,
            duration_seconds=time.time() - start_time
        )
        
    except Exception as e:
        logger.exception(f"Error running fixture {fixture_name}")
        return TestResult(
            fixture_name=fixture_name,
            status=TestStatus.ERROR,
            expected_ayahs=0,
            matched_ayahs=0,
            high_confidence_matches=0,
            low_confidence_matches=[],
            missing_ayahs=[],
            extra_ayahs=[],
            error_message=str(e),
            duration_seconds=time.time() - start_time
        )


def run_all_tests() -> List[TestResult]:
    """Run all regression tests."""
    fixtures = list_fixtures()
    
    if not fixtures:
        logger.warning("No test fixtures found. Use 'capture' command to create one.")
        return []
    
    results = []
    
    print(f"\n{'=' * 60}")
    print(f"Running {len(fixtures)} regression test(s)")
    print(f"{'=' * 60}\n")
    
    for fixture_path, metadata in fixtures:
        logger.info(f"Running: {metadata.name}")
        result = run_fixture(fixture_path)
        results.append(result)
        print(result.summary())
        print()
    
    # Summary
    passed = sum(1 for r in results if r.status == TestStatus.PASSED)
    failed = sum(1 for r in results if r.status == TestStatus.FAILED)
    errors = sum(1 for r in results if r.status == TestStatus.ERROR)
    
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {passed} passed, {failed} failed, {errors} errors")
    print(f"{'=' * 60}")
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Regression Test Framework for Quran Transcription"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run all regression tests")
    run_parser.add_argument("--fixture", "-f", help="Run specific fixture by name")
    
    # Capture command
    capture_parser = subparsers.add_parser("capture", help="Capture a new test fixture")
    capture_parser.add_argument("--audio", "-a", required=True, help="Path to audio file")
    capture_parser.add_argument("--surah", "-s", type=int, required=True, help="Surah number")
    capture_parser.add_argument("--start", type=int, required=True, help="Start ayah")
    capture_parser.add_argument("--end", type=int, required=True, help="End ayah")
    capture_parser.add_argument("--id", "-i", default="test", help="Test identifier")
    capture_parser.add_argument("--desc", "-d", default="", help="Description")
    capture_parser.add_argument("--model", "-m", default="turbo", help="Whisper model (tiny/base/small/medium/large/turbo)")
    capture_parser.add_argument("--expected-json", "-e", help="Path to existing detection results JSON (skips re-detection)")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all test fixtures")
    
    args = parser.parse_args()
    
    if args.command == "run":
        if args.fixture:
            fixture_path = FIXTURES_DIR / args.fixture
            if not fixture_path.exists():
                print(f"Fixture not found: {args.fixture}")
                sys.exit(1)
            result = run_fixture(fixture_path)
            print(result.summary())
            sys.exit(0 if result.status == TestStatus.PASSED else 1)
        else:
            results = run_all_tests()
            failed = any(r.status != TestStatus.PASSED for r in results)
            sys.exit(1 if failed else 0)
    
    elif args.command == "capture":
        try:
            fixture_path = capture_fixture(
                audio_path=args.audio,
                surah=args.surah,
                start_ayah=args.start,
                end_ayah=args.end,
                identifier=args.id,
                description=args.desc,
                model_name=args.model,
                expected_json_path=args.expected_json
            )
            print(f"\n✓ Created fixture: {fixture_path}")
        except Exception as e:
            print(f"\n✗ Failed to create fixture: {e}")
            sys.exit(1)
    
    elif args.command == "list":
        fixtures = list_fixtures()
        if not fixtures:
            print("No test fixtures found.")
        else:
            print(f"\nFound {len(fixtures)} test fixture(s):\n")
            for fixture_path, metadata in fixtures:
                print(f"  {fixture_path.name}")
                print(f"    {metadata.name}")
                print(f"    Surah {metadata.surah}:{metadata.start_ayah}-{metadata.end_ayah}")
                print(f"    Expected: {metadata.expected_ayah_count} ayahs")
                print()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
