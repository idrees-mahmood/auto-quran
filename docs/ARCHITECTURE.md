# System Architecture

## Detection Pipeline Overview

```
Audio File → Whisper Transcription → Ayah Detection → Word Alignment → Video Generation
                    ↓                       ↓                ↓
            TranscribedWord[]         AyahMatch[]      Tarteel JSON
```

---

## Core Components

### 1. Audio Processing (`audio_processing_utils.py`)

| Class | Purpose |
|-------|---------|
| `WhisperTranscriber` | Transcribes Arabic audio to text with word timestamps |
| `AudioPreprocessor` | Normalizes audio (16kHz mono) for optimal Whisper |
| `ArabicNormalizer` | Removes diacritics, normalizes letter variants |
| `TranscribedWord` | Dataclass: `word`, `start`, `end`, `confidence` |

### 2. Ayah Detection (`alignment_utils.py`)

| Class/Function | Purpose |
|----------------|---------|
| `AyahDetector` | Main detection orchestrator |
| `_detect_sequential` | Linear progression through ayahs |
| `_detect_with_repetition` | Handles Qari repetitions with segment splitting |
| `segment_by_pauses` | Groups words by natural pauses |

### 3. Word-Level Classification (`alignment_utils.py`)

| Component | Purpose |
|-----------|---------|
| `WordClassification` | Dataclass mapping each word to Quran position |
| `classify_transcription_words()` | Word-by-word Quran mapping |
| `reconstruct_ayahs()` | Groups classified words into ayah events |

---

## Detection Algorithms

### Sequential Detection (Default)
Best for: Standard recitations without repetitions.

```
Expected Ayah 1 → Match → Consume words → Expected Ayah 2 → ...
```

- Tries window sizes around reference word count
- Strict boundary validation
- 97.5% accuracy on test fixtures

### Repetition-Aware Detection
Best for: Recitations where Qari repeats ayahs.

```
Segment → Match to any ayah → Track occurrence count → Handle backwards jumps
```

Features:
- **Segment splitting**: When segment > ayah length, splits and re-queues remainder
- **Occurrence tracking**: Counts how many times each ayah appears
- **Expected next bias**: Bonus for sequential progression

Enable via UI: Advanced Options → "Allow Repetitions"

### Word-Level Classification
Best for: Precise word-to-Quran mapping with accurate reference text.

```
For each transcribed word:
  Find best matching reference word (forward search)
  Record: (ayah, word_index, reference_text)
```

Enable via UI: Advanced Options → "Word-Level Classification"

---

## Key Data Structures

### `TranscribedWord` (audio_processing_utils.py)
```python
@dataclass
class TranscribedWord:
    word: str           # Arabic text from Whisper
    start: float        # Start time in seconds
    end: float          # End time in seconds
    confidence: float   # 0.0-1.0
```

### `WordClassification` (alignment_utils.py)
```python
@dataclass
class WordClassification:
    word_index: int         # Index in transcription
    surah: int
    ayah: int
    ayah_word_index: int    # Position within ayah
    occurrence: int         # Which occurrence of this ayah
    confidence: float
    transcribed_text: str   # What Whisper heard
    reference_text: str     # Accurate Quran text
    start_time: float
    end_time: float
```

### `RecitationEvent` (alignment_utils.py)
```python
@dataclass
class RecitationEvent:
    surah: int
    ayah: int
    occurrence: int         # 1=first, 2=repeat, etc.
    start_time: float
    end_time: float
    confidence: float
    transcribed_text: str
    word_indices: Tuple[int, int]
    is_partial: bool
    partial_type: str       # "full", "start", "middle", "end"
    reference_word_count: int
```

---

## UI Configuration

### Detection Settings (app.py)
Located in: Custom Audio → Detect Ayahs → Advanced Options

| Toggle | Effect | Use Case |
|--------|--------|----------|
| Word-Level Classification | Uses `classify_transcription_words()` | Precise word mapping |
| Allow Repetitions | Uses `_detect_with_repetition()` | Qari repeats ayahs |
| Debug Mode | Verbose logging in terminal | Troubleshooting |

---

## Storage Paths

| Path | Contents |
|------|----------|
| `data/quran/quran.json` | Quran text with diacritics |
| `data/transcriptions/` | Cached Whisper outputs |
| `data/fixtures/` | Regression test fixtures |
| `tests/` | Test scripts |
| `tests/debug/` | Debug/analysis scripts |
