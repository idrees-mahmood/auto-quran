# Quran Transcription & Ayah Matching API

## Overview

This API provides a complete pipeline for transcribing Quran recitation audio and matching it to the canonical Quran text with word-level timestamps. It handles:

1. **Audio Processing** - Preprocessing audio for optimal transcription
2. **Whisper Transcription** - AI-powered Arabic speech-to-text with word timestamps
3. **Ayah Detection** - Matching transcribed text to Quran verses
4. **Word Alignment** - Precise word-by-word timestamp mapping
5. **Export** - Tarteel-compatible JSON format for video generation

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Audio File    │────▶│  Transcription   │────▶│  Ayah Detection │
│  (.mp3/.wav)    │     │   (Whisper AI)   │     │ (Fuzzy Matching)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Tarteel JSON   │◀────│  Word Alignment  │◀────│ Detected Ayahs  │
│    Export       │     │  (Interpolation) │     │  with Timing    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

---

## API Endpoints

### POST `/api/v1/transcribe`

Transcribe an audio file and return word-level timestamps.

#### Request

```json
{
  "audio_path": "/path/to/recitation.mp3",
  "model": "base",
  "device": "auto",
  "language": "ar",
  "preprocess": true,
  "use_checkpoint": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `audio_path` | string | **required** | Path to audio file (MP3, WAV, etc.) |
| `model` | string | `"base"` | Whisper model: `tiny`, `base`, `small`, `medium`, `large`, `turbo` |
| `device` | string | `"auto"` | Device: `auto`, `cpu`, `cuda`, `mps` |
| `language` | string | `"ar"` | Language code (Arabic) |
| `preprocess` | boolean | `true` | Whether to preprocess audio (normalize, resample) |
| `use_checkpoint` | boolean | `true` | Whether to use/save checkpoint for resumability |

#### Response

```json
{
  "success": true,
  "transcription": {
    "text": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ إِذَا وَقَعَتِ الْوَاقِعَةُ...",
    "words": [
      {
        "word": "بِسْمِ",
        "start": 0.0,
        "end": 0.48,
        "confidence": 0.95
      },
      {
        "word": "اللَّهِ",
        "start": 0.48,
        "end": 0.92,
        "confidence": 0.97
      }
    ],
    "duration": 125.5,
    "word_count": 245
  },
  "metadata": {
    "model": "base",
    "device": "mps",
    "audio_hash": "a1b2c3d4e5f6...",
    "processing_time_seconds": 42.3
  }
}
```

---

### POST `/api/v1/detect-ayahs`

Detect which ayahs are present in transcribed text.

#### Request

```json
{
  "transcribed_words": [...],
  "surah_hint": 56,
  "start_ayah": 1,
  "end_ayah": null,
  "confidence_threshold": 0.7,
  "skip_preamble": true,
  "sequential_mode": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `transcribed_words` | array | **required** | Array of word objects from transcription |
| `surah_hint` | integer | `null` | Restrict search to specific surah (1-114) |
| `start_ayah` | integer | `1` | Starting ayah number |
| `end_ayah` | integer | `null` | Ending ayah number (null = until end) |
| `confidence_threshold` | float | `0.7` | Minimum confidence score (0.0-1.0) |
| `skip_preamble` | boolean | `true` | Skip isti'adha and basmallah |
| `sequential_mode` | boolean | `true` | Use sequential detection (recommended when surah is known) |

#### Response

```json
{
  "success": true,
  "detected_ayahs": [
    {
      "surah": 56,
      "ayah": 1,
      "confidence": 0.92,
      "start_time": 5.2,
      "end_time": 7.8,
      "transcribed_text": "إذا وقعت الواقعة",
      "reference_text": "إِذَا وَقَعَتِ الْوَاقِعَةُ",
      "word_indices": [12, 15],
      "reference_word_count": 3,
      "status": "matched"
    },
    {
      "surah": 56,
      "ayah": 2,
      "confidence": 0.88,
      "start_time": 7.8,
      "end_time": 10.5,
      "transcribed_text": "ليس لوقعتها كاذبة",
      "reference_text": "لَيْسَ لِوَقْعَتِهَا كَاذِبَةٌ",
      "word_indices": [15, 19],
      "reference_word_count": 4,
      "status": "matched"
    }
  ],
  "statistics": {
    "total_ayahs_detected": 40,
    "total_words_processed": 245,
    "average_confidence": 0.85,
    "skipped_preamble_words": 9,
    "resync_events": 2
  }
}
```

---

### POST `/api/v1/align-words`

Perform word-level alignment for detected ayahs.

#### Request

```json
{
  "detected_ayahs": [...],
  "transcribed_words": [...],
  "interpolate_missing": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `detected_ayahs` | array | **required** | Array of detected ayah objects |
| `transcribed_words` | array | **required** | Original transcribed words array |
| `interpolate_missing` | boolean | `true` | Interpolate timestamps for words missed by Whisper |

#### Response

```json
{
  "success": true,
  "aligned_ayahs": [
    {
      "surah": 56,
      "ayah": 1,
      "reference_text": "إِذَا وَقَعَتِ الْوَاقِعَةُ",
      "word_alignments": [
        {
          "word_position": 1,
          "reference_word": "إِذَا",
          "transcribed_word": "إذا",
          "start": 5.2,
          "end": 5.8,
          "match_type": "direct",
          "confidence": 0.95
        },
        {
          "word_position": 2,
          "reference_word": "وَقَعَتِ",
          "transcribed_word": "وقعت",
          "start": 5.8,
          "end": 6.5,
          "match_type": "direct",
          "confidence": 0.92
        },
        {
          "word_position": 3,
          "reference_word": "الْوَاقِعَةُ",
          "transcribed_word": "الواقعة",
          "start": 6.5,
          "end": 7.8,
          "match_type": "direct",
          "confidence": 0.94
        }
      ]
    }
  ],
  "statistics": {
    "total_words_aligned": 120,
    "direct_matches": 108,
    "interpolated": 12,
    "alignment_accuracy": 0.90
  }
}
```

---

### POST `/api/v1/export-tarteel`

Export aligned data to Tarteel-compatible JSON format.

#### Request

```json
{
  "aligned_ayahs": [...],
  "audio_url": "https://example.com/audio/recitation.mp3",
  "output_path": "output/my_recitation.json",
  "include_metadata": true
}
```

#### Response

```json
{
  "success": true,
  "output_path": "output/my_recitation.json",
  "tarteel_format": {
    "56:1": {
      "surah_number": 56,
      "ayah_number": 1,
      "audio_url": "https://example.com/audio/recitation.mp3",
      "duration": 2600,
      "segments": [
        [1, 5200, 5800],
        [2, 5800, 6500],
        [3, 6500, 7800]
      ]
    }
  }
}
```

---

### POST `/api/v1/process` (Full Pipeline)

Run the complete transcription and matching pipeline in one call.

#### Request

```json
{
  "audio_path": "/path/to/recitation.mp3",
  "surah_hint": 56,
  "start_ayah": 1,
  "end_ayah": 40,
  "output_path": "output/surah_56.json",
  "audio_url": "https://cdn.example.com/recitations/surah56.mp3",
  "options": {
    "model": "base",
    "device": "auto",
    "confidence_threshold": 0.7,
    "skip_preamble": true,
    "preprocess_audio": true,
    "use_checkpoint": true,
    "debug": false
  }
}
```

#### Response

```json
{
  "success": true,
  "pipeline_results": {
    "transcription": {
      "word_count": 245,
      "duration": 125.5,
      "processing_time": 42.3
    },
    "detection": {
      "ayahs_detected": 40,
      "average_confidence": 0.85,
      "resync_events": 2
    },
    "alignment": {
      "words_aligned": 120,
      "direct_matches": 108,
      "interpolated": 12
    },
    "export": {
      "output_path": "output/surah_56.json",
      "format": "tarteel"
    }
  },
  "detected_ayahs": [...],
  "tarteel_output": {...}
}
```

---

## Error Responses

All endpoints return consistent error responses:

```json
{
  "success": false,
  "error": {
    "code": "TRANSCRIPTION_FAILED",
    "message": "Whisper transcription failed: Audio file not found",
    "details": {
      "audio_path": "/path/to/missing.mp3",
      "exception": "FileNotFoundError"
    }
  }
}
```

### Error Codes

| Code | Description |
|------|-------------|
| `AUDIO_NOT_FOUND` | Audio file does not exist |
| `AUDIO_INVALID` | Audio file is corrupted or unsupported format |
| `TRANSCRIPTION_FAILED` | Whisper transcription failed |
| `MODEL_LOAD_FAILED` | Failed to load Whisper model |
| `DETECTION_FAILED` | Ayah detection failed |
| `ALIGNMENT_FAILED` | Word alignment failed |
| `INVALID_SURAH` | Invalid surah number (must be 1-114) |
| `INVALID_AYAH` | Invalid ayah number for the specified surah |
| `QURAN_DATA_NOT_FOUND` | Quran reference data file not found |
| `EXPORT_FAILED` | Failed to export to specified format |

---

## Data Structures

### TranscribedWord

```typescript
interface TranscribedWord {
  word: string;           // The transcribed Arabic word
  start: number;          // Start time in seconds
  end: number;            // End time in seconds
  confidence: number | null;  // Whisper confidence (0.0-1.0)
}
```

### DetectedAyah

```typescript
interface DetectedAyah {
  surah: number;              // Surah number (1-114)
  ayah: number;               // Ayah number
  confidence: number;         // Match confidence (0.0-1.0)
  start_time: number;         // Start time in seconds
  end_time: number;           // End time in seconds
  transcribed_text: string;   // What Whisper transcribed
  reference_text: string;     // Canonical Quran text (with diacritics)
  word_indices: [number, number];  // [start_idx, end_idx] in transcribed_words
  reference_word_count: number;    // Number of words in reference ayah
  status: 'matched' | 'skipped' | 'resynced';
}
```

### WordAlignment

```typescript
interface WordAlignment {
  word_position: number;      // Position in reference (1-indexed)
  reference_word: string;     // Canonical word with diacritics
  transcribed_word: string;   // What Whisper transcribed
  start: number;              // Start time in seconds
  end: number;                // End time in seconds
  match_type: 'direct' | 'fuzzy' | 'interpolated';
  confidence: number;         // Word-level match confidence
}
```

### TarteelSegment

```typescript
// Tarteel format: [word_position, start_ms, end_ms]
type TarteelSegment = [number, number, number];

interface TarteelAyah {
  surah_number: number;
  ayah_number: number;
  audio_url: string;
  duration: number;           // Total duration in milliseconds
  segments: TarteelSegment[];
}
```

---

## Algorithm Details

### Sequential Ayah Detection

When `surah_hint` is provided, the algorithm uses **sequential detection**:

1. **Skip Preamble** - Detect and skip isti'adha (أعوذ بالله من الشيطان الرجيم) and basmallah (بسم الله الرحمن الرحيم)

2. **Window Matching** - For each expected ayah:
   - Get reference word count from Quran corpus
   - Try window sizes: `[ref_count - 2, ref_count + 3]`
   - Score each window using fuzzy word matching

3. **Scoring Formula**:
   ```
   combined_score = (alignment_score × 0.7) + (fuzzy_score × 0.3)
                   - boundary_penalty - size_penalty
   ```
   
   Where:
   - `alignment_score`: Fuzzy word-by-word matching (tolerates Whisper errors)
   - `fuzzy_score`: Overall string similarity
   - `boundary_penalty`: Penalty if trailing words match next ayah (prevents "word stealing")
   - `size_penalty`: 5% penalty per word difference from expected count

4. **Resync Logic** - When 3 consecutive ayahs fail to match:
   - Search forward up to 20 words
   - Try to match any of the next 10 ayahs
   - Jump to resync point if found

### Arabic Text Normalization

The normalizer handles Arabic text variations for matching:

| Variation | Normalized |
|-----------|------------|
| أ إ آ ٱ (Alif variants) | ا |
| ى (Alif maksura) | ي |
| All diacritics (tashkeel) | Removed |

This allows matching between:
- Whisper output: `الرحمن` (no diacritics)
- Quran reference: `الرَّحْمَٰنِ` (with diacritics)

### Fuzzy Word Matching

Individual words are matched using fuzzy comparison (RapidFuzz or difflib):
- Threshold: 60% similarity
- Handles Whisper errors like "سجر" vs "سدر"
- Prevents cascading failures from transcription mistakes

---

## Configuration

### Environment Variables

```bash
# Whisper model cache directory
WHISPER_CACHE_DIR=/path/to/models

# Default Whisper model
QURAN_WHISPER_MODEL=base

# Checkpoint directory
QURAN_CHECKPOINT_DIR=temp/checkpoints

# Quran data path
QURAN_DATA_PATH=data/quran/quran.json
```

### Model Selection Guide

| Model | Size | Speed | Accuracy | Recommended For |
|-------|------|-------|----------|-----------------|
| `tiny` | 39 MB | Very Fast | ~70% | Quick testing |
| `base` | 74 MB | Fast | ~75% | Development, short clips |
| `small` | 244 MB | Medium | ~80% | Production (balanced) |
| `medium` | 769 MB | Slow | ~85% | High accuracy needed |
| `large` | 1.5 GB | Very Slow | ~90% | Maximum accuracy |
| `turbo` | 809 MB | Medium | ~85% | Best speed/accuracy balance |

---

## Usage Examples

### Python

```python
from quran_transcription_api import QuranTranscriptionAPI

api = QuranTranscriptionAPI()

# Full pipeline
result = api.process(
    audio_path="recitation.mp3",
    surah_hint=56,
    start_ayah=1,
    end_ayah=40,
    output_path="surah_56.json"
)

print(f"Detected {result['pipeline_results']['detection']['ayahs_detected']} ayahs")
```

### cURL

```bash
curl -X POST http://localhost:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path": "/path/to/recitation.mp3",
    "surah_hint": 56,
    "output_path": "output/surah_56.json"
  }'
```

---

## Performance Considerations

1. **GPU Acceleration**: Use `device: "cuda"` or `device: "mps"` (Apple Silicon) for 3-5x faster transcription

2. **Checkpointing**: Enable `use_checkpoint` to resume failed processing and avoid re-transcribing

3. **Sequential Mode**: Always use when surah is known - it's more accurate and faster than sliding window

4. **Batch Processing**: For multiple files, reuse the loaded Whisper model by keeping the API instance alive

5. **Memory**: Large models (medium, large) require 4-8 GB RAM. Use smaller models for constrained environments.

---

## Changelog

### v1.0.0 (2024-11)
- Initial API release
- Whisper transcription with MPS/CUDA/CPU support
- Sequential ayah detection with boundary validation
- Fuzzy word matching for Whisper error tolerance
- Tarteel format export
- Checkpoint/resume support
