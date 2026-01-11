# LLM Onboarding - AI Quran Video Composer

> Quick reference for AI assistants working with this codebase.

## What This Project Does

Generates Quran recitation videos with:
1. Word-level synchronized Arabic text overlays
2. English translations
3. AI-selected or black background videos
4. Audio from pre-processed reciters OR custom uploads

## Two Interfaces

| Interface | File | Use Case |
|-----------|------|----------|
| **Streamlit UI** | `app.py` | User-friendly web interface |
| **Jupyter Notebooks** | `video_gen.ipynb`, `audio_processing.ipynb` | Advanced/development |

## Two Workflows

### Workflow 1: Pre-processed Audio
Uses Tarteel.ai JSON files with existing word timestamps.
```
data/audio/*_updated.json → video_gen.ipynb → MP4 video
```

### Workflow 2: Custom Audio
Processes any MP3/WAV through Whisper AI.
```
Custom audio → Whisper → Ayah detection → Word alignment → JSON → Video
```

## Key Files

### Core Processing
| File | Purpose |
|------|---------|
| `utils.py` | `get_words_with_timestamps()`, `create_word_timed_video()`, `create_text_image()` |
| `audio_processing_utils.py` | `WhisperTranscriber`, `AudioPreprocessor`, `ArabicNormalizer` |
| `alignment_utils.py` | `AyahDetector`, `WordAligner`, `convert_to_tarteel_format()` |

### APIs & Utilities
| File | Purpose |
|------|---------|
| `LLM_utils.py` | OpenAI API for background video suggestions |
| `pexel_utils.py` | Pexels API for video downloads |
| `prompts.py` | LLM prompts with Islamic content filters |
| `quran_utils.py` | `Reciter` enum mapping names to JSON files |

### Testing
| File | Purpose |
|------|---------|
| `regression_tests.py` | CLI for capture/run/list test fixtures |
| `tests/test_repetition.py` | Test repetition-aware detection |
| `tests/test_word_classification.py` | Test word-level classification |
| `data/fixtures/` | Stored test cases |
| `data/transcriptions/` | Cached Whisper outputs |

## Detection Algorithms

### Sequential (Default)
Linear progression through ayahs. Best for standard recitations.
```python
detector.detect_ayahs_from_transcription(words, allow_repetition=False)
```

### Repetition-Aware
Handles Qari repetitions with segment splitting.
```python
detector.detect_ayahs_from_transcription(words, allow_repetition=True)
```
Enable in UI: Advanced Options → "Allow Repetitions"

### Word-Level Classification
Maps each word to exact Quran position with accurate reference text.
```python
classifications = detector.classify_transcription_words(words, surah=8)
ayahs = reconstruct_ayahs(classifications, quran_data)
```
Enable in UI: Advanced Options → "Word-Level Classification"

## Key Dataclasses

| Class | Location | Purpose |
|-------|----------|---------|
| `TranscribedWord` | `audio_processing_utils.py` | Word from Whisper |
| `WordClassification` | `alignment_utils.py` | Word-to-Quran mapping |
| `RecitationEvent` | `alignment_utils.py` | Detected ayah event |


## Data Formats

### Tarteel JSON (Word Timestamps)
```json
{
  "1:1": {
    "segments": [[1, 0, 480], [2, 480, 920]],
    "duration": 2500,
    "audio_url": "https://..."
  }
}
```
- `segments`: `[[word_position, start_ms, end_ms], ...]`
- Keys: `"surah:ayah"`

### Internal Word Dictionary
```python
{
    "word": "بِسْمِ",
    "start": 0.0,      # seconds (converted from ms)
    "end": 0.48,
    "aya": 1,
    "word_position": 1,
    "translation": {"en": "In the name"}
}
```

## Critical Conventions

### Timestamps
- Tarteel JSON: milliseconds
- Internal processing: seconds (divide by 1000)

### Arabic Text
Normalization for matching (in `ArabicNormalizer`):
- Remove diacritics
- Normalize Alif: أ إ آ ٱ → ا
- Normalize Ya: ى → ي

### Islamic Compliance
`prompts.py` filters LLM suggestions:
- NO people, body parts, clothing
- Prefer nature scenes
- Use "islam" prefix for worship queries

## Session State (Streamlit)

Key variables in `st.session_state`:
- `transcription_result` - Whisper output
- `detected_ayahs` - Matched ayahs
- `aligned_ayahs` - Word-aligned data
- `generated_video` - Output path

## Storage Paths

| Purpose | Path |
|---------|------|
| Quran text | `data/quran/quran.json` |
| Translations | `data/quran/English wbw translation.json` |
| Reciter data | `data/audio/*_updated.json` |
| Transcription cache | `data/transcriptions/` |
| Test fixtures | `data/fixtures/` |
| Processed audio | `data/audio_processed/` |
| Temp files | `temp/` |

## Common Tasks

### Add New Reciter
1. Download JSON from Tarteel.ai (with segmentation)
2. Add to `Reciter` enum in `quran_utils.py`
3. Update UI dropdown in `app.py`

### Run Tests
```bash
python regression_tests.py run           # All tests
python regression_tests.py run --name X  # Specific test
```

### Test Mode
No API keys needed:
- UI: Toggle "Test Mode"
- Notebooks: `TEST_MODE = True`
- Result: Black background, full text overlays

## Dependencies

System: FFmpeg, Google Chrome
Python: moviepy, openai-whisper, rapidfuzz, streamlit, html2image

## macOS Notes

- FFmpeg path: `/opt/homebrew/bin/` (added in `utils.py`)
- MPS acceleration: automatic with `device="auto"`
- Setup: `./setup_macos.sh`
