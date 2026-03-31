# Development Guide

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      User Interfaces                         │
├─────────────────────────────┬────────────────────────────────┤
│   Streamlit UI (app.py)     │   Jupyter Notebooks            │
│   - Video Generation Tab    │   - video_gen.ipynb            │
│   - Custom Audio Tab        │   - audio_processing.ipynb     │
│   - Settings Management     │   - data_processing.ipynb      │
└─────────────────────────────┴────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    Core Processing Modules                   │
├──────────────────┬───────────────────┬───────────────────────┤
│ Audio Processing │ Video Composition │ External APIs         │
│ - audio_proc...  │ - utils.py        │ - LLM_utils.py        │
│ - alignment...   │ - quran_utils.py  │ - pexel_utils.py      │
└──────────────────┴───────────────────┴───────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                         Data Layer                           │
├─────────────────┬────────────────────┬───────────────────────┤
│ data/quran/     │ data/audio/        │ data/fonts/           │
│ - quran.json    │ - *_updated.json   │ - *.ttf               │
│ - English*.json │ (Tarteel format)   │                       │
└─────────────────┴────────────────────┴───────────────────────┘
```

## Module Reference

### Core Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `utils.py` | Video composition, text rendering | `get_words_with_timestamps()`, `create_text_image()`, `create_word_timed_video()` |
| `quran_utils.py` | Reciter enum and mapping | `Reciter` enum |
| `LLM_utils.py` | OpenAI API for background selection | `get_video_suggestions()`, `make_openai_request()` |
| `pexel_utils.py` | Pexels API for video downloads | `select_and_download_video()` |
| `prompts.py` | LLM prompts with Islamic filters | Video selection prompts |

### Audio Processing Modules

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `audio_processing_utils.py` | Whisper transcription, audio preprocessing | `WhisperTranscriber`, `AudioPreprocessor`, `ArabicNormalizer` |
| `alignment_utils.py` | Ayah detection, word alignment | `AyahDetector`, `WordAligner`, `convert_to_tarteel_format()` |
| `dtw_alignment.py` | DTW-based alignment engine | `build_banded_similarity_matrix()`, `run_dp_alignment()`, `build_recitation_events()`, `DTWConfig` |

## Data Formats

### Tarteel Format (Reciter JSON)
```json
{
  "1:1": {
    "segments": [[1, 0, 480], [2, 480, 920]],  // [word_pos, start_ms, end_ms]
    "duration": 2500,
    "audio_url": "https://..."
  }
}
```

### Word Dictionary (Internal)
```python
{
    "word": "بِسْمِ",
    "start": 0.0,          # seconds
    "end": 0.48,
    "aya": 1,
    "word_position": 1,
    "translation": {"en": "In the name"}
}
```

### Transcription Checkpoint
```json
{
  "transcription": { "text": "...", "words": [...] },
  "metadata": {
    "audio_hash": "sha256...",
    "model_name": "base",
    "timestamp": 1697285341.2
  }
}
```

## Key Algorithms

### Arabic Text Normalization
Handles diacritics and letter variants for fuzzy matching:
- Remove diacritics (U+064B to U+065F, U+0670)
- Normalize Alif variants: أ إ آ ٱ → ا
- Normalize Ya variants: ى → ي

### Ayah Detection (Sequential)
Default mode — fast, no repetition handling:
1. Try matching expected ayah at current position
2. Consume words based on reference word count (±5 flexibility)
3. Move to next ayah on match
4. Single-word advance on no match

### Ayah Detection (DTW)
Used when `mode='dtw'` — handles repetitions, noise, and partial ayahs:
1. **Banded similarity matrix** — score every (word-window, ayah) pair within a diagonal band
2. **DP alignment** — find minimum-cost path through MATCH / SKIP_AYAH / NOISE transitions
3. **Noise second-pass** — greedily scan uncovered word regions for repeated ayahs
4. Produces `RecitationEvent` objects with `event_type` in `{full, partial, repetition, skip}`

### Word Alignment
Using `difflib.SequenceMatcher`:
- `equal`: Direct 1:1 mapping
- `replace`: Proportional distribution
- `insert`: Linear interpolation for missed words
- `delete`: Map to nearest reference position

## Islamic Content Rules

The LLM prompts in `prompts.py` enforce strict filters:
- **NO** people, body parts, clothing
- **NO** abstract/inappropriate imagery
- Prefer tangible nature scenes
- Use "islam" prefix for worship-related queries

## Testing

### Test Mode
Set `TEST_MODE = True` in notebooks or UI:
- Black background (no Pexels API)
- Skips LLM calls (no OpenAI API)
- Full text overlay functionality
- Perfect for timing verification

### Regression Tests
```bash
# Capture a test fixture
python regression_tests.py capture --audio path/to/audio.mp3 --name test_name --surah 56 --start 1 --end 40

# Run all tests
python regression_tests.py run

# List fixtures
python regression_tests.py list
```

See `docs/REGRESSION_TESTS.md` for full documentation.

## Storage Paths

| Purpose | Path |
|---------|------|
| Quran data | `data/quran/` |
| Reciter JSON | `data/audio/` |
| Fonts | `data/fonts/` |
| Transcription cache | `data/transcriptions/` |
| Test fixtures | `data/fixtures/` |
| Temp files | `temp/` (gitignored) |

## macOS / Apple Silicon

### MPS Acceleration
Automatic detection via `device="auto"`:
1. Check for MPS (Apple Silicon) → 2-3x faster
2. Check for CUDA (NVIDIA) 
3. Fallback to CPU

Note: Whisper may fall back to CPU due to sparse tensor limitations.

### Setup
```bash
./setup_macos.sh
```

Cross-platform recommended installer:

```bash
./install.sh
```

`install.sh` attempts `uv sync --extra dev` first, then falls back to `venv` + `pip` if uv is unavailable or sync fails.

## Common Development Tasks

### Adding a New Reciter
1. Download JSON from Tarteel.ai (with segmentation tag)
2. Add to `quran_utils.Reciter` enum
3. Update UI dropdown in `app.py`

### Custom Audio Processing
1. Use `audio_processing.ipynb` or UI Custom Audio tab
2. Process: Upload → Transcribe → Detect → Align → Export
3. Generated JSON works with video generation workflow

### Modifying Video Composition
Edit `create_word_timed_video()` in `utils.py`:
- Layering: Background → Overlay → Text
- Default 9:16 aspect ratio (1080x1920)
- Text via html2image + Chrome

## Dependencies

### System Requirements
- FFmpeg (Homebrew on macOS: `/opt/homebrew/bin/`)
- Google Chrome (for html2image)

### Key Python Packages
- `moviepy` - Video composition
- `openai-whisper` - Transcription
- `rapidfuzz` - Fuzzy matching
- `streamlit` - Web UI
- `html2image` - Text rendering

### Dependency Management (uv)

Primary dependency management is `uv`:

```bash
uv sync --extra dev
```

Run project commands with `uv run`:

```bash
uv run streamlit run app.py
uv run python regression_tests.py run
uv run pytest
```

Maintenance policy:
- `pyproject.toml` is the primary source of truth for dependencies
- `requirements.txt` is kept as a legacy `pip` fallback
- When dependencies change, update `pyproject.toml` first, then align `requirements.txt` if needed

## API Keys

| Service | Purpose | Required For |
|---------|---------|--------------|
| OpenAI | Background suggestions | Production mode |
| Pexels | Background videos | Production mode |

Not needed for Test Mode.
