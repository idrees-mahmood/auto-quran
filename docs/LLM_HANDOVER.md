# LLM Handover Document

> Status as of: January 2026

## Current State Summary

The ayah detection system has been significantly enhanced with:

1. **Segment Splitting** - Fixed multi-ayah segment handling
2. **Repetition Detection** - Tracks when Qari repeats ayahs
3. **Word-Level Classification** - Maps each word to exact Quran position

---

## Known Issues / TODOs

### High Priority
| Issue | Description | File |
|-------|-------------|------|
| Ayahs 11-14 gap | Anfal recording skips these ayahs even with fixes | `alignment_utils.py` |
| Ayah 1 missing | Surah 56 test misses first ayah (preamble skip issue) | `alignment_utils.py` |

### Medium Priority
| Issue | Description |
|-------|-------------|
| Repetition detection | Single-word matching disabled; needs multi-word sequence matching |
| Word classification | Forward-only matching; backwards jumps not fully supported |

### Low Priority
- UI could display word-level classifications in detail
- Streamlit `use_container_width` deprecation warning

---

## Testing Commands

```bash
# Run regression tests
python regression_tests.py run

# Test repetition detection on Anfal
python tests/test_repetition.py

# Test word-level classification
python tests/test_word_classification.py

# Debug segment analysis (in tests/debug/)
python tests/debug/debug_anfal_segments.py
```

### Test Status
- Regression: **97.5%** (39/40 ayahs)
- Missing: Surah 56:1 (preamble detection issue)

---

## Key Decision Points

### Algorithm Choice
| Scenario | Use |
|----------|-----|
| Standard recitation | Sequential detection (default) |
| Qari repeats ayahs | Repetition-aware + segment splitting |
| Need accurate Quran text | Word-level classification |

### Confidence Thresholds
| Context | Default | Notes |
|---------|---------|-------|
| Detection | 0.65-0.70 | Lower = more matches, more false positives |
| Repetition penalty | +0.15 | Extra confidence required for backwards jumps |

---

## Quick Reference

### Main Entry Points
| Task | Location |
|------|----------|
| Start UI | `./launch_ui.sh` or `streamlit run app.py` |
| Detection logic | `alignment_utils.py:AyahDetector` |
| Word classification | `alignment_utils.py:classify_transcription_words()` |

### Key Functions
```python
# Sequential detection (default)
detector.detect_ayahs_from_transcription(words, allow_repetition=False)

# Repetition-aware detection
detector.detect_ayahs_from_transcription(words, allow_repetition=True)

# Word-level classification
classifications = detector.classify_transcription_words(words, surah=8)
ayahs = reconstruct_ayahs(classifications, quran_data)
```

### Data Flow
```
Whisper → TranscribedWord[] → AyahDetector → RecitationEvent[] → Video
```

---

## Files Changed in Recent Sessions

| File | Changes |
|------|---------|
| `alignment_utils.py` | Added word classification, segment splitting, repetition detection |
| `app.py` | Added UI toggles for word classification and repetition |
| `tests/test_repetition.py` | Test for Anfal repetitions |
| `tests/test_word_classification.py` | Test for word-level mapping |
| `docs/ARCHITECTURE.md` | NEW - System architecture |

---

## Architecture Docs
See [docs/ARCHITECTURE.md](ARCHITECTURE.md) for:
- Detection algorithm details
- Key dataclasses
- UI configuration options
