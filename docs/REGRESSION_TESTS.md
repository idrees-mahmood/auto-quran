# Regression Test Framework

Automated testing to ensure algorithm changes don't break previously working transcription-to-ayah matching.

## Quick Start

```bash
# Run all tests
python regression_tests.py run

# List existing test fixtures
python regression_tests.py list

# Capture a new baseline
python regression_tests.py capture \
  --audio "path/to/audio.mp3" \
  --surah 56 --start 1 --end 40 \
  --id "reciter_name"
```

## Directory Structure

```
data/
├── transcriptions/              # Whisper transcriptions (cached by audio hash)
│   └── {hash}_{filename}.json   # Reusable across tests
├── fixtures/                    # Test case definitions
│   └── {surah}_{range}_{id}/
│       ├── metadata.json        # Test configuration
│       └── expected.json        # Expected ayah matches
└── audio_processed/             # Preprocessed audio files (16kHz mono WAV)
```

## Concepts

### Test Fixture
A fixture defines what a "correct" result looks like for a specific audio file:
- **Audio reference**: Path and hash of the source audio
- **Expected ayahs**: List of surah:ayah pairs that should be detected
- **Confidence threshold**: 80% minimum for a "pass"

### Pass Criteria
A test **passes** when:
1. ✅ All expected ayahs are detected
2. ✅ All detected ayahs have ≥80% confidence

A test **fails** when:
- ❌ Any expected ayah is missing, OR
- ❌ Any detected ayah has <80% confidence

## Commands

### `capture` - Create a Test Fixture

```bash
python regression_tests.py capture \
  --audio "path/to/recitation.mp3" \
  --surah 56 \
  --start 1 \
  --end 40 \
  --id "sheikh_name" \
  --model turbo \
  --expected-json /path/to/expected.json
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--audio`, `-a` | Yes | - | Path to audio file |
| `--surah`, `-s` | Yes | - | Surah number |
| `--start` | Yes | - | Start ayah number |
| `--end` | Yes | - | End ayah number |
| `--id`, `-i` | No | "test" | Short identifier for the test |
| `--desc`, `-d` | No | "" | Human-readable description |
| `--model`, `-m` | No | "turbo" | Whisper model for transcription |
| `--expected-json`, `-e` | No | - | Pre-defined expected ayahs (skips detection) |

#### Expected JSON Format
```json
[
  {"surah": 56, "ayah": 1},
  {"surah": 56, "ayah": 2},
  {"surah": 56, "ayah": 3}
]
```

Or simply a list of ayah numbers (surah inferred from `--surah`):
```json
[1, 2, 3, 4, 5]
```

### `run` - Execute Tests

```bash
# Run all fixtures
python regression_tests.py run

# Run specific fixture
python regression_tests.py run --fixture 56_1-40_ali_salah_omar
```

### `list` - View Fixtures

```bash
python regression_tests.py list
```

Output:
```
Found 1 test fixture(s):

  56_1-40_ali_salah_omar
    Surah 56 Ayahs 1-40
    Surah 56:1-40
    Expected: 40 ayahs
```

## Test Output

```
════════════════════════════════════════════════════════════
Running 1 regression test(s)
════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────
Fixture: 56_1-40_ali_salah_omar
Status: PASSED
────────────────────────────────────────────────────────────
Expected Ayahs: 40
Matched Ayahs:  40
High Confidence (≥80%): 40
Pass Rate: 100.0%

Duration: 1.23s

════════════════════════════════════════════════════════════
SUMMARY: 1 passed, 0 failed, 0 errors
════════════════════════════════════════════════════════════
```

## Workflow

### 1. Establish Baseline
After confirming the algorithm works correctly via the UI:

```bash
# Create expected.json with all ayahs that should match
cat > /tmp/expected.json << 'EOF'
[
  {"surah": 56, "ayah": 1},
  {"surah": 56, "ayah": 2},
  ...
]
EOF

# Capture fixture
python regression_tests.py capture \
  --audio "path/to/audio.mp3" \
  --surah 56 --start 1 --end 40 \
  --id "reciter_name" \
  --expected-json /tmp/expected.json
```

### 2. Run Before Changes
```bash
python regression_tests.py run
# Should show PASSED
```

### 3. Make Algorithm Changes
Edit `alignment_utils.py` or other detection code.

### 4. Run After Changes
```bash
python regression_tests.py run
# Should still show PASSED (no regression)
```

### 5. If Test Fails
- Review the missing/low-confidence ayahs
- Either fix the algorithm or update the expected baseline if the change is intentional

## Transcription Caching

Transcriptions are cached by audio file hash in `data/transcriptions/`:
- Same audio file → same transcription (no re-processing)
- Different audio file → new transcription
- Hash mismatch → re-transcribe

This allows fast test runs since Whisper doesn't need to re-process audio.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All tests passed |
| 1 | One or more tests failed/errored |

Useful for CI/CD integration:
```bash
python regression_tests.py run || echo "Regression detected!"
```

## Adding New Test Cases

1. **Get a working audio file** with known correct ayah range
2. **Verify via UI** that detection works correctly
3. **Create expected.json** listing all ayahs that should match
4. **Run capture** to create the fixture
5. **Commit** the fixture to version control (metadata.json + expected.json)

Note: Audio files and transcriptions are typically gitignored due to size.
