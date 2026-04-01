# WhisperX Engine Transition — Design Spec

**Date:** 2026-04-01
**Status:** Approved
**Scope:** Add whisperx as a selectable transcription engine alongside openai-whisper, across local and remote paths, with UI controls.

---

## 1. Goal

Replace the single-engine openai-whisper transcription path with a dual-engine system. Users select `openai-whisper` or `whisperx` via a UI dropdown. whisperx uses faster-whisper (batched inference) + wav2vec2 forced alignment for higher accuracy and faster throughput, especially on GPU.

---

## 2. Architecture

The `engine` parameter flows top-to-bottom through the stack. Nothing downstream of `WhisperTranscriber` changes.

```
UI (app.py)
  └─ engine selector: "openai-whisper" | "whisperx"
       │
       ├─ Local path
       │    WhisperTranscriber(model_name, device, engine)
       │    → normalised output dict (existing shape)
       │
       └─ Remote path
            transcribe_audio_via_remote(..., engine=...)
            POST /api/v1/transcribe-file  { engine: "whisperx" }
            server.py → WhisperTranscriber(model_name, device, engine)
            → same normalised output over HTTP
```

`extract_word_timestamps()`, `TranscribedWord`, the DTW alignment pipeline, and all downstream consumers are **unchanged**. whisperx's `score` field is mapped to `probability` inside `WhisperTranscriber` before returning.

---

## 3. `WhisperTranscriber` (`src/audio_processing_utils.py`)

### Constructor

```python
WhisperTranscriber(model_name="turbo", device="auto", engine="openai-whisper")
```

### `load_model()`

Branches on `self.engine`:
- `"openai-whisper"` → `whisper.load_model(model_name, device=device)` (unchanged)
- `"whisperx"` → `whisperx.load_model(model_name, device, compute_type=compute_type)`
  - `compute_type` auto-selected: `"float16"` for CUDA/MPS, `"int8"` for CPU

### `transcribe()`

Branches on `self.engine`:
- `"openai-whisper"` → existing code path, unchanged
- `"whisperx"` → two-step pipeline:
  1. `model.transcribe(audio, batch_size=16, language=language)` — batched fast transcription
  2. `whisperx.load_align_model(language_code, device)` → `whisperx.align(segments, ...)` — forced wav2vec2 alignment for precise word timestamps
  - `score` per word mapped to `probability` before return

### Alignment model caching

A module-level dict `_ALIGN_MODEL_CACHE: Dict[Tuple[str, str], Tuple[model, metadata]]` keyed by `(language, device)` caches the wav2vec2 alignment model across calls. Downloads ~400MB from HuggingFace on first use for Arabic (`facebook/wav2vec2-large-xlsr-53-arabic`). No HuggingFace token required.

---

## 4. Remote Server (`src/whisper_service/server.py`)

### `/api/v1/transcribe-file`

New optional form field:
```
engine: "openai-whisper" | "whisperx"   (default: "openai-whisper")
```
Validated against the allowed set. Passed to `get_transcriber(model_name, device, engine)`.

Transcriber cache key becomes `(model_name, device, engine)`.

### `/api/v1/capabilities`

Response gains:
```json
{ "engines": ["openai-whisper", "whisperx"] }
```
`"whisperx"` included only if `importlib.util.find_spec("whisperx")` is not None — graceful degradation for environments without whisperx installed.

---

## 5. Remote Client (`src/whisper_remote_client.py`)

`transcribe_audio_via_remote()` gains:
```python
engine: str = "openai-whisper"
```
Added to `data` dict in POST body. No other changes to auth, error handling, or response parsing.

---

## 6. UI (`app.py`)

New **Engine** selector in the transcription settings panel, below the Model selector:

```
Engine:  ○ openai-whisper   ● whisperx
```

- **Remote backend selected:** available engines come from `fetch_whisper_capabilities()["engines"]`. Defaults to `"whisperx"` if present in the list.
- **Local backend selected:** available engines determined by `importlib.util.find_spec` checks at render time. Defaults to `"openai-whisper"` if whisperx is not installed locally.
- Stored in `st.session_state["transcription_engine"]`.
- Passed to `WhisperTranscriber(engine=...)` (local) or `transcribe_audio_via_remote(engine=...)` (remote).

---

## 7. Dependencies

### `src/whisper_service/requirements.txt`
Add:
```
whisperx>=3.1.1
```

### `src/whisper_service/Dockerfile`
No structural changes. whisperx installs cleanly into the existing `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` image. The wav2vec2 Arabic alignment model is downloaded on first whisperx request and cached at `~/.cache/huggingface/` inside the container.

Optional pre-warm layer to avoid slow first request:
```dockerfile
RUN python -c "import whisperx; whisperx.load_align_model('ar', 'cpu')"
```

### `requirements.txt` / `pyproject.toml` (local app)
Add `whisperx` as an optional dependency. The UI engine selector only shows whisperx if it is importable.

---

## 8. Testing

### New: `tests/test_whisper_transcriber_engine.py`
- Monkeypatches `whisperx.load_model`, `whisperx.load_align_model`, `whisperx.align`
- Asserts `transcribe()` output contains `probability` (not `score`) — normalisation confirmed
- Asserts `compute_type` is `"float16"` for CUDA, `"int8"` for CPU
- Asserts alignment model cache is reused on second call (no double download)

### Updated: `tests/test_whisper_remote_client.py`
- Existing POST capture tests gain assertion: `captured["data"]["engine"] == "openai-whisper"` (default)
- New test: `engine="whisperx"` is forwarded in POST data

---

## 9. Files Changed

| File | Change |
|------|--------|
| `src/audio_processing_utils.py` | Add `engine` param to `WhisperTranscriber`, two-step whisperx pipeline |
| `src/whisper_service/server.py` | Accept `engine` field, update cache key, add `engines` to capabilities |
| `src/whisper_remote_client.py` | Add `engine` param to `transcribe_audio_via_remote` |
| `app.py` | Engine selector UI, pass engine through both paths |
| `src/whisper_service/requirements.txt` | Add `whisperx>=3.1.1` |
| `requirements.txt` | Add `whisperx` (optional) |
| `tests/test_whisper_transcriber_engine.py` | New test module |
| `tests/test_whisper_remote_client.py` | Add engine field assertions |

---

## 10. Out of Scope

- Speaker diarization (single-speaker recitation)
- Streaming / real-time transcription
- whisperx for languages other than Arabic (architecture supports it, not tested)
- Replacing the DTW alignment pipeline with whisperx alignment output
