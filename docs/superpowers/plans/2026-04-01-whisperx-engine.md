# WhisperX Engine Transition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `whisperx` as a selectable transcription engine alongside `openai-whisper`, accessible from both the local app and the remote Docker service, with a UI engine selector.

**Architecture:** An `engine` parameter (`"openai-whisper"` | `"whisperx"`) is added to `WhisperTranscriber`, which internally runs either the existing openai-whisper path or the whisperx two-step pipeline (batched transcription + wav2vec2 forced alignment). The `engine` param flows through the remote HTTP API as a new optional form field. The UI reads available engines from server capabilities and exposes a selectbox.

**Tech Stack:** `whisperx>=3.1.1` (wraps `faster-whisper`, `ctranslate2`, `pyannote.audio`), Flask, Streamlit, pytest/monkeypatch

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `src/audio_processing_utils.py` | Modify | Add `engine` param to `WhisperTranscriber`; whisperx two-step pipeline; alignment model cache |
| `src/whisper_service/server.py` | Modify | Accept `engine` form field; `engines` list in capabilities response |
| `src/whisper_remote_client.py` | Modify | Add `engine` param to `transcribe_audio_via_remote` |
| `app.py` | Modify | Engine selector UI; wire engine through both local and remote paths |
| `src/whisper_service/requirements.txt` | Modify | Add `whisperx>=3.1.1` |
| `requirements.txt` | Modify | Add `whisperx>=3.1.1` |
| `pyproject.toml` | Modify | Add `whisperx>=3.1.1` to optional deps |
| `tests/test_whisper_transcriber_engine.py` | Create | New unit tests for whisperx path |
| `tests/test_whisper_remote_client.py` | Modify | Add `engine` field assertions |

---

## Task 1: WhisperX support in `WhisperTranscriber`

**Files:**
- Modify: `src/audio_processing_utils.py`
- Create: `tests/test_whisper_transcriber_engine.py`

- [ ] **Step 1: Create the failing test file**

Create `tests/test_whisper_transcriber_engine.py`:

```python
"""Tests for WhisperTranscriber engine parameter and whisperx pipeline."""

from unittest.mock import MagicMock

import pytest

import src.audio_processing_utils as apu


@pytest.fixture(autouse=True)
def clear_align_cache():
    apu._ALIGN_MODEL_CACHE.clear()
    yield
    apu._ALIGN_MODEL_CACHE.clear()


def _make_mock_whisperx(word_score: float = 0.95) -> MagicMock:
    mock_wx = MagicMock()
    mock_wx.load_audio.return_value = b"fake-audio"
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "segments": [{"start": 0.0, "end": 2.0, "text": "مرحبا", "words": []}],
        "language": "ar",
    }
    mock_wx.load_model.return_value = mock_model
    mock_wx.load_align_model.return_value = (MagicMock(), {"language": "ar"})
    aligned_words = [{"word": "مرحبا", "start": 0.0, "end": 0.5, "score": word_score}]
    mock_wx.align.return_value = {
        "segments": [{"start": 0.0, "end": 2.0, "text": "مرحبا", "words": aligned_words}],
        "word_segments": aligned_words,
        "language": "ar",
    }
    return mock_wx


def test_whisperx_score_normalised_to_probability(monkeypatch, tmp_path):
    mock_wx = _make_mock_whisperx(word_score=0.95)
    monkeypatch.setattr(apu, "_whisperx_lib", mock_wx)

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake-audio-bytes")

    t = apu.WhisperTranscriber(model_name="turbo", device="cpu", engine="whisperx")
    t.model = mock_wx.load_model.return_value

    result = t.transcribe(str(audio_file))

    for segment in result["segments"]:
        for word in segment.get("words", []):
            assert "probability" in word, "score should be renamed to probability"
            assert "score" not in word, "original score key should be removed"
            assert word["probability"] == 0.95


def test_whisperx_compute_type_float16_for_cuda():
    assert apu._whisperx_compute_type("cuda") == "float16"


def test_whisperx_compute_type_float16_for_mps():
    assert apu._whisperx_compute_type("mps") == "float16"


def test_whisperx_compute_type_int8_for_cpu():
    assert apu._whisperx_compute_type("cpu") == "int8"


def test_alignment_model_cached_on_second_call(monkeypatch, tmp_path):
    mock_wx = _make_mock_whisperx()
    monkeypatch.setattr(apu, "_whisperx_lib", mock_wx)

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake-audio-bytes")

    t = apu.WhisperTranscriber(model_name="turbo", device="cpu", engine="whisperx")
    t.model = mock_wx.load_model.return_value

    t.transcribe(str(audio_file))
    t.transcribe(str(audio_file))

    assert mock_wx.load_align_model.call_count == 1, \
        "load_align_model should only be called once due to cache"


def test_openai_whisper_path_does_not_call_whisperx(monkeypatch, tmp_path):
    mock_wx = MagicMock()
    monkeypatch.setattr(apu, "_whisperx_lib", mock_wx)

    mock_whisper_lib = MagicMock()
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"segments": [], "text": "", "language": "ar"}
    mock_whisper_lib.load_model.return_value = mock_model
    monkeypatch.setattr(apu, "whisper", mock_whisper_lib)

    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake-audio-bytes")

    t = apu.WhisperTranscriber(model_name="turbo", device="cpu", engine="openai-whisper")
    t.transcribe(str(audio_file))

    mock_wx.load_audio.assert_not_called()
    mock_wx.align.assert_not_called()
    mock_model.transcribe.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_whisper_transcriber_engine.py -v
```

Expected: `AttributeError: module 'src.audio_processing_utils' has no attribute '_ALIGN_MODEL_CACHE'` (or similar — tests fail for the right reason).

- [ ] **Step 3: Add `_whisperx_lib` module-level import to `audio_processing_utils.py`**

In `src/audio_processing_utils.py`, after the existing whisper import block (after line ~31), add:

```python
try:
    import whisperx as _whisperx_lib
except ImportError:
    _whisperx_lib = None
```

- [ ] **Step 4: Add module-level cache and helper functions**

In `src/audio_processing_utils.py`, after the `logger = logging.getLogger(__name__)` line (around line 46), add:

```python
_ALIGN_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


def _whisperx_compute_type(device: str) -> str:
    """Select CTranslate2 compute type based on device."""
    if device in ("cuda", "mps"):
        return "float16"
    return "int8"


def _get_align_model(language: str, device: str) -> Tuple[Any, Any]:
    """Return cached wav2vec2 alignment model, downloading on first call."""
    key = (language, device)
    if key not in _ALIGN_MODEL_CACHE:
        if _whisperx_lib is None:
            raise ImportError("whisperx is not installed. Run: pip install whisperx")
        model_a, metadata = _whisperx_lib.load_align_model(
            language_code=language, device=device
        )
        _ALIGN_MODEL_CACHE[key] = (model_a, metadata)
    return _ALIGN_MODEL_CACHE[key]
```

- [ ] **Step 5: Update `WhisperTranscriber.__init__` to accept `engine` param**

In `src/audio_processing_utils.py`, change the `__init__` signature at line ~274:

```python
def __init__(self, model_name: str = "turbo", device: str = "auto", engine: str = "openai-whisper"):
    """
    Initialize Whisper transcriber.

    Args:
        model_name: Whisper model size (tiny, base, small, medium, large, turbo)
        device: Device to run on (cpu, cuda, mps, auto)
               'auto' will detect best available device
        engine: Transcription engine ('openai-whisper' or 'whisperx')
    """
    if engine not in ("openai-whisper", "whisperx"):
        raise ValueError(f"Unknown engine '{engine}'. Use 'openai-whisper' or 'whisperx'.")

    if engine == "openai-whisper" and whisper is None:
        raise ImportError(
            "OpenAI Whisper is required. Install with: pip install openai-whisper\n"
            "For better performance, consider: pip install whisperx"
        )

    self.model_name = model_name
    self.engine = engine
    self.device = self._detect_device(device)
    self.model = None

    logger.info(
        f"Initializing Whisper model: {model_name} on device: {self.device} (engine: {engine})"
    )
```

- [ ] **Step 6: Update `load_model()` to branch on engine**

Replace the existing `load_model` method body:

```python
def load_model(self):
    """Load the model (lazy loading). Branches on self.engine."""
    if self.model is not None:
        return

    if self.engine == "whisperx":
        if _whisperx_lib is None:
            raise ImportError("whisperx is not installed. Run: pip install whisperx")
        compute_type = _whisperx_compute_type(self.device)
        self.model = _whisperx_lib.load_model(
            self.model_name, self.device, compute_type=compute_type
        )
        logger.info(
            f"WhisperX model loaded: {self.model_name} on {self.device} ({compute_type})"
        )
    else:
        try:
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info(f"Whisper model loaded: {self.model_name} on {self.device}")
        except (NotImplementedError, RuntimeError):
            if self.device == "mps":
                logger.warning("MPS not supported for this model, falling back to CPU")
                self.device = "cpu"
                self.model = whisper.load_model(self.model_name, device="cpu")
                logger.info(f"Whisper model loaded: {self.model_name} on CPU")
            else:
                raise
```

- [ ] **Step 7: Update `transcribe()` to branch on engine and add `_transcribe_whisperx()`**

Replace the `transcribe` method body:

```python
def transcribe(
    self,
    audio_path: str,
    language: str = "ar",
    word_timestamps: bool = True,
    save_json: Optional[str] = None
) -> Dict[str, Any]:
    """
    Transcribe audio file. Delegates to openai-whisper or whisperx based on self.engine.
    """
    self.load_model()

    logger.info(f"Transcribing audio: {audio_path}")
    logger.info(
        f"Language: {language}, Word timestamps: {word_timestamps}, Engine: {self.engine}"
    )

    if self.engine == "whisperx":
        result = self._transcribe_whisperx(audio_path, language)
    else:
        result = self.model.transcribe(
            audio_path,
            language=language,
            word_timestamps=word_timestamps,
            verbose=False,
        )

    if save_json:
        os.makedirs(os.path.dirname(save_json), exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Raw transcription saved to: {save_json}")

    return result
```

Add the new `_transcribe_whisperx` method directly after `transcribe`:

```python
def _transcribe_whisperx(self, audio_path: str, language: str) -> Dict[str, Any]:
    """
    Run the two-step whisperx pipeline:
    1. Batched transcription via faster-whisper
    2. Forced wav2vec2 alignment for precise word timestamps
    Output is normalised so 'score' becomes 'probability' for downstream compatibility.
    """
    audio = _whisperx_lib.load_audio(audio_path)

    result = self.model.transcribe(audio, batch_size=16, language=language)
    detected_language = result.get("language", language)
    logger.info(f"WhisperX transcription complete, detected language={detected_language}")

    model_a, metadata = _get_align_model(detected_language, self.device)
    result = _whisperx_lib.align(
        result["segments"], model_a, metadata, audio, self.device, print_progress=False
    )
    logger.info("WhisperX forced alignment complete")

    # Normalise: whisperx uses 'score', existing extract_word_timestamps() expects 'probability'
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            if "score" in word and "probability" not in word:
                word["probability"] = word.pop("score")

    return result
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
pytest tests/test_whisper_transcriber_engine.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 9: Run the full test suite to check no regressions**

```bash
pytest tests/ -v
```

Expected: All previously-passing tests still pass.

- [ ] **Step 10: Commit**

```bash
git add src/audio_processing_utils.py tests/test_whisper_transcriber_engine.py
git commit -m "feat: add whisperx engine option to WhisperTranscriber with two-step pipeline"
```

---

## Task 2: Remote server — `engine` form field + `engines` in capabilities

**Files:**
- Modify: `src/whisper_service/server.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_whisper_service_server.py`:

```python
"""Tests for whisper service server engine field and capabilities."""

import importlib.util
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def app():
    with patch("src.whisper_service.server.detect_available_devices") as mock_detect:
        mock_detect.return_value = (["auto", "cpu"], False)
        from src.whisper_service import server
        return server.create_app()


def test_capabilities_includes_openai_whisper_always(app):
    with app.test_client() as client:
        resp = client.get("/api/v1/capabilities")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "engines" in data["capabilities"]
    assert "openai-whisper" in data["capabilities"]["engines"]


def test_capabilities_includes_whisperx_when_installed(app, monkeypatch):
    import importlib.util as ilu
    original = ilu.find_spec
    monkeypatch.setattr(ilu, "find_spec", lambda name: MagicMock() if name == "whisperx" else original(name))

    with app.test_client() as client:
        resp = client.get("/api/v1/capabilities")
    data = resp.get_json()
    assert "whisperx" in data["capabilities"]["engines"]


def test_capabilities_excludes_whisperx_when_not_installed(app, monkeypatch):
    import importlib.util as ilu
    original = ilu.find_spec
    monkeypatch.setattr(ilu, "find_spec", lambda name: None if name == "whisperx" else original(name))

    with app.test_client() as client:
        resp = client.get("/api/v1/capabilities")
    data = resp.get_json()
    assert "whisperx" not in data["capabilities"]["engines"]


def test_transcribe_file_rejects_unknown_engine(app, tmp_path):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake")
    with app.test_client() as client:
        resp = client.post(
            "/api/v1/transcribe-file",
            data={"model": "base", "device": "cpu", "engine": "invalid-engine"},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "engine" in data["error"]["message"].lower()
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_whisper_service_server.py -v
```

Expected: FAIL — `KeyError: 'engines'` or similar.

- [ ] **Step 3: Update `capabilities` route to include `engines` list**

In `src/whisper_service/server.py`, add this import at the top of the file (with other stdlib imports):

```python
import importlib.util
```

Update the `capabilities` route (currently around line 146):

```python
@app.route("/api/v1/capabilities", methods=["GET"])
def capabilities():
    auth_error = _check_auth(request)
    if auth_error is not None:
        payload, status = auth_error
        return jsonify(payload), status

    devices, gpu_available = detect_available_devices()
    engines = ["openai-whisper"]
    if importlib.util.find_spec("whisperx") is not None:
        engines.append("whisperx")

    return jsonify(
        {
            "success": True,
            "capabilities": {
                "models": WHISPER_MODELS,
                "devices": devices,
                "gpu_available": gpu_available,
                "engines": engines,
            },
        }
    )
```

- [ ] **Step 4: Add `engine` validation and update `get_transcriber` cache key**

Replace the existing `get_transcriber` function:

```python
ALLOWED_ENGINES = {"openai-whisper", "whisperx"}


def get_transcriber(model_name: str, requested_device: str, engine: str = "openai-whisper") -> WhisperTranscriber:
    """Reuse loaded transcribers to avoid repeated model load overhead."""
    key = (model_name, requested_device, engine)

    with _CACHE_LOCK:
        transcriber = _TRANSCRIBER_CACHE.get(key)
        if transcriber is None:
            transcriber = WhisperTranscriber(
                model_name=model_name, device=requested_device, engine=engine
            )
            _TRANSCRIBER_CACHE[key] = transcriber

    return transcriber
```

- [ ] **Step 5: Update `transcribe_file` route to read and validate `engine`**

In the `transcribe_file` route, after reading `model_name` and `requested_device` (around line 184), add:

```python
engine = request.form.get("engine", "openai-whisper")

if engine not in ALLOWED_ENGINES:
    return (
        jsonify(
            {
                "success": False,
                "error": {
                    "message": f"Unsupported engine '{engine}'. Allowed: {', '.join(sorted(ALLOWED_ENGINES))}"
                },
            }
        ),
        400,
    )
```

Then update the `get_transcriber` call (the only call to `get_transcriber` in this route):

```python
transcriber = get_transcriber(model_name, requested_device, engine)
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_whisper_service_server.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 7: Run full test suite**

```bash
pytest tests/ -v
```

Expected: No regressions.

- [ ] **Step 8: Commit**

```bash
git add src/whisper_service/server.py tests/test_whisper_service_server.py
git commit -m "feat: add engine field to whisper service transcribe endpoint and capabilities"
```

---

## Task 3: Remote client — forward `engine` param

**Files:**
- Modify: `src/whisper_remote_client.py`
- Modify: `tests/test_whisper_remote_client.py`

- [ ] **Step 1: Add `engine` assertion to existing test**

In `tests/test_whisper_remote_client.py`, in `test_transcribe_audio_via_remote_posts_file_and_returns_transcription` (around line 63), add this assertion after the existing `assert captured["data"]["word_timestamps"] == "true"`:

```python
assert captured["data"]["engine"] == "openai-whisper"
```

- [ ] **Step 2: Add a new test for explicit engine forwarding**

Add at the end of `tests/test_whisper_remote_client.py`:

```python
def test_transcribe_audio_via_remote_forwards_engine(monkeypatch, tmp_path):
    monkeypatch.delenv("WHISPER_REMOTE_API_KEY", raising=False)
    monkeypatch.delenv("WHISPER_CF_ACCESS_CLIENT_ID", raising=False)
    monkeypatch.delenv("WHISPER_CF_ACCESS_CLIENT_SECRET", raising=False)
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake-audio-bytes")

    captured = {}

    def fake_post(url, files, data, headers, timeout):
        captured["data"] = dict(data)
        return DummyResponse(
            {"success": True, "transcription": {"text": "ok", "segments": []}}
        )

    monkeypatch.setattr("src.whisper_remote_client.requests.post", fake_post)

    transcribe_audio_via_remote(
        base_url="http://whisper.local:9000",
        audio_path=str(audio_path),
        model_name="turbo",
        device="cuda",
        engine="whisperx",
    )

    assert captured["data"]["engine"] == "whisperx"
```

- [ ] **Step 3: Run to verify new assertion and test fail**

```bash
pytest tests/test_whisper_remote_client.py -v
```

Expected: `test_transcribe_audio_via_remote_posts_file_and_returns_transcription` fails (`KeyError: 'engine'`) and `test_transcribe_audio_via_remote_forwards_engine` fails similarly.

- [ ] **Step 4: Update `transcribe_audio_via_remote` to accept and forward `engine`**

In `src/whisper_remote_client.py`, update the `transcribe_audio_via_remote` function signature (around line 167):

```python
def transcribe_audio_via_remote(
    base_url: str,
    audio_path: str,
    model_name: str,
    device: str,
    language: str = "ar",
    word_timestamps: bool = True,
    timeout_seconds: int = 1800,
    engine: str = "openai-whisper",
    api_key: str | None = None,
    cf_access_client_id: str | None = None,
    cf_access_client_secret: str | None = None,
) -> Dict[str, Any]:
```

In the `data` dict (around line 201), add the `engine` field:

```python
data = {
    "model": model_name,
    "device": device,
    "language": language,
    "word_timestamps": str(word_timestamps).lower(),
    "engine": engine,
}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_whisper_remote_client.py -v
```

Expected: All tests PASS.

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/ -v
```

- [ ] **Step 7: Commit**

```bash
git add src/whisper_remote_client.py tests/test_whisper_remote_client.py
git commit -m "feat: forward engine param through remote whisper client"
```

---

## Task 4: UI engine selector in `app.py`

**Files:**
- Modify: `app.py`

No automated tests for Streamlit UI. Verify manually by running `streamlit run app.py`.

- [ ] **Step 1: Add `transcription_engine` to session state defaults**

In `app.py`, in the `defaults` dict (around line 140), add after the `whisper_server_capabilities` entry:

```python
'transcription_engine': 'openai-whisper',
```

- [ ] **Step 2: Add engine options derivation after model/device options (around line 1330)**

After the existing block that builds `model_options` and `device_options` (after line ~1336), add:

```python
# Derive available engines from capabilities (remote) or local importlib (local)
import importlib.util as _ilu
DEFAULT_ENGINES = ["openai-whisper"]
engine_options = DEFAULT_ENGINES[:]
if backend_mode == "remote":
    remote_caps = ss.get("whisper_server_capabilities")
    if isinstance(remote_caps, dict):
        engine_options = remote_caps.get("engines") or DEFAULT_ENGINES
else:
    if _ilu.find_spec("whisperx") is not None:
        engine_options = ["openai-whisper", "whisperx"]

current_engine = ss.get("transcription_engine", "openai-whisper")
if current_engine not in engine_options:
    current_engine = engine_options[0]
```

- [ ] **Step 3: Add the engine selectbox below the 2-column model/device layout**

After the `col_config2` block ends (after line ~1375, before the "Start Transcription" button), add:

```python
whisper_engine = st.selectbox(
    "Transcription Engine",
    engine_options,
    index=engine_options.index(current_engine),
    help=(
        "openai-whisper: original Whisper inference. "
        "whisperx: batched faster-whisper + wav2vec2 forced alignment "
        "(higher accuracy, faster on GPU)."
    ),
)
ss.transcription_engine = whisper_engine
```

- [ ] **Step 4: Pass `engine` to the remote transcription call**

In the `transcribe_audio_via_remote(...)` call (around line 1393), add `engine=whisper_engine`:

```python
transcription_result = transcribe_audio_via_remote(
    base_url=ss.whisper_server_url,
    audio_path=ss.processed_audio_path,
    model_name=whisper_model,
    device=device_option,
    language="ar",
    word_timestamps=True,
    timeout_seconds=1800,
    engine=whisper_engine,
    api_key=ss.whisper_remote_api_key,
    cf_access_client_id=ss.whisper_cf_access_client_id,
    cf_access_client_secret=ss.whisper_cf_access_client_secret,
)
```

- [ ] **Step 5: Pass `engine` to `transcribe_audio_workflow` (local path)**

Update the `transcribe_audio_workflow` call (around line 1414):

```python
transcription_result = transcribe_audio_workflow(
    audio_path=ss.processed_audio_path,
    model_name=whisper_model,
    device=device_option,
    engine=whisper_engine,
    output_dir=output_dir,
    progress_bar=progress_bar,
    status_text=status_text,
)
```

- [ ] **Step 6: Update `transcribe_audio_workflow` signature and `WhisperTranscriber` instantiation**

In `app.py`, update `transcribe_audio_workflow` function signature (around line 519):

```python
def transcribe_audio_workflow(
    audio_path: str,
    model_name: str,
    device: str,
    output_dir: str,
    progress_bar,
    status_text,
    engine: str = "openai-whisper",
) -> Optional[Dict]:
```

Update the `WhisperTranscriber` instantiation inside it (line 556):

```python
transcriber = WhisperTranscriber(model_name=model_name, device=device, engine=engine)
```

- [ ] **Step 7: Manual verification**

Run the app and verify:
```bash
streamlit run app.py
```

1. Upload an audio file and preprocess it
2. Confirm the **Transcription Engine** selectbox appears below Model and Device selectors
3. Switch backend to **Local** — confirm only `openai-whisper` shows (or both if whisperx is locally installed)
4. Switch backend to **Remote** and click **Check Server** — confirm whisperx appears in the engine list if the server has it installed

- [ ] **Step 8: Commit**

```bash
git add app.py
git commit -m "feat: add engine selector UI and wire engine param through local and remote transcription paths"
```

---

## Task 5: Update dependencies

**Files:**
- Modify: `src/whisper_service/requirements.txt`
- Modify: `requirements.txt`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add whisperx to the service requirements**

In `src/whisper_service/requirements.txt`, add:

```
whisperx>=3.1.1
```

Final file:
```
flask>=3.0.0,<4
openai-whisper>=20231117
python-dotenv>=1.0.1
whisperx>=3.1.1
```

- [ ] **Step 2: Add whisperx to the root requirements**

In `requirements.txt`, add after the `openai-whisper` line:

```
whisperx>=3.1.1
```

- [ ] **Step 3: Add whisperx as an optional dependency in `pyproject.toml`**

In `pyproject.toml`, update `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
dev = [
  "pytest>=8.2.0",
]
whisperx = [
  "whisperx>=3.1.1",
]
```

- [ ] **Step 4: Rebuild the Docker service image on the server**

On `amplab.co.uk`:
```bash
git pull
cd config/whisper_remote
docker compose -f docker-compose.cloudflare.yml down
docker compose -f docker-compose.cloudflare.yml build whisper-service
docker compose -f docker-compose.cloudflare.yml up -d
docker compose -f docker-compose.cloudflare.yml logs whisper-service --tail=30
```

Expected: Service starts without errors. `curl https://whisper.amplab.co.uk/api/v1/capabilities` returns `"engines": ["openai-whisper", "whisperx"]`.

- [ ] **Step 5: Run full test suite one final time**

```bash
pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/whisper_service/requirements.txt requirements.txt pyproject.toml
git commit -m "feat: add whisperx dependency to service, local, and optional project deps"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|-----------------|-----------|
| `engine` param on `WhisperTranscriber` | Task 1 Step 5 |
| Two-step whisperx pipeline (transcribe + align) | Task 1 Step 7 |
| `score` → `probability` normalisation | Task 1 Step 7, tested in Step 1 |
| `float16`/`int8` compute_type auto-selection | Task 1 Steps 4 & 6, tested |
| Alignment model cache | Task 1 Step 4, tested |
| Server `engine` form field validation | Task 2 Steps 4-5 |
| `engines` list in capabilities response | Task 2 Step 3, tested |
| `whisperx` only in engines if installed | Task 2 Step 3, tested |
| `engine` forwarded in POST from client | Task 3, tested |
| UI engine selector | Task 4 Steps 2-3 |
| Remote default to whisperx if available | Task 4 Step 2 (`current_engine` from caps) |
| Local shows whisperx only if importable | Task 4 Step 2 (`find_spec` check) |
| `engine` wired through both call paths | Task 4 Steps 4-6 |
| `whisperx>=3.1.1` in service requirements | Task 5 Step 1 |
| `whisperx` in local requirements | Task 5 Steps 2-3 |

No gaps found.
