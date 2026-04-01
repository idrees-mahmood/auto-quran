"""Tests for whisper service server engine field and capabilities."""

import importlib.util
import io
from unittest.mock import MagicMock, patch

import pytest


def test_capabilities_requires_api_key_when_enabled(monkeypatch):
    monkeypatch.setenv("WHISPER_SERVER_REQUIRE_AUTH", "true")
    monkeypatch.setenv("WHISPER_SERVER_API_KEY", "server-secret")

    from src.whisper_service.server import create_app

    app = create_app()
    client = app.test_client()

    response = client.get("/api/v1/capabilities")

    assert response.status_code == 401
    payload = response.get_json()
    assert payload["success"] is False
    assert "Unauthorized" in payload["error"]["message"]


def test_capabilities_allows_with_valid_api_key(monkeypatch):
    monkeypatch.setenv("WHISPER_SERVER_REQUIRE_AUTH", "true")
    monkeypatch.setenv("WHISPER_SERVER_API_KEY", "server-secret")

    from src.whisper_service.server import create_app

    app = create_app()
    client = app.test_client()

    response = client.get(
        "/api/v1/capabilities",
        headers={"X-API-Key": "server-secret"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True


def test_transcribe_rejects_without_api_key(monkeypatch):
    monkeypatch.setenv("WHISPER_SERVER_REQUIRE_AUTH", "true")
    monkeypatch.setenv("WHISPER_SERVER_API_KEY", "server-secret")

    from src.whisper_service.server import create_app

    app = create_app()
    client = app.test_client()

    data = {
        "audio_file": (io.BytesIO(b"fake-audio"), "sample.wav"),
        "model": "base",
        "device": "cpu",
    }
    response = client.post("/api/v1/transcribe-file", data=data, content_type="multipart/form-data")

    assert response.status_code == 401


def test_transcribe_accepts_bearer_token_header(monkeypatch):
    monkeypatch.setenv("WHISPER_SERVER_REQUIRE_AUTH", "true")
    monkeypatch.setenv("WHISPER_SERVER_API_KEY", "server-secret")

    from src.whisper_service import server

    class DummyTranscriber:
        device = "cpu"

        def transcribe(self, audio_path, language, word_timestamps):
            return {"text": "ok", "segments": []}

    monkeypatch.setattr(server, "get_transcriber", lambda model_name, requested_device, engine="openai-whisper": DummyTranscriber())

    app = server.create_app()
    client = app.test_client()

    data = {
        "audio_file": (io.BytesIO(b"fake-audio"), "sample.wav"),
        "model": "base",
        "device": "cpu",
    }
    response = client.post(
        "/api/v1/transcribe-file",
        data=data,
        content_type="multipart/form-data",
        headers={"Authorization": "Bearer server-secret"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["transcription"]["text"] == "ok"


@pytest.fixture()
def app():
    with patch("src.whisper_service.server.detect_available_devices") as mock_detect:
        mock_detect.return_value = (["auto", "cpu"], False)
        from src.whisper_service import server
        import importlib
        importlib.reload(server)
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
            data={
                "model": "base",
                "device": "cpu",
                "engine": "invalid-engine",
                "audio_file": (audio.open("rb"), "test.wav"),
            },
            content_type="multipart/form-data",
        )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "engine" in data["error"]["message"].lower()
