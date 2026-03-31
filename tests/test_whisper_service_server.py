import io


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

    monkeypatch.setattr(server, "get_transcriber", lambda model_name, requested_device: DummyTranscriber())

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
