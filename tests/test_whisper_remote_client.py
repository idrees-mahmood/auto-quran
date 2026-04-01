import requests

from src.whisper_remote_client import (
    DEFAULT_WHISPER_MODELS,
    fetch_whisper_capabilities,
    transcribe_audio_via_remote,
)


class DummyResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_fetch_whisper_capabilities_success(monkeypatch):
    def fake_get(url, timeout, headers):
        assert url == "http://localhost:8001/api/v1/capabilities"
        assert timeout == 3
        assert isinstance(headers, dict)
        return DummyResponse(
            {
                "success": True,
                "capabilities": {
                    "models": ["base", "turbo"],
                    "devices": ["auto", "cpu", "cuda"],
                    "gpu_available": True,
                    "engines": ["openai-whisper", "whisperx"],
                },
            }
        )

    monkeypatch.setattr("src.whisper_remote_client.requests.get", fake_get)

    capabilities = fetch_whisper_capabilities("http://localhost:8001", timeout_seconds=3)

    assert capabilities["available"] is True
    assert capabilities["models"] == ["base", "turbo"]
    assert capabilities["devices"] == ["auto", "cpu", "cuda"]
    assert capabilities["gpu_available"] is True
    assert capabilities["engines"] == ["openai-whisper", "whisperx"]


def test_fetch_whisper_capabilities_returns_fallback_on_error(monkeypatch):
    def fake_get(url, timeout, headers):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr("src.whisper_remote_client.requests.get", fake_get)

    capabilities = fetch_whisper_capabilities("http://localhost:8001", timeout_seconds=1)

    assert capabilities["available"] is False
    assert capabilities["models"] == DEFAULT_WHISPER_MODELS
    assert capabilities["devices"] == ["auto", "cpu"]
    assert "connection refused" in capabilities["error"]


def test_transcribe_audio_via_remote_posts_file_and_returns_transcription(monkeypatch, tmp_path):
    monkeypatch.delenv("WHISPER_REMOTE_API_KEY", raising=False)
    monkeypatch.delenv("WHISPER_CF_ACCESS_CLIENT_ID", raising=False)
    monkeypatch.delenv("WHISPER_CF_ACCESS_CLIENT_SECRET", raising=False)
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake-audio-bytes")

    captured = {}

    def fake_post(url, files, data, headers, timeout):
        captured["url"] = url
        captured["timeout"] = timeout
        captured["data"] = dict(data)
        captured["filename"] = files["audio_file"][0]
        captured["headers"] = dict(headers)
        return DummyResponse(
            {
                "success": True,
                "transcription": {
                    "text": "test transcription",
                    "segments": [],
                },
            }
        )

    monkeypatch.setattr("src.whisper_remote_client.requests.post", fake_post)

    result = transcribe_audio_via_remote(
        base_url="http://whisper.local:9000",
        audio_path=str(audio_path),
        model_name="turbo",
        device="cuda",
        timeout_seconds=45,
    )

    assert result["text"] == "test transcription"
    assert captured["url"] == "http://whisper.local:9000/api/v1/transcribe-file"
    assert captured["timeout"] == 45
    assert captured["filename"] == "sample.wav"
    assert captured["data"]["model"] == "turbo"
    assert captured["data"]["device"] == "cuda"
    assert captured["data"]["language"] == "ar"
    assert captured["data"]["word_timestamps"] == "true"
    assert captured["data"]["engine"] == "openai-whisper"
    assert captured["headers"] == {}


def test_transcribe_audio_via_remote_sends_auth_headers(monkeypatch, tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake-audio-bytes")

    captured = {}

    def fake_post(url, files, data, headers, timeout):
        captured["headers"] = dict(headers)
        return DummyResponse(
            {
                "success": True,
                "transcription": {
                    "text": "ok",
                    "segments": [],
                },
            }
        )

    monkeypatch.setattr("src.whisper_remote_client.requests.post", fake_post)

    transcribe_audio_via_remote(
        base_url="http://whisper.local:9000",
        audio_path=str(audio_path),
        model_name="turbo",
        device="cuda",
        api_key="test-api-key",
        cf_access_client_id="cf-client-id",
        cf_access_client_secret="cf-client-secret",
    )

    assert captured["headers"]["Authorization"] == "Bearer test-api-key"
    assert captured["headers"]["X-API-Key"] == "test-api-key"
    assert captured["headers"]["CF-Access-Client-Id"] == "cf-client-id"
    assert captured["headers"]["CF-Access-Client-Secret"] == "cf-client-secret"


def test_fetch_whisper_capabilities_reads_auth_from_env(monkeypatch):
    captured = {}

    def fake_get(url, timeout, headers):
        captured["headers"] = dict(headers)
        return DummyResponse(
            {
                "success": True,
                "capabilities": {
                    "models": ["base"],
                    "devices": ["auto", "cpu"],
                    "gpu_available": False,
                },
            }
        )

    monkeypatch.setenv("WHISPER_REMOTE_API_KEY", "env-api-key")
    monkeypatch.setenv("WHISPER_CF_ACCESS_CLIENT_ID", "env-cf-id")
    monkeypatch.setenv("WHISPER_CF_ACCESS_CLIENT_SECRET", "env-cf-secret")
    monkeypatch.setattr("src.whisper_remote_client.requests.get", fake_get)

    fetch_whisper_capabilities("http://localhost:8001", timeout_seconds=3)

    assert captured["headers"]["Authorization"] == "Bearer env-api-key"
    assert captured["headers"]["CF-Access-Client-Id"] == "env-cf-id"
    assert captured["headers"]["CF-Access-Client-Secret"] == "env-cf-secret"


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

    from src.whisper_remote_client import transcribe_audio_via_remote
    transcribe_audio_via_remote(
        base_url="http://whisper.local:9000",
        audio_path=str(audio_path),
        model_name="turbo",
        device="cuda",
        engine="whisperx",
    )

    assert captured["data"]["engine"] == "whisperx"
