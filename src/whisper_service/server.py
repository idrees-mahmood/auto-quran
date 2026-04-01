"""Container-friendly Whisper transcription service."""

from __future__ import annotations

import importlib.util
import logging
import os
import tempfile
import threading
import time
from typing import Dict, Optional, Tuple

from src.audio_processing_utils import WhisperTranscriber


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "turbo"]
ALLOWED_ENGINES = {"openai-whisper", "whisperx"}

_TRANSCRIBER_CACHE: Dict[Tuple[str, str, str], WhisperTranscriber] = {}
_CACHE_LOCK = threading.Lock()


def _load_environment() -> None:
    """Load .env for local runs while remaining no-op if python-dotenv is absent."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    env_file = os.getenv("WHISPER_ENV_FILE", "").strip()
    if env_file:
        load_dotenv(dotenv_path=env_file, override=False)
    else:
        load_dotenv(override=False)


_load_environment()


def _parse_bool(raw_value: Optional[str], default: bool = True) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _extract_api_key(request) -> str:
    header_api_key = (request.headers.get("X-API-Key") or "").strip()
    if header_api_key:
        return header_api_key

    auth_header = (request.headers.get("Authorization") or "").strip()
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()

    return ""


def _check_auth(request):
    require_auth = _parse_bool(os.getenv("WHISPER_SERVER_REQUIRE_AUTH"), default=False)
    configured_api_key = (os.getenv("WHISPER_SERVER_API_KEY") or "").strip()

    if configured_api_key and not require_auth:
        # If an API key is configured, require it by default.
        require_auth = True

    if not require_auth:
        return None

    if not configured_api_key:
        return {
            "success": False,
            "error": {
                "message": "Unauthorized: server auth is enabled but WHISPER_SERVER_API_KEY is not configured."
            },
        }, 401

    provided_api_key = _extract_api_key(request)
    if provided_api_key != configured_api_key:
        return {
            "success": False,
            "error": {"message": "Unauthorized: invalid or missing API key."},
        }, 401

    return None


def detect_available_devices() -> Tuple[list[str], bool]:
    """Return available processing devices and whether GPU acceleration is available."""
    devices = ["auto", "cpu"]
    gpu_available = False

    try:
        import torch

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devices.append("mps")
            gpu_available = True

        if torch.cuda.is_available():
            devices.append("cuda")
            gpu_available = True
    except Exception:
        pass

    return devices, gpu_available


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


def create_app():
    try:
        from flask import Flask, jsonify, request
    except ImportError as exc:
        raise ImportError(
            "Flask is required for whisper_service. Install dependencies from "
            "src/whisper_service/requirements.txt or run uv sync."
        ) from exc

    app = Flask(__name__)

    @app.route("/api/v1/health", methods=["GET"])
    def health():
        devices, gpu_available = detect_available_devices()
        return jsonify(
            {
                "status": "healthy",
                "service": "whisper-service",
                "models": WHISPER_MODELS,
                "devices": devices,
                "gpu_available": gpu_available,
            }
        )

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

    @app.route("/api/v1/transcribe-file", methods=["POST"])
    def transcribe_file():
        auth_error = _check_auth(request)
        if auth_error is not None:
            payload, status = auth_error
            return jsonify(payload), status

        audio_file = request.files.get("audio_file")
        if audio_file is None:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": {"message": "Missing required multipart file field: audio_file"},
                    }
                ),
                400,
            )

        model_name = request.form.get("model", "base")
        requested_device = request.form.get("device", "auto")
        engine = request.form.get("engine", "openai-whisper")
        language = request.form.get("language", "ar")
        word_timestamps = _parse_bool(request.form.get("word_timestamps"), default=True)

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

        if model_name not in WHISPER_MODELS:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": {
                            "message": f"Unsupported model '{model_name}'. Allowed: {', '.join(WHISPER_MODELS)}"
                        },
                    }
                ),
                400,
            )

        temp_file_path = None
        started_at = time.time()

        try:
            original_name = audio_file.filename or "uploaded_audio.wav"
            suffix = os.path.splitext(original_name)[1] or ".wav"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                audio_file.save(temp_file)
                temp_file_path = temp_file.name

            transcriber = get_transcriber(model_name, requested_device, engine)
            transcription = transcriber.transcribe(
                temp_file_path,
                language=language,
                word_timestamps=word_timestamps,
            )

            return jsonify(
                {
                    "success": True,
                    "transcription": transcription,
                    "metadata": {
                        "model": model_name,
                        "device_requested": requested_device,
                        "device_used": transcriber.device,
                        "processing_time_seconds": round(time.time() - started_at, 3),
                    },
                }
            )

        except Exception as exc:
            logger.exception("Remote whisper transcription failed")
            return (
                jsonify(
                    {
                        "success": False,
                        "error": {"message": str(exc)},
                    }
                ),
                500,
            )

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    return app


def main():
    host = os.getenv("WHISPER_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("WHISPER_SERVER_PORT", "8001"))
    debug = os.getenv("WHISPER_SERVER_DEBUG", "false").lower() == "true"

    app = create_app()
    logger.info("Starting whisper service on %s:%s", host, port)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
