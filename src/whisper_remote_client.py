"""Client utilities for communicating with a remote Whisper transcription server."""

from __future__ import annotations

import mimetypes
import os
from typing import Any, Dict

import requests


DEFAULT_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "turbo"]
DEFAULT_CPU_DEVICES = ["auto", "cpu"]


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _fallback_capabilities(error_message: str) -> Dict[str, Any]:
    return {
        "available": False,
        "models": DEFAULT_WHISPER_MODELS,
        "devices": DEFAULT_CPU_DEVICES,
        "gpu_available": False,
        "error": error_message,
    }


def _build_auth_headers(
    api_key: str | None = None,
    cf_access_client_id: str | None = None,
    cf_access_client_secret: str | None = None,
) -> Dict[str, str]:
    """Build auth headers for remote whisper deployments behind Access or API keys."""
    effective_api_key = (api_key or os.getenv("WHISPER_REMOTE_API_KEY", "")).strip()
    effective_cf_access_client_id = (
        cf_access_client_id or os.getenv("WHISPER_CF_ACCESS_CLIENT_ID", "")
    ).strip()
    effective_cf_access_client_secret = (
        cf_access_client_secret or os.getenv("WHISPER_CF_ACCESS_CLIENT_SECRET", "")
    ).strip()

    headers: Dict[str, str] = {}

    if effective_api_key:
        headers["Authorization"] = f"Bearer {effective_api_key}"
        headers["X-API-Key"] = effective_api_key

    if effective_cf_access_client_id:
        headers["CF-Access-Client-Id"] = effective_cf_access_client_id
    if effective_cf_access_client_secret:
        headers["CF-Access-Client-Secret"] = effective_cf_access_client_secret

    return headers


def fetch_whisper_capabilities(
    base_url: str,
    timeout_seconds: int = 5,
    api_key: str | None = None,
    cf_access_client_id: str | None = None,
    cf_access_client_secret: str | None = None,
) -> Dict[str, Any]:
    """Fetch capabilities from a remote Whisper server."""
    try:
        headers = _build_auth_headers(
            api_key=api_key,
            cf_access_client_id=cf_access_client_id,
            cf_access_client_secret=cf_access_client_secret,
        )
        response = requests.get(
            f"{_normalize_base_url(base_url)}/api/v1/capabilities",
            timeout=timeout_seconds,
            headers=headers,
        )
        response.raise_for_status()
        payload = response.json()

        if isinstance(payload, dict) and payload.get("success"):
            capabilities = payload.get("capabilities", {})
        elif isinstance(payload, dict):
            capabilities = payload
        else:
            raise RuntimeError("Capabilities response was not a JSON object")

        models = capabilities.get("models") or DEFAULT_WHISPER_MODELS
        devices = capabilities.get("devices") or DEFAULT_CPU_DEVICES

        if "auto" not in devices:
            devices = ["auto"] + [d for d in devices if d != "auto"]

        return {
            "available": True,
            "models": models,
            "devices": devices,
            "gpu_available": bool(
                capabilities.get("gpu_available", any(d in devices for d in ("cuda", "mps")))
            ),
        }
    except Exception as exc:
        return _fallback_capabilities(str(exc))


def transcribe_audio_via_remote(
    base_url: str,
    audio_path: str,
    model_name: str,
    device: str,
    language: str = "ar",
    word_timestamps: bool = True,
    timeout_seconds: int = 1800,
    api_key: str | None = None,
    cf_access_client_id: str | None = None,
    cf_access_client_secret: str | None = None,
) -> Dict[str, Any]:
    """Upload an audio file to a remote Whisper server and return transcription payload."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    endpoint = f"{_normalize_base_url(base_url)}/api/v1/transcribe-file"
    content_type = mimetypes.guess_type(audio_path)[0] or "application/octet-stream"
    headers = _build_auth_headers(
        api_key=api_key,
        cf_access_client_id=cf_access_client_id,
        cf_access_client_secret=cf_access_client_secret,
    )

    with open(audio_path, "rb") as audio_file:
        files = {
            "audio_file": (os.path.basename(audio_path), audio_file, content_type),
        }
        data = {
            "model": model_name,
            "device": device,
            "language": language,
            "word_timestamps": str(word_timestamps).lower(),
        }

        response = requests.post(
            endpoint,
            files=files,
            data=data,
            headers=headers,
            timeout=timeout_seconds,
        )

    response.raise_for_status()
    payload = response.json()

    if not payload.get("success"):
        error = payload.get("error", {})
        if isinstance(error, dict):
            message = error.get("message", "Remote transcription failed")
        else:
            message = str(error)
        raise RuntimeError(message)

    transcription = payload.get("transcription")
    if not isinstance(transcription, dict):
        raise RuntimeError("Remote response did not contain a transcription object")

    return transcription
