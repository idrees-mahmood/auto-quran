FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Optional: pass a CUDA wheel index to enable GPU acceleration in the container.
# Example: --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_INDEX_URL=""

RUN python -m pip install --upgrade pip
RUN if [ -n "$TORCH_INDEX_URL" ]; then \
        python -m pip install --index-url "$TORCH_INDEX_URL" torch; \
    else \
        python -m pip install torch; \
    fi

COPY src/whisper_service/requirements.txt /app/src/whisper_service/requirements.txt
RUN python -m pip install -r /app/src/whisper_service/requirements.txt

COPY src /app/src

ENV PYTHONPATH=/app
EXPOSE 8001

CMD ["python", "-m", "src.whisper_service.server"]
