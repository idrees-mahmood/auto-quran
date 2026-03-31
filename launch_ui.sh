#!/bin/bash

# AI Quran Video Composer — Launch UI

set -e

# Prefer uv when available
if command -v uv &> /dev/null && [ -f "pyproject.toml" ]; then
    uv sync --quiet
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    echo "http://localhost:8501  (Ctrl+C to stop)"
    streamlit run app.py
    exit 0
fi

# Fall back to .venv (created by uv even when uv isn't on PATH)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "http://localhost:8501  (Ctrl+C to stop)"
    streamlit run app.py
    exit 0
fi

# Fall back to legacy venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "http://localhost:8501  (Ctrl+C to stop)"
    streamlit run app.py
    exit 0
fi

echo "No virtual environment found. Run setup first:"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  ./setup_macos.sh"
else
    echo "  ./install.sh"
fi
exit 1
