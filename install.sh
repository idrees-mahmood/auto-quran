#!/bin/bash

# AI Quran Video Composer - Installation Script (macOS / Linux)
# On Windows use: scripts/setup_windows.ps1

set -e  # Exit on any error

echo "AI Quran Video Composer — Installation"
echo "======================================="

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version is compatible"
else
    echo "❌ Python $python_version is not compatible. Please install Python 3.10 or higher."
    exit 1
fi

# Check FFmpeg
echo "📋 Checking FFmpeg installation..."
if command -v ffmpeg &> /dev/null; then
    ffmpeg_version=$(ffmpeg -version 2>&1 | head -n1 | awk '{print $3}')
    echo "✅ FFmpeg $ffmpeg_version is installed"
else
    echo "  FFmpeg is not installed."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  Install with: brew install ffmpeg"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "  Install with: sudo apt update && sudo apt install ffmpeg"
    fi
    echo "  Then run this script again."
    exit 1
fi

# Attempt uv sync first, then gracefully fall back to pip if unavailable/fails.
echo "⚡ Attempting uv sync first..."
if command -v uv &> /dev/null && [ -f "pyproject.toml" ]; then
    set +e
    uv sync --extra dev
    uv_exit_code=$?
    set -e

    if [ "$uv_exit_code" -eq 0 ]; then
        echo "  uv sync succeeded."

        if [ -f ".venv/bin/activate" ]; then
            source .venv/bin/activate
        fi

        echo ""
        echo "Installation complete."
        echo ""
        echo "Next steps:"
        echo "  ./launch_ui.sh          # start the web UI"
        echo "  uv run pytest           # run tests"
        echo ""
        exit 0
    else
        echo "⚠️ uv sync failed (exit code: $uv_exit_code). Falling back to venv + pip..."
    fi
else
    echo "ℹ️ uv not available (or pyproject.toml missing). Falling back to venv + pip..."
fi

# Create virtual environment
echo "🐍 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Installation complete."
echo ""
echo "Next steps:"
echo "  source venv/bin/activate   # activate the virtual environment"
echo "  ./launch_ui.sh             # start the web UI"
echo ""
echo "For GPU transcription on a separate machine, see docs/REMOTE_WHISPER_CLOUDFLARE.md"
echo ""
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Linux note: Google Chrome is required for text rendering."
    echo "  Ubuntu/Debian: sudo apt install google-chrome-stable"
    echo ""
fi