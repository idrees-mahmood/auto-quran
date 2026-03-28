#!/bin/bash

# macOS Setup Script for AI Quran Video Composer
# This script optimizes the installation for Apple Silicon (M1/M2/M3) Macs

set -e  # Exit on error

echo "🍎 AI Quran Video Composer - macOS Setup"
echo "========================================"
echo ""

# Detect macOS and chip
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  This script is designed for macOS"
    exit 1
fi

# Check for Apple Silicon
CHIP=$(uname -m)
if [[ "$CHIP" == "arm64" ]]; then
    echo "✓ Apple Silicon detected ($CHIP)"
    echo "  MPS (Metal) acceleration will be available for Whisper"
    USE_MPS=true
else
    echo "✓ Intel Mac detected ($CHIP)"
    USE_MPS=false
fi

echo ""
echo "Checking prerequisites..."
echo ""

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found"
    echo "   Install from: https://brew.sh"
    echo "   Then run this script again"
    exit 1
else
    echo "✓ Homebrew installed"
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    echo "   Install with: brew install python@3.11"
    exit 1
else
    PYTHON_VERSION=$(python3 --version)
    echo "✓ $PYTHON_VERSION"
fi

# Check for FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  FFmpeg not found - installing..."
    brew install ffmpeg
    echo "✓ FFmpeg installed"
else
    echo "✓ FFmpeg installed"
fi

# Check for Google Chrome
if [ -d "/Applications/Google Chrome.app" ]; then
    echo "✓ Google Chrome installed"
else
    echo "⚠️  Google Chrome not found"
    echo "   Required for text rendering"
    echo "   Download from: https://www.google.com/chrome/"
    echo "   Or install with: brew install --cask google-chrome"
    read -p "   Install now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        brew install --cask google-chrome
        echo "✓ Google Chrome installed"
    fi
fi

# Prefer uv when available
if command -v uv &> /dev/null && [ -f "pyproject.toml" ]; then
    echo ""
    echo "⚡ Attempting uv sync first..."
    set +e
    uv sync --extra dev
    uv_exit_code=$?
    set -e

    if [ "$uv_exit_code" -eq 0 ]; then
        echo "✅ uv sync succeeded."

        # Activate uv virtual environment for this script session.
        if [ -f ".venv/bin/activate" ]; then
            source .venv/bin/activate
        fi

        if [[ $USE_MPS == true ]]; then
            echo ""
            echo "Ensuring PyTorch with Metal (MPS) support..."
            uv pip install torch torchvision torchaudio
            echo "✓ PyTorch with MPS support installed"
        fi

        echo ""
        echo "✅ Setup complete with uv!"
        echo ""
        echo "📝 Next steps:"
        echo ""
        echo "1. Activate in current shell (optional):"
        echo "   source .venv/bin/activate"
        echo ""
        echo "2. Launch UI:"
        echo "   streamlit run app.py"
        echo ""
        echo "3. For video generation notebook:"
        echo "   jupyter notebook video_gen.ipynb"
        echo ""
        echo "4. For custom audio processing notebook:"
        echo "   jupyter notebook audio_processing.ipynb"
        echo ""
        echo "🚀 Happy video generating!"
        exit 0
    else
        echo "⚠️ uv sync failed (exit code: $uv_exit_code). Falling back to venv + pip..."
    fi
fi

echo ""
echo "Installing Python dependencies..."
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install base requirements
echo "Installing base dependencies..."
pip install -r requirements.txt

# Install PyTorch with MPS support for Apple Silicon
if [[ $USE_MPS == true ]]; then
    echo ""
    echo "Installing PyTorch with Metal (MPS) support..."
    pip install torch torchvision torchaudio
    echo "✓ PyTorch with MPS support installed"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. For video generation with pre-processed audio:"
echo "   jupyter notebook video_gen.ipynb"
echo ""
echo "3. For custom audio processing:"
echo "   jupyter notebook audio_processing.ipynb"
echo ""
echo "💡 Tip: if you install uv, you can use"
echo "   uv sync"
echo "   uv run streamlit run app.py"
echo ""
echo "💡 Tips for macOS:"
echo "   - TEST_MODE is enabled by default in video_gen.ipynb"
echo "   - This uses black backgrounds to save API credits"
echo "   - Perfect for testing word timing before making API calls"
if [[ $USE_MPS == true ]]; then
    echo "   - Whisper will use MPS acceleration (2-3x faster than CPU)"
    echo "   - Set DEVICE='auto' in audio_processing.ipynb (default)"
fi
echo ""
echo "🚀 Happy video generating!"
