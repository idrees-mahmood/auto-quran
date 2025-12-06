#!/bin/bash

# Quran Video Generator - Launch Script
# This script starts the Streamlit UI

set -e

echo "🚀 Starting Quran Video Generator..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run setup:"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  ./setup_macos.sh"
    else
        echo "  python3 -m venv venv"
        echo "  source venv/bin/activate"
        echo "  pip install -r requirements.txt"
    fi
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if Streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "❌ Streamlit not installed!"
    echo "Installing Streamlit..."
    pip install streamlit>=1.29.0
fi

# Launch Streamlit app
echo "🌐 Launching UI..."
echo ""
echo "The application will open in your browser at:"
echo "  👉 http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

streamlit run app.py
