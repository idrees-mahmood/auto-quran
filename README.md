# AI Quran Video Composer

Generate beautiful Quran recitation videos with word-level synchronized text overlays and AI-selected background videos.

## 🎬 Example Videos

- [Surah An-Nisa (4:134)](./examples/4-134-134-MAHMOUD_KHALIL_AL_HUSARY.mp4) - Mahmoud Khalil Al-Husary
- [Surah Al-Hijr (15:2-5)](./examples/15-2-5-MUHAMMAD_AL_MINSHAWI.mp4) - Muhammad Al-Minshawi
- [Surah Taha (20:124-126)](./examples/20-124-126-MAHMOUD_KHALIL_AL_HUSARY.mp4) - Mahmoud Khalil Al-Husary

---

## ✨ Features

- **Word-level synchronization** - Text appears in sync with recitation
- **AI background selection** - Contextually appropriate video backgrounds
- **Pre-processed reciters** - Mahmoud Khalil Al-Husary, Muhammad Al-Minshawi
- **Custom audio support** - Process any Quran recitation with Whisper AI
- **Test mode** - Preview videos without API costs

---

## 🚀 Quick Start

### macOS (Recommended)

```bash
git clone https://github.com/yourusername/Ai-Quran-Video-Compser.git
cd Ai-Quran-Video-Compser
./setup_macos.sh  # Installs all dependencies
./launch_ui.sh    # Opens web UI at http://localhost:8501
```

`setup_macos.sh` attempts `uv sync --extra dev` first, then falls back to `venv` + `pip` if needed.

### Manual Setup

#### Preferred installer (recommended)

```bash
git clone https://github.com/yourusername/Ai-Quran-Video-Compser.git
cd Ai-Quran-Video-Compser
./install.sh
```

`install.sh` attempts `uv sync --extra dev` first, then falls back to `venv` + `pip` if needed.

#### Prerequisites
- Python 3.10+
- FFmpeg: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux)
- Google Chrome (for text rendering)

#### Installation

Recommended (`uv`):

```bash
git clone https://github.com/yourusername/Ai-Quran-Video-Compser.git
cd Ai-Quran-Video-Compser
uv sync --extra dev
```

Then run apps/commands with `uv run`, for example:

```bash
uv run streamlit run app.py
```

Classic `venv` + `pip`:

```bash
git clone https://github.com/yourusername/Ai-Quran-Video-Compser.git
cd Ai-Quran-Video-Compser
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📖 Usage

### Web UI (Easiest)

```bash
./launch_ui.sh
```

`launch_ui.sh` now prefers `uv` automatically when available.

1. **Video Generation Tab**: Select surah, ayahs, reciter → Generate
2. **Custom Audio Tab**: Upload any recitation → Transcribe → Generate
3. **Test Mode**: Enable to preview without API costs (black background)

See [UI_GUIDE.md](UI_GUIDE.md) for detailed instructions.

### Jupyter Notebooks (Advanced)

```bash
jupyter notebook video_gen.ipynb  # For pre-processed audio
jupyter notebook audio_processing.ipynb  # For custom audio
```

---

## ⚙️ Configuration

### API Keys (for Production Mode)

| Service | Purpose | Get Key |
|---------|---------|---------|
| OpenAI | AI background suggestions | [platform.openai.com](https://platform.openai.com/api-keys) |
| Pexels | Background videos | [pexels.com/api](https://www.pexels.com/api/) |

Configure in UI sidebar or set in notebooks. Not needed for Test Mode.

### Test Mode

Generate videos with black backgrounds - no API keys required:
- UI: Toggle "Test Mode" checkbox
- Notebooks: Set `TEST_MODE = True`

---

## 📁 Project Structure

```
├── app.py                    # Streamlit web UI
├── video_gen.ipynb          # Video generation notebook
├── audio_processing.ipynb   # Custom audio processing
├── utils.py                 # Video composition utilities
├── audio_processing_utils.py # Whisper transcription
├── alignment_utils.py       # Ayah detection & alignment
├── data/
│   ├── quran/              # Quran text & translations
│   ├── audio/              # Reciter timestamp data
│   └── fonts/              # Arabic fonts
├── docs/                   # Documentation
└── examples/               # Sample output videos
```

---

## 🍎 Apple Silicon Optimization

On M1/M2/M3 Macs, transcription is 2-3x faster with MPS acceleration (automatic).

---

## ⚠️ Important Notes

- **Review generated videos** for Islamic compliance before publishing
- **API costs apply** in Production Mode (~$0.10-0.50/video)
- Background videos are from Pexels - verify content appropriateness

---

## 📚 Documentation

- [UI_GUIDE.md](UI_GUIDE.md) - User guide for web interface
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) - Developer documentation
- [docs/TRANSCRIPTION_API.md](docs/TRANSCRIPTION_API.md) - Transcription API reference
- [docs/REGRESSION_TESTS.md](docs/REGRESSION_TESTS.md) - Testing framework

## 🧰 Dependency Management

This repository supports both:
- `uv sync` (recommended)
- `pip install -r requirements.txt` (legacy fallback)

Maintenance policy:
- `pyproject.toml` is the primary source of truth for dependencies
- `requirements.txt` is maintained as a compatibility fallback for `pip` workflows

Useful commands:

```bash
uv sync --extra dev
uv run streamlit run app.py
uv run python regression_tests.py run
uv run pytest
```

Activate the uv environment in your current shell:

```bash
source .venv/bin/activate
```

Optional convenience alias for your shell profile:

```bash
alias uvsync='uv sync --extra dev && source .venv/bin/activate'
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional reciter support
- More language translations
- Enhanced video effects
- Performance optimizations

---

## 📄 License

[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0](LICENSE-CC-BY-NC-ND-4.0.md)

For commercial use, contact project maintainers.

---

## 🙏 Acknowledgments

- [Tarteel.ai](https://qul.tarteel.ai/) - Quran audio timing data
- [Pexels](https://www.pexels.com/) - Background videos
- [OpenAI](https://openai.com/) - Whisper & GPT

---

**Made with ❤️ for the Muslim community**

_"And We have certainly made the Quran easy for remembrance, so is there any who will remember?"_ - Quran 54:17
