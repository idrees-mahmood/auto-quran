# AI Quran Video Composer

Generate Quran recitation videos with word-level synchronized text overlays and AI-selected background videos.

## Example Videos

- [Surah An-Nisa (4:134)](./examples/4-134-134-MAHMOUD_KHALIL_AL_HUSARY.mp4) - Mahmoud Khalil Al-Husary
- [Surah Al-Hijr (15:2-5)](./examples/15-2-5-MUHAMMAD_AL_MINSHAWI.mp4) - Muhammad Al-Minshawi
- [Surah Taha (20:124-126)](./examples/20-124-126-MAHMOUD_KHALIL_AL_HUSARY.mp4) - Mahmoud Khalil Al-Husary

---

## Features

- **Word-level synchronization** — text appears in sync with recitation, including repeated ayahs
- **AI background selection** — contextually appropriate video backgrounds via OpenAI + Pexels
- **Pre-processed reciters** — Mahmoud Khalil Al-Husary, Muhammad Al-Minshawi
- **Custom audio support** — transcribe any recitation with local or remote Whisper
- **DTW alignment** — handles repetitions, noise, and partial ayahs robustly
- **Test mode** — generate videos with no API costs

---

## Quick Start

### macOS

```bash
git clone <repo-url>
cd auto-quran
./setup_macos.sh
./launch_ui.sh
```

`setup_macos.sh` installs Homebrew dependencies, sets up Python with MPS acceleration on Apple Silicon, and prefers `uv` when available.

### Linux

```bash
git clone <repo-url>
cd auto-quran
./install.sh
./launch_ui.sh
```

See [Linux prerequisites](#linux-prerequisites) below if FFmpeg or Chrome are missing.

### Windows

```powershell
git clone <repo-url>
cd auto-quran
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
.venv\Scripts\activate
streamlit run app.py
```

See [scripts/setup_windows.ps1](scripts/setup_windows.ps1) for what it installs and manual fallback steps.

---

## Prerequisites

| Requirement | macOS | Linux | Windows |
|---|---|---|---|
| Python 3.10–3.12 | `brew install python@3.11` | `apt install python3` | [python.org](https://www.python.org/downloads/) |
| FFmpeg | auto-installed by `setup_macos.sh` | `apt install ffmpeg` | `winget install ffmpeg` |
| Google Chrome | prompt in `setup_macos.sh` | `apt install google-chrome-stable` | [google.com/chrome](https://www.google.com/chrome/) |

### Linux prerequisites

Chrome is required for Arabic text rendering. On Ubuntu/Debian:

```bash
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list
sudo apt update && sudo apt install -y google-chrome-stable
```

For NVIDIA GPU acceleration on Linux, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and use the GPU Docker image (see Remote Whisper below).

---

## Usage

### Web UI

```bash
./launch_ui.sh          # macOS / Linux
.venv\Scripts\activate  # Windows — then:
streamlit run app.py
```

Opens at **http://localhost:8501**.

1. **Video Generation** — pick surah, ayah range, reciter → Generate
2. **Custom Audio** — upload any recitation → Transcribe (local or remote Whisper) → Generate
3. **Test Mode** — black background, no API calls needed

See [docs/UI_GUIDE.md](docs/UI_GUIDE.md) for full instructions.

### Remote Whisper Service (GPU Server)

Run transcription on a separate GPU machine so your dev machine doesn't need to load Whisper.

**On the GPU server** (e.g. RTX 3090 box):

```bash
# CPU-only
docker build -t auto-quran-whisper -f src/whisper_service/Dockerfile .
docker run --rm -p 8001:8001 auto-quran-whisper

# NVIDIA GPU
docker build \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  -t auto-quran-whisper:gpu \
  -f src/whisper_service/Dockerfile .
docker run --rm --gpus all -p 8001:8001 auto-quran-whisper:gpu
```

**In the UI** (on your dev machine):

1. Custom Audio → Transcribe
2. Backend: **Remote Whisper Server**
3. Server URL: `http://<server-ip>:8001`
4. Click **Check Server Capabilities** — models and devices populate automatically

### Team Setup: Secure Remote Access (Cloudflare Tunnel)

For a shared server accessible over the internet without opening inbound ports:

```bash
cp config/whisper_remote/env/whisper-service.env.example config/whisper_remote/env/whisper-service.env
cp config/whisper_remote/env/cloudflared.env.example   config/whisper_remote/env/cloudflared.env
cp config/whisper_remote/env/ui.remote.env.example     .env
# Edit each file and fill in tokens/keys

docker compose -f config/whisper_remote/docker-compose.cloudflare.yml up -d --build
```

In the UI, select **Remote Whisper Server** → choose `Test` or `Production` environment.

See [docs/REMOTE_WHISPER_CLOUDFLARE.md](docs/REMOTE_WHISPER_CLOUDFLARE.md) for full setup steps.

### Jupyter Notebooks (Advanced)

```bash
uv run jupyter notebook notebooks/video_gen.ipynb
uv run jupyter notebook notebooks/audio_processing.ipynb
```

---

## Configuration

### API Keys (Production Mode)

| Service | Purpose | Required |
|---|---|---|
| OpenAI | AI background suggestions | Production only |
| Pexels | Background video downloads | Production only |

Set in the UI sidebar. Not needed for Test Mode.

### Test Mode

No API keys needed — generates videos with black backgrounds:
- UI: toggle **Test Mode**
- Notebooks: `TEST_MODE = True`

---

## Project Structure

```
app.py                         # Streamlit web UI
src/
  alignment_utils.py           # Ayah detection & alignment
  audio_processing_utils.py    # Whisper transcription
  dtw_alignment.py             # DTW-based alignment engine
  utils.py                     # Video composition
  whisper_service/             # Containerised Whisper REST API
    server.py
    Dockerfile
config/
  whisper_remote/              # Remote Whisper deployment configs
scripts/
  setup_windows.ps1            # Windows setup
data/
  quran/                       # Quran text & translations
  audio/                       # Reciter timestamp data (Tarteel format)
  fonts/                       # Arabic fonts
docs/                          # Documentation
examples/                      # Sample output videos
notebooks/                     # Jupyter notebooks
tests/                         # Test suite
```

---

## Dependency Management

`pyproject.toml` is the source of truth. `uv` is recommended; `pip` works as a fallback.

```bash
uv sync --extra dev          # install / sync
uv run streamlit run app.py  # run UI
uv run pytest                # run tests
source .venv/bin/activate    # activate env in current shell (macOS/Linux)
.venv\Scripts\activate       # activate env in current shell (Windows)
```

---

## Apple Silicon

On M1/M2/M3, Whisper uses Metal (MPS) acceleration automatically — 2–3x faster than CPU.

---

## Notes

- Review generated videos for Islamic compliance before publishing
- API costs apply in Production Mode (~$0.10–0.50/video)
- Verify Pexels background content for appropriateness

---

## Documentation

- [docs/UI_GUIDE.md](docs/UI_GUIDE.md) — web UI guide
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — developer reference
- [docs/REMOTE_WHISPER_CLOUDFLARE.md](docs/REMOTE_WHISPER_CLOUDFLARE.md) — secure remote Whisper setup
- [docs/TRANSCRIPTION_API.md](docs/TRANSCRIPTION_API.md) — API reference
- [docs/REGRESSION_TESTS.md](docs/REGRESSION_TESTS.md) — testing framework

---

## License

[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0](LICENSE-CC-BY-NC-ND-4.0.md)

---

## Acknowledgments

- [Tarteel.ai](https://qul.tarteel.ai/) — Quran audio timing data
- [Pexels](https://www.pexels.com/) — background videos
- [OpenAI](https://openai.com/) — Whisper & GPT

---

**Made with love for the Muslim community**

_"And We have certainly made the Quran easy for remembrance, so is there any who will remember?"_ — Quran 54:17
