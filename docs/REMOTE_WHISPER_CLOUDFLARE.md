# Remote Whisper Setup

The Whisper service runs on the GPU server as a Docker container and exposes a REST API on port 8001. There are two ways for teammates to reach it:

- **Direct connection** — simplest, works on the same LAN or VPN
- **Cloudflare Tunnel** — for access over the internet without opening firewall ports

---

## Option A: Direct Connection (LAN / VPN)

Use this when all devs are on the same network or VPN as the GPU server.

**1. On the GPU server**, start the service:

```bash
# CPU-only
docker build -t auto-quran-whisper -f src/whisper_service/Dockerfile .
docker run -d --restart unless-stopped -p 8001:8001 auto-quran-whisper

# NVIDIA GPU (RTX 3090 etc.)
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
docker build --build-arg TORCH_INDEX_URL=$TORCH_INDEX_URL \
  -t auto-quran-whisper:gpu -f src/whisper_service/Dockerfile .
docker run -d --restart unless-stopped --gpus all -p 8001:8001 auto-quran-whisper:gpu
```

**2. Optionally set an API key** by adding `-e WHISPER_SERVER_API_KEY=<secret>` to the `docker run` command.

**3. On each dev machine**, add to `.env` in the project root:

```bash
WHISPER_SERVER_URL_LOCAL=http://<server-ip>:8001
WHISPER_REMOTE_API_KEY=<secret>   # only if you set one above
```

**4. In the UI**: Custom Audio → Transcribe → Backend: **Remote Whisper Server** → URL: `http://<server-ip>:8001` → **Check Server Capabilities**.

---

## Option B: Cloudflare Tunnel + Access (Internet Access)

Use this to expose the server securely over the internet without opening inbound firewall ports.

### Prerequisites

- Docker Desktop (Windows/Mac/Linux) or Docker Engine (Linux) on the GPU server
- Cloudflare account with Zero Trust enabled and your domain's DNS managed by Cloudflare
- `cloudflared` CLI — one-time tunnel creation can be done from any machine

### 1. Install cloudflared on the GPU server (Windows)

```powershell
winget install Cloudflare.cloudflared
```

Or download the MSI from https://github.com/cloudflare/cloudflared/releases and install it.

### 2. Create the Tunnel and DNS Route

Run these once from the GPU server (PowerShell):

```powershell
cloudflared tunnel login
cloudflared tunnel create auto-quran-whisper
cloudflared tunnel route dns auto-quran-whisper whisper.amplab.co.uk
cloudflared tunnel token auto-quran-whisper
```

Copy the token printed by the last command — you'll need it in step 4.

### 3. Configure Cloudflare Access

In [Cloudflare Zero Trust](https://one.dash.cloudflare.com):

1. **Access → Applications → Add an application → Self-hosted**
2. Application domain: `whisper.amplab.co.uk`
3. Add a policy — allow your team by email domain (`amplab.co.uk`) or specific addresses
4. **Access → Service Auth → Create Service Token** — save the `Client ID` and `Client Secret`

### 4. Configure Environment Files

On the GPU server, copy the templates and fill in values:

**Windows (PowerShell):**
```powershell
Copy-Item config\whisper_remote\env\whisper-service.env.example `
          config\whisper_remote\env\whisper-service.env
Copy-Item config\whisper_remote\env\cloudflared.env.example `
          config\whisper_remote\env\cloudflared.env
Copy-Item config\whisper_remote\env\ui.remote.env.example .env
```

**macOS / Linux:**
```bash
cp config/whisper_remote/env/whisper-service.env.example config/whisper_remote/env/whisper-service.env
cp config/whisper_remote/env/cloudflared.env.example     config/whisper_remote/env/cloudflared.env
cp config/whisper_remote/env/ui.remote.env.example       .env
```

Edit the three files:

**`config/whisper_remote/env/whisper-service.env`**
```
WHISPER_SERVER_API_KEY=<strong-random-secret>
```

**`config/whisper_remote/env/cloudflared.env`**
```
CLOUDFLARE_TUNNEL_TOKEN=<token from step 2>
```

**`.env`** (on each dev machine's copy of the repo)
```
WHISPER_SERVER_URL_TEST=https://whisper.amplab.co.uk
WHISPER_REMOTE_API_KEY=<same secret as above>
WHISPER_CF_ACCESS_CLIENT_ID=<Service Token Client ID>
WHISPER_CF_ACCESS_CLIENT_SECRET=<Service Token Client Secret>
```

### 5. Start the Stack on the GPU Server

**Windows (PowerShell) — NVIDIA GPU:**
```powershell
$env:TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"
docker compose -f config\whisper_remote\docker-compose.cloudflare.yml up -d --build
```

**Windows (PowerShell) — CPU only:**
```powershell
docker compose -f config\whisper_remote\docker-compose.cloudflare.yml up -d --build
```

**Linux — NVIDIA GPU:**
```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  docker compose -f config/whisper_remote/docker-compose.cloudflare.yml up -d --build
```

Docker Desktop must be running. The `--gpus all` flag in `docker run` requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on Linux, or [NVIDIA drivers with WSL2](https://developer.nvidia.com/cuda/wsl) on Windows.

### 6. Verify

From any machine with internet access:

```bash
# Health (no auth required)
curl https://whisper.amplab.co.uk/api/v1/health

# Capabilities (auth required)
curl https://whisper.amplab.co.uk/api/v1/capabilities \
  -H "Authorization: Bearer <WHISPER_SERVER_API_KEY>" \
  -H "CF-Access-Client-Id: <CLIENT_ID>" \
  -H "CF-Access-Client-Secret: <CLIENT_SECRET>"
```

**Windows (PowerShell):**
```powershell
Invoke-RestMethod https://whisper.amplab.co.uk/api/v1/health
```

### 7. Team Integration

For each teammate:

1. Copy `config/whisper_remote/env/ui.remote.env.example` to `.env` in their local repo copy.
2. Fill in `WHISPER_SERVER_URL_TEST`, `WHISPER_REMOTE_API_KEY`, and the CF Access token values.
3. Start the UI: `./launch_ui.sh` (macOS/Linux) or `streamlit run app.py` (Windows).
4. In **Custom Audio Processing → Transcribe**:
   - Backend: `Remote Whisper Server`
   - Environment: `Test`
   - Click `Check Server Capabilities`

The UI sends the auth headers automatically.
