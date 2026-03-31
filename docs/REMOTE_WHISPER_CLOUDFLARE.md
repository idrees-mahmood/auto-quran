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

- Docker + Docker Compose on the remote host
- Cloudflare account with Zero Trust enabled
- `cloudflared` CLI installed locally for one-time tunnel setup
- DNS zone in Cloudflare (for example `example.com`)

### 1. Create Tunnel and DNS Route

Run these once from any machine that has `cloudflared` and your Cloudflare login:

```bash
cloudflared tunnel login
cloudflared tunnel create auto-quran-whisper-test
cloudflared tunnel route dns auto-quran-whisper-test whisper-test.example.com
cloudflared tunnel token auto-quran-whisper-test
```

Take the token from the last command and store it in `config/whisper_remote/env/cloudflared.env`.

### 2. Configure Cloudflare Access

In Cloudflare Zero Trust:

1. Create a **Self-hosted** Access application for `https://whisper-test.example.com/*`.
2. Add policy for your team identity provider (email/domain/group allow-list).
3. Create a **Service Token** for non-interactive app-to-app access.
4. Save:
   - `CF-Access-Client-Id`
   - `CF-Access-Client-Secret`

### 3. Configure Environment Files

Create runtime env files from templates:

```bash
cp config/whisper_remote/env/whisper-service.env.example config/whisper_remote/env/whisper-service.env
cp config/whisper_remote/env/cloudflared.env.example config/whisper_remote/env/cloudflared.env
cp config/whisper_remote/env/ui.remote.env.example .env
```

Update values:

- `config/whisper_remote/env/whisper-service.env`
  - `WHISPER_SERVER_API_KEY`: strong shared secret
- `config/whisper_remote/env/cloudflared.env`
  - `CLOUDFLARE_TUNNEL_TOKEN`: output from `cloudflared tunnel token ...`
- `.env` (Streamlit app)
  - `WHISPER_SERVER_URL_TEST`: Cloudflare URL
  - `WHISPER_REMOTE_API_KEY`: same as service API key
  - `WHISPER_CF_ACCESS_CLIENT_ID` and `WHISPER_CF_ACCESS_CLIENT_SECRET`: Access service token values

### 4. Start the Remote Stack

On the remote host:

```bash
docker compose -f config/whisper_remote/docker-compose.cloudflare.yml up -d --build
```

Optional GPU build on NVIDIA host:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  docker compose -f config/whisper_remote/docker-compose.cloudflare.yml up -d --build
```

### 5. Verify Health and Auth

From any machine:

```bash
curl -sS https://whisper-test.example.com/api/v1/health
```

Verify protected endpoint:

```bash
curl -sS https://whisper-test.example.com/api/v1/capabilities \
  -H "Authorization: Bearer <WHISPER_SERVER_API_KEY>" \
  -H "CF-Access-Client-Id: <CF_ACCESS_CLIENT_ID>" \
  -H "CF-Access-Client-Secret: <CF_ACCESS_CLIENT_SECRET>"
```

### 6. Team Integration (Test/Prod)

For each teammate running the UI:

1. Copy `config/whisper_remote/env/ui.remote.env.example` to `.env`.
2. Fill in test/prod URLs and secrets.
3. Start UI with `./launch_ui.sh`.
4. In **Custom Audio Processing → Transcribe**:
   - Backend: `Remote Whisper Server`
   - Environment: `Test` or `Production`
   - Click `Check Server Capabilities`.

The UI now supports environment-targeted endpoints and sends required auth headers automatically.
