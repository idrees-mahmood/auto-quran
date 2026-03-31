# AI Quran Video Composer - Windows Setup
# Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1

$ErrorActionPreference = "Stop"

Write-Host "AI Quran Video Composer - Windows Setup"
Write-Host "========================================="
Write-Host ""

# ---- Python ----
Write-Host "Checking Python..."
$python = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -eq 3 -and $minor -ge 10) {
                $python = $cmd
                Write-Host "  Found: $ver"
                break
            }
        }
    } catch {}
}
if (-not $python) {
    Write-Host ""
    Write-Host "Python 3.10 or higher is required."
    Write-Host "Download from: https://www.python.org/downloads/"
    Write-Host "  - Check 'Add Python to PATH' during installation."
    Write-Host ""
    Write-Host "Or install via winget:"
    Write-Host "  winget install Python.Python.3.11"
    exit 1
}

# ---- FFmpeg ----
Write-Host "Checking FFmpeg..."
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    $ffver = (ffmpeg -version 2>&1 | Select-Object -First 1)
    Write-Host "  Found: $ffver"
} else {
    Write-Host ""
    Write-Host "FFmpeg is not installed."
    Write-Host "Install options:"
    Write-Host "  winget install Gyan.FFmpeg"
    Write-Host "  choco install ffmpeg"
    Write-Host "  Or download from: https://ffmpeg.org/download.html"
    Write-Host ""
    Write-Host "After installing, restart this terminal and run this script again."
    exit 1
}

# ---- Google Chrome ----
Write-Host "Checking Google Chrome..."
$chromePaths = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
)
$chromeFound = $chromePaths | Where-Object { Test-Path $_ }
if ($chromeFound) {
    Write-Host "  Found: $($chromeFound[0])"
} else {
    Write-Host "  Google Chrome not found. Required for Arabic text rendering."
    Write-Host "  Download from: https://www.google.com/chrome/"
    Write-Host "  Or: winget install Google.Chrome"
    Write-Host ""
    Write-Host "  Install Chrome and re-run this script, or continue and install it later."
}

# ---- uv (preferred package manager) ----
Write-Host ""
Write-Host "Checking for uv..."
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "  uv found."
    $useUv = $true
} else {
    Write-Host "  uv not found - installing..."
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        # Refresh PATH
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("PATH", "User")
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            Write-Host "  uv installed."
            $useUv = $true
        } else {
            Write-Host "  uv installed but not on PATH yet - will fall back to venv."
            $useUv = $false
        }
    } catch {
        Write-Host "  Could not install uv automatically. Falling back to venv + pip."
        $useUv = $false
    }
}

# ---- Install dependencies ----
Write-Host ""
if ($useUv -and (Test-Path "pyproject.toml")) {
    Write-Host "Installing dependencies with uv..."
    uv sync --extra dev
    Write-Host ""
    Write-Host "Setup complete."
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  .venv\Scripts\activate     # activate the virtual environment"
    Write-Host "  streamlit run app.py        # start the web UI"
    Write-Host ""
} else {
    Write-Host "Installing dependencies with pip..."
    & $python -m venv venv
    & .\venv\Scripts\python.exe -m pip install --upgrade pip
    & .\venv\Scripts\pip.exe install -r requirements.txt
    Write-Host ""
    Write-Host "Setup complete."
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  .\venv\Scripts\activate    # activate the virtual environment"
    Write-Host "  streamlit run app.py        # start the web UI"
    Write-Host ""
}

Write-Host "For GPU transcription on a separate machine, see docs/REMOTE_WHISPER_CLOUDFLARE.md"
Write-Host ""
