# -----------------------------------------------------------
# Qwen Agent — One-command bootstrap (Windows)
#
# Usage:
#   irm https://raw.githubusercontent.com/<repo>/setup.ps1 | iex
#   -- or --
#   .\setup.ps1
#
# What this does:
#   1. Installs Ollama if not present
#   2. Starts Ollama server
#   3. Pulls qwen3.5 model
#   4. Sets up Python venv + dependencies
#   5. Seeds memory.json if not present
#   6. Launches the agent
# -----------------------------------------------------------

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $ScriptDir) { $ScriptDir = Get-Location }
$Model = if ($env:MODEL) { $env:MODEL } else { "qwen3.5" }
$VenvDir = Join-Path $ScriptDir ".venv"
$MemoryFile = Join-Path $ScriptDir "memory.json"

function Log($msg)  { Write-Host "[+] $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "[!] $msg" -ForegroundColor Yellow }
function Err($msg)  { Write-Host "[x] $msg" -ForegroundColor Red }
function Step($msg) { Write-Host "`n--- $msg ---" -ForegroundColor Cyan }

# -----------------------------------------------------------
# 1) Install Ollama
# -----------------------------------------------------------
Step "Checking Ollama"

$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollamaCmd) {
    Log "Ollama already installed."
} else {
    Warn "Ollama not found. Installing..."
    try {
        Invoke-RestMethod https://ollama.com/install.ps1 | Invoke-Expression
        Log "Ollama installed."
    } catch {
        Err "Ollama installation failed: $_"
        exit 1
    }
}

# -----------------------------------------------------------
# 2) Start Ollama server
# -----------------------------------------------------------
Step "Starting Ollama server"

$serverUp = $false
try {
    $resp = Invoke-RestMethod http://localhost:11434/api/tags -TimeoutSec 2
    $serverUp = $true
    Log "Ollama server already running."
} catch {}

if (-not $serverUp) {
    Warn "Starting Ollama server..."
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        try {
            $resp = Invoke-RestMethod http://localhost:11434/api/tags -TimeoutSec 2
            $serverUp = $true
            break
        } catch {}
    }
    if ($serverUp) {
        Log "Ollama server started."
    } else {
        Err "Ollama server failed to start within 30s."
        exit 1
    }
}

# -----------------------------------------------------------
# 3) Pull model
# -----------------------------------------------------------
Step "Pulling model: $Model"

$models = (ollama list 2>$null) -join "`n"
if ($models -match $Model) {
    Log "Model $Model already available."
} else {
    Warn "Downloading $Model (this may take a few minutes)..."
    ollama pull $Model
    Log "Model $Model pulled."
}

# -----------------------------------------------------------
# 4) Python environment
# -----------------------------------------------------------
Step "Setting up Python environment"

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    $py = Get-Command python3 -ErrorAction SilentlyContinue
}
if (-not $py) {
    Err "Python 3 not found. Install Python 3.9+ from python.org and try again."
    exit 1
}
$pyExe = $py.Source
$pyVersion = & $pyExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Log "Python $pyVersion found at $pyExe"

# Create venv
if (-not (Test-Path $VenvDir)) {
    Warn "Creating virtual environment..."
    & $pyExe -m venv $VenvDir
    Log "Virtual environment created."
}

# Activate
$activateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Log "Virtual environment activated."
} else {
    Err "Could not find venv activation script."
    exit 1
}

# Install deps
$reqFile = Join-Path $ScriptDir "requirements.txt"
if (Test-Path $reqFile) {
    Warn "Installing dependencies..."
    pip install --quiet --upgrade pip
    pip install --quiet -r $reqFile
    Log "Dependencies installed."
} else {
    Warn "No requirements.txt. Installing minimal deps..."
    pip install --quiet ollama pydantic requests
    Log "Minimal dependencies installed."
}

# -----------------------------------------------------------
# 5) Seed memory
# -----------------------------------------------------------
Step "Checking memory.json"

if (Test-Path $MemoryFile) {
    $memData = Get-Content $MemoryFile | ConvertFrom-Json
    $memCount = ($memData.memories | Measure-Object).Count
    $auditCount = ($memData.audit_log | Measure-Object).Count
    Log "memory.json exists: $memCount memories, $auditCount audit entries."
} else {
    Warn "Creating seed memory.json..."
    $now = & python -c "from datetime import datetime, timezone; print(datetime.now(timezone.utc).isoformat())"
    $seed = @{
        metadata = @{
            created = $now
            version = 3
            owner = "model"
            last_modified = $now
            memory_count = 0
            audit_entries = 1
        }
        memories = @()
        audit_log = @(
            @{
                timestamp = $now
                event = "created"
                actor = "executor"
                description = "Initialized empty memory store via setup script."
            }
        )
    } | ConvertTo-Json -Depth 10
    $seed | Out-File -FilePath $MemoryFile -Encoding utf8
    Log "memory.json seeded."
}

# -----------------------------------------------------------
# 6) Verify
# -----------------------------------------------------------
Step "Verifying setup"

& python -c "import ollama, pydantic, requests; print('  imports: ok')"
if ($LASTEXITCODE -ne 0) { Err "Import check failed."; exit 1 }

& python -c @"
import sys; sys.path.insert(0, r'$ScriptDir')
from qwen3_5 import Op, ProofPlan, PersistentMemory, Trace
from training_algebra import TrainingOp, TrainingStep
print(f'  Tool ops: {len(Op)}')
print(f'  Training ops: {len(TrainingOp)}')
"@
if ($LASTEXITCODE -ne 0) { Err "Module check failed."; exit 1 }

Log "All checks passed."

# -----------------------------------------------------------
# Done
# -----------------------------------------------------------
Step "Setup complete"

Write-Host ""
Write-Host "Everything is ready." -ForegroundColor Green
Write-Host ""
Write-Host "  Run the agent:"
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "    python qwen3_5.py 'your request here'" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Deploy to Modal:"
Write-Host "    modal deploy modal_app.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Restore memory from backup:"
Write-Host "    python qwen3_5.py --restore" -ForegroundColor Cyan
Write-Host ""
