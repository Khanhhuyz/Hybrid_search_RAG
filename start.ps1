# GRAG System Startup Script
# Run this from the GRAG root directory
# Usage: .\start.ps1

Write-Host "=== GRAG - GraphRAG System ===" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Check Ollama ─────────────────────────────────────────────────────
Write-Host "[1/3] Checking Ollama..." -ForegroundColor Yellow
try {
    $ollamaResp = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 -UseBasicParsing
    Write-Host "  Ollama is running" -ForegroundColor Green
} catch {
    Write-Host "  WARNING: Ollama not detected. Please start Ollama before using the system." -ForegroundColor DarkYellow
    Write-Host "  Run: ollama serve" -ForegroundColor DarkYellow
    Write-Host "  And pull models: ollama pull nomic-embed-text && ollama pull llama3.2" -ForegroundColor DarkYellow
}

# ── Step 2: Start Backend ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "[2/3] Starting FastAPI backend..." -ForegroundColor Yellow

$backendDir = Join-Path $PSScriptRoot "backend"
$venvPython = Join-Path $backendDir ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "  Creating Python virtual environment using Python 3.12..." -ForegroundColor DarkYellow
    Push-Location $backendDir
    py -3.12 -m venv .venv
    & ".venv\Scripts\pip.exe" install -r requirements.txt --quiet
    Copy-Item ".env.example" ".env" -ErrorAction SilentlyContinue
    Pop-Location
}

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$backendDir'; .\.venv\Scripts\uvicorn.exe app.main:app --reload --port 8000" -WindowStyle Normal

Write-Host "  Backend starting at http://localhost:8000" -ForegroundColor Green
Write-Host "  API docs: http://localhost:8000/docs" -ForegroundColor DarkGray

# ── Step 3: Start Frontend ────────────────────────────────────────────────────
Write-Host ""
Write-Host "[3/3] Starting Next.js frontend..." -ForegroundColor Yellow

$frontendDir = Join-Path $PSScriptRoot "frontend"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontendDir'; npm run dev" -WindowStyle Normal

Write-Host "  Frontend starting at http://localhost:3000" -ForegroundColor Green

# ── Summary ──────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== All services starting ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Frontend:     http://localhost:3000" -ForegroundColor White
Write-Host "  Backend API:  http://localhost:8000" -ForegroundColor White
Write-Host "  API Docs:     http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "Give services 10-15 seconds to fully initialize." -ForegroundColor DarkGray
