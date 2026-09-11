$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -U "mineru[core]>=3.0"

Write-Host ""
Write-Host "MinerU installed. Set PARSER_BACKEND=mineru in .env, then rebuild the index in React or FastAPI."
