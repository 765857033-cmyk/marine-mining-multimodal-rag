$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. .\scripts\Resolve-ApiPort.ps1

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

$port = Resolve-ApiPort
Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$PSScriptRoot\run_api.ps1`""

Write-Host "FastAPI / React: http://127.0.0.1:$port"
