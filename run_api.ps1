$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. .\scripts\Resolve-ApiPort.ps1

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$port = Resolve-ApiPort
Write-Host "服务地址: http://127.0.0.1:$port"
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port $port --reload
