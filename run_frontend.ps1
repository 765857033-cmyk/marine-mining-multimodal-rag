$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "React 静态前端由 FastAPI 托管，不需要单独启动。"
Write-Host "请运行 .\run_api.ps1 或 .\run.ps1，然后访问 http://127.0.0.1:8000"
