$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install -U docling langchain-docling

Write-Host "Docling installed."
Write-Host "Set PARSER_BACKEND=docling in .env or select Docling in the Streamlit sidebar."
