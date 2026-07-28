$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install streamlit python-dotenv pandas pymupdf
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
