$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install -U `
    sentence-transformers `
    faiss-cpu `
    langgraph `
    langchain `
    langchain-openai `
    openai

Write-Host "Accuracy dependencies installed."
Write-Host "Recommended .env:"
Write-Host "EMBEDDING_BACKEND=sentence-transformers"
Write-Host "SENTENCE_TRANSFORMER_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
Write-Host "RERANK_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
