from __future__ import annotations

import shutil
from pathlib import Path
from threading import RLock
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import MiningRagAgent
from .config import AppConfig
from .feedback import FeedbackStore
from .memory import MemoryStore
from .models import AgentAnswer
from .multi_agents import MultiAgentRagSystem
from .observability import list_traces, load_trace
from .vector_store import EmbeddingProvider, VectorIndex


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    session_id: str = Field(default="", max_length=128)
    memory_scope: str = Field(default="default", max_length=128)


class RebuildRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    trace_id: str = Field(min_length=1)
    session_id: str = ""
    rating: int = Field(ge=-1, le=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    issue_type: str = ""
    correction: str = ""
    comment: str = ""
    citations: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)


class FeedbackReviewRequest(BaseModel):
    status: str = Field(pattern="^(pending|approved|rejected|resolved)$")


class LongMemoryRequest(BaseModel):
    content: str = Field(min_length=1)
    scope: str = "default"
    importance: float = Field(default=1.0, ge=0.0, le=10.0)


def create_app(config: AppConfig | None = None) -> FastAPI:
    load_dotenv()
    app = FastAPI(
        title="Mining Agentic RAG API",
        description="FastAPI service for PDF ingestion, Agentic RAG chat, and source tracing.",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:8000",
            "http://localhost:8000",
            "http://127.0.0.1:8001",
            "http://localhost:8001",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.config = config or AppConfig.from_env()
    app.state.config.upload_dir.mkdir(parents=True, exist_ok=True)
    app.state.lock = RLock()
    app.state.memory_store = MemoryStore(app.state.config.state_db_path)
    app.state.feedback_store = FeedbackStore(
        app.state.config.state_db_path,
        app.state.config.feedback_bad_cases_path,
    )
    load_agent_runtime(app)

    @app.get("/health")
    def health() -> dict[str, Any]:
        cfg: AppConfig = app.state.config
        return {
            "status": "ok",
            "index_loaded": app.state.index_loaded,
            "parser_backend": cfg.parser_backend,
            "embedding_backend": cfg.embedding_backend,
            "vector_backend": cfg.vector_backend,
            "llm_backend": cfg.llm_backend,
            "mineru_backend": cfg.mineru_backend,
            "mineru_method": cfg.mineru_method,
            "multimodal_enabled": cfg.multimodal_enabled,
            "vision_model": cfg.vision_model,
            "rerank_backend": cfg.rerank_backend,
            "multi_agent_enabled": True,
            "agents": app.state.multi_agent_system.agent_names,
        }

    @app.get("/index/stats")
    def index_stats() -> dict[str, Any]:
        index: VectorIndex = app.state.index
        return {
            "index_loaded": app.state.index_loaded,
            "index_dir": str(app.state.config.index_dir),
            "search_chunks": len(index.search_chunks),
            "parent_chunks": len(index.parent_chunks),
            "embedding_name": index.embedding.name,
            "vector_backend": index.backend,
            "sources": sorted({chunk.source for chunk in index.parent_chunks}),
            "modalities": count_modalities(index.parent_chunks),
        }

    @app.post("/chat")
    def chat(request: ChatRequest) -> dict[str, Any]:
        try:
            with app.state.lock:
                answer = app.state.multi_agent_system.answer(
                    request.question,
                    history=[message.model_dump() for message in request.history],
                    session_id=request.session_id,
                    memory_scope=request.memory_scope,
                )
                return serialize_answer(request.question, answer, app.state.index_loaded, app.state.config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Agent invoke failed: {exc}") from exc

    @app.get("/memory/{session_id}")
    def get_memory(session_id: str, scope: str = "default") -> dict[str, Any]:
        return app.state.memory_store.snapshot(session_id, scope)

    @app.delete("/memory/{session_id}")
    def clear_memory(session_id: str) -> dict[str, str]:
        app.state.memory_store.clear_session(session_id)
        return {"status": "ok", "session_id": session_id}

    @app.post("/memory/long-term")
    def add_long_memory(request: LongMemoryRequest) -> dict[str, str]:
        memory_id = app.state.memory_store.add_long_term(
            request.content,
            scope=request.scope,
            importance=request.importance,
            capacity=app.state.config.memory_long_capacity,
        )
        return {"status": "ok", "memory_id": memory_id}

    @app.delete("/memory/long-term/{memory_id}")
    def delete_long_memory(memory_id: str) -> dict[str, Any]:
        return {"status": "ok", "deleted": app.state.memory_store.delete_long_term(memory_id)}

    @app.get("/traces")
    def traces(limit: int = 50) -> list[dict]:
        return list_traces(app.state.config.trace_dir, max(1, min(limit, 200)))

    @app.get("/traces/{trace_id}")
    def trace_detail(trace_id: str) -> dict:
        trace = load_trace(app.state.config.trace_dir, trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="Trace not found.")
        return trace

    @app.post("/feedback")
    def submit_feedback(request: FeedbackRequest) -> dict:
        if request.rating not in {-1, 1}:
            raise HTTPException(status_code=422, detail="rating must be -1 or 1.")
        return app.state.feedback_store.submit(**request.model_dump())

    @app.get("/feedback")
    def list_feedback(status: str | None = None, limit: int = 100) -> list[dict]:
        return app.state.feedback_store.list(status=status, limit=limit)

    @app.post("/feedback/{feedback_id}/review")
    def review_feedback(feedback_id: str, request: FeedbackReviewRequest) -> dict:
        try:
            return app.state.feedback_store.update_status(feedback_id, request.status)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Feedback not found.") from exc

    @app.post("/documents/upload")
    def upload_documents(files: list[UploadFile] = File(...)) -> dict[str, Any]:
        if not files:
            raise HTTPException(status_code=400, detail="At least one PDF file is required.")
        cfg: AppConfig = app.state.config
        cfg.upload_dir.mkdir(parents=True, exist_ok=True)

        pdf_paths: list[Path] = []
        for file in files:
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"Only PDF files are supported: {file.filename}")
            target = cfg.upload_dir / Path(file.filename).name
            with target.open("wb") as writer:
                shutil.copyfileobj(file.file, writer)
            pdf_paths.append(target)

        return rebuild_from_paths(app, pdf_paths)

    @app.post("/index/rebuild")
    def rebuild_index(request: RebuildRequest) -> dict[str, Any]:
        cfg: AppConfig = app.state.config
        if request.paths:
            pdf_paths = [Path(path) for path in request.paths]
        else:
            pdf_paths = sorted(cfg.upload_dir.glob("*.pdf"))
        missing = [str(path) for path in pdf_paths if not path.exists()]
        if missing:
            raise HTTPException(status_code=400, detail={"missing_files": missing})
        if not pdf_paths:
            raise HTTPException(status_code=400, detail="No PDF files found for rebuild.")
        return rebuild_from_paths(app, pdf_paths)

    mount_frontend(app)
    return app


def mount_frontend(app: FastAPI) -> None:
    cfg: AppConfig = app.state.config
    app.mount("/assets", StaticFiles(directory=cfg.upload_dir), name="assets")
    frontend_static = Path(__file__).resolve().parents[1] / "frontend" / "static"
    if frontend_static.exists():
        app.mount("/", StaticFiles(directory=frontend_static, html=True), name="frontend")


def rebuild_from_paths(app: FastAPI, pdf_paths: list[Path]) -> dict[str, Any]:
    cfg: AppConfig = app.state.config
    try:
        result = app.state.multi_agent_system.rebuild_knowledge_base(pdf_paths)
        index = result["index"]
        with app.state.lock:
            load_agent_runtime(app)
        return {
            "status": "ok",
            "uploaded_sources": result["uploaded_sources"],
            "parent_chunks": len(index.parent_chunks),
            "search_chunks": len(index.search_chunks),
            "modalities": result["modalities"],
            "index_dir": str(cfg.index_dir),
            "multi_agent_trace_id": result["trace_id"],
            "multi_agent_trace_events": result["trace_events"],
            "agents": result["agents"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Index rebuild failed: {exc}") from exc


def load_agent_runtime(app: FastAPI) -> None:
    cfg: AppConfig = app.state.config
    index = VectorIndex(EmbeddingProvider(cfg.embedding_backend), cfg.vector_backend)
    app.state.index_loaded = index.load(cfg.index_dir)
    app.state.index = index
    app.state.multi_agent_system = MultiAgentRagSystem(cfg, index, memory_store=app.state.memory_store)
    app.state.agent = MiningRagAgent(index, cfg, memory_store=app.state.memory_store)


def serialize_answer(question: str, answer: AgentAnswer, index_loaded: bool, config: AppConfig) -> dict[str, Any]:
    return {
        "question": question,
        "answer": answer.answer,
        "route": answer.route,
        "need_retrieval": answer.need_retrieval,
        "rewritten_query": answer.rewritten_query,
        "query_variants": answer.query_variants,
        "router_reason": answer.router_reason,
        "router_confidence": answer.router_confidence,
        "verification_passed": answer.verification_passed,
        "verification_reason": answer.verification_reason,
        "retrieval_rounds": answer.retrieval_rounds,
        "index_loaded": index_loaded,
        "sources": [serialize_source(result, config) for result in answer.sources],
        "trace": answer.trace,
        "trace_id": answer.trace_id,
        "trace_events": answer.trace_events,
        "citation_validation": answer.citation_validation,
        "memory": answer.memory,
    }


def serialize_source(result, config: AppConfig) -> dict[str, Any]:
    chunk = result.chunk
    return {
        "source": chunk.source,
        "page": chunk.page,
        "modality": chunk.modality,
        "chunk_role": chunk.chunk_role,
        "parent_id": chunk.parent_id,
        "score": result.score,
        "final_score": result.final_score,
        "vector_score": result.vector_score,
        "keyword_score": result.keyword_score,
        "retrieval_method": result.retrieval_method,
        "reason": result.reason,
        "heading": chunk.metadata.get("heading", ""),
        "matched_child_text": chunk.metadata.get("matched_child_text", ""),
        "text_preview": chunk.text[:700],
        "asset_url": asset_url(chunk.metadata.get("asset_path", ""), config),
        "metadata": safe_metadata(chunk.metadata),
    }


def asset_url(asset_path: str, config: AppConfig) -> str:
    if not asset_path:
        return ""
    try:
        path = Path(asset_path).resolve()
        upload_dir = config.upload_dir.resolve()
        relative = path.relative_to(upload_dir)
    except Exception:
        return ""
    return "/assets/" + "/".join(relative.parts)


def safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, list):
            safe[key] = [item for item in value if isinstance(item, (str, int, float, bool, type(None)))][:20]
        else:
            safe[key] = str(value)
    return safe


def count_modalities(chunks) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.modality] = counts.get(chunk.modality, 0) + 1
    return counts


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
