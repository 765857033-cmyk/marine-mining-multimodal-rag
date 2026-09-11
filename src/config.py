from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    index_dir: Path = Path("data/index")
    upload_dir: Path = Path("data/uploads")
    state_db_path: Path = Path("data/agent_state.db")
    trace_dir: Path = Path("data/traces")
    feedback_bad_cases_path: Path = Path("evaluation/feedback_bad_cases.jsonl")
    chunk_size: int = 850
    chunk_overlap: int = 120
    parent_child_enabled: bool = True
    parent_chunk_size: int = 1800
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 420
    child_chunk_overlap: int = 80
    parser_backend: str = "mineru"
    parser_fallback: bool = False
    mineru_command: str = "mineru"
    mineru_backend: str = "pipeline"
    mineru_method: str = "auto"
    mineru_extra_args: str = ""
    mineru_timeout: int = 900
    mineru_export_artifacts: bool = True
    top_k: int = 10
    keyword_top_k: int = 10
    rerank_top_k: int = 5
    min_relevance: float = 0.12
    min_keyword_coverage: float = 0.28
    min_complex_sources: int = 2
    max_retrieval_rounds: int = 2
    multi_query_count: int = 4
    multimodal_enabled: bool = True
    extract_tables: bool = True
    extract_images: bool = True
    max_images_per_pdf: int = 24
    max_image_summary_chars: int = 900
    vision_model: str = "gpt-4o-mini"
    vector_backend: str = "chroma"
    embedding_backend: str = "auto"
    llm_backend: str = "auto"
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""
    oceangpt_model: str = "OceanGPT"
    oceangpt_base_url: str = ""
    custom_llm_model: str = ""
    custom_llm_base_url: str = ""
    llm_timeout: int = 120
    llm_router_enabled: bool = True
    memory_enabled: bool = True
    memory_max_turns: int = 6
    memory_short_max_chars: int = 6000
    memory_summary_max_chars: int = 4000
    memory_long_capacity: int = 64
    memory_long_recall_k: int = 8
    citation_validation_enabled: bool = True
    citation_min_coverage: float = 0.6
    citation_repair_max_attempts: int = 1
    trace_enabled: bool = True
    rerank_backend: str = "auto"
    rerank_model: str = ""

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            index_dir=Path(os.getenv("INDEX_DIR", "data/index")),
            upload_dir=Path(os.getenv("UPLOAD_DIR", "data/uploads")),
            state_db_path=Path(os.getenv("STATE_DB_PATH", "data/agent_state.db")),
            trace_dir=Path(os.getenv("TRACE_DIR", "data/traces")),
            feedback_bad_cases_path=Path(
                os.getenv("FEEDBACK_BAD_CASES_PATH", "evaluation/feedback_bad_cases.jsonl")
            ),
            chunk_size=int(os.getenv("CHUNK_SIZE", "850")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
            parent_child_enabled=os.getenv("PARENT_CHILD_ENABLED", "true").lower() in {"1", "true", "yes", "on"},
            parent_chunk_size=int(os.getenv("PARENT_CHUNK_SIZE", "1800")),
            parent_chunk_overlap=int(os.getenv("PARENT_CHUNK_OVERLAP", "200")),
            child_chunk_size=int(os.getenv("CHILD_CHUNK_SIZE", "420")),
            child_chunk_overlap=int(os.getenv("CHILD_CHUNK_OVERLAP", "80")),
            parser_backend=os.getenv("PARSER_BACKEND", "mineru").lower(),
            parser_fallback=os.getenv("PARSER_FALLBACK", "false").lower() in {"1", "true", "yes", "on"},
            mineru_command=os.getenv("MINERU_COMMAND", "mineru"),
            mineru_backend=os.getenv("MINERU_BACKEND", "pipeline"),
            mineru_method=os.getenv("MINERU_METHOD", "auto"),
            mineru_extra_args=os.getenv("MINERU_EXTRA_ARGS", ""),
            mineru_timeout=int(os.getenv("MINERU_TIMEOUT", "900")),
            mineru_export_artifacts=os.getenv("MINERU_EXPORT_ARTIFACTS", "true").lower()
            in {"1", "true", "yes", "on"},
            top_k=int(os.getenv("TOP_K", "10")),
            keyword_top_k=int(os.getenv("KEYWORD_TOP_K", "10")),
            rerank_top_k=int(os.getenv("RERANK_TOP_K", "5")),
            min_relevance=float(os.getenv("MIN_RELEVANCE", "0.12")),
            min_keyword_coverage=float(os.getenv("MIN_KEYWORD_COVERAGE", "0.28")),
            min_complex_sources=int(os.getenv("MIN_COMPLEX_SOURCES", "2")),
            max_retrieval_rounds=int(os.getenv("MAX_RETRIEVAL_ROUNDS", "2")),
            multi_query_count=int(os.getenv("MULTI_QUERY_COUNT", "4")),
            multimodal_enabled=os.getenv("MULTIMODAL_ENABLED", "true").lower() in {"1", "true", "yes", "on"},
            extract_tables=os.getenv("EXTRACT_TABLES", "true").lower() in {"1", "true", "yes", "on"},
            extract_images=os.getenv("EXTRACT_IMAGES", "true").lower() in {"1", "true", "yes", "on"},
            max_images_per_pdf=int(os.getenv("MAX_IMAGES_PER_PDF", "24")),
            max_image_summary_chars=int(os.getenv("MAX_IMAGE_SUMMARY_CHARS", "900")),
            vision_model=os.getenv("VISION_MODEL", "gpt-4o-mini"),
            vector_backend=os.getenv("VECTOR_BACKEND", "chroma").lower(),
            embedding_backend=os.getenv("EMBEDDING_BACKEND", "auto").lower(),
            llm_backend=os.getenv("LLM_BACKEND", "auto").lower(),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
            oceangpt_model=os.getenv("OCEANGPT_MODEL", "OceanGPT"),
            oceangpt_base_url=os.getenv("OCEANGPT_BASE_URL", ""),
            custom_llm_model=os.getenv("CUSTOM_LLM_MODEL", ""),
            custom_llm_base_url=os.getenv("CUSTOM_LLM_BASE_URL", ""),
            llm_timeout=int(os.getenv("LLM_TIMEOUT", "120")),
            llm_router_enabled=os.getenv("LLM_ROUTER_ENABLED", "true").lower() in {"1", "true", "yes", "on"},
            memory_enabled=os.getenv("MEMORY_ENABLED", "true").lower() in {"1", "true", "yes", "on"},
            memory_max_turns=int(os.getenv("MEMORY_MAX_TURNS", "6")),
            memory_short_max_chars=int(os.getenv("MEMORY_SHORT_MAX_CHARS", "6000")),
            memory_summary_max_chars=int(os.getenv("MEMORY_SUMMARY_MAX_CHARS", "4000")),
            memory_long_capacity=int(os.getenv("MEMORY_LONG_CAPACITY", "64")),
            memory_long_recall_k=int(os.getenv("MEMORY_LONG_RECALL_K", "8")),
            citation_validation_enabled=os.getenv("CITATION_VALIDATION_ENABLED", "true").lower()
            in {"1", "true", "yes", "on"},
            citation_min_coverage=float(os.getenv("CITATION_MIN_COVERAGE", "0.6")),
            citation_repair_max_attempts=int(os.getenv("CITATION_REPAIR_MAX_ATTEMPTS", "1")),
            trace_enabled=os.getenv("TRACE_ENABLED", "true").lower() in {"1", "true", "yes", "on"},
            rerank_backend=os.getenv("RERANK_BACKEND", "auto").lower(),
            rerank_model=os.getenv("RERANK_MODEL", ""),
        )
