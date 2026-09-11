from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent import MiningRagAgent
from .config import AppConfig
from .memory import MemoryStore
from .models import AgentAnswer, ConversationTurn, DocumentChunk
from .observability import start_timer, trace_event
from .pdf_ingest import ingest_pdfs
from .vector_store import EmbeddingProvider, VectorIndex


@dataclass(slots=True)
class DocumentParseResult:
    chunks: list[DocumentChunk]
    trace_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class KnowledgeBuildResult:
    index: VectorIndex
    trace_events: list[dict[str, Any]] = field(default_factory=list)


class DocParserAgent:
    """文档解析 Agent：负责调用 MinerU，把 PDF 转换成结构化证据块。"""

    name = "DocParserAgent"

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def parse(self, pdf_paths: list[Path]) -> DocumentParseResult:
        trace_events: list[dict[str, Any]] = []
        started = start_timer()
        chunks = ingest_pdfs(pdf_paths, self.config)
        trace_event(
            trace_events,
            self.name,
            started,
            input_files=[path.name for path in pdf_paths],
            parser_backend=self.config.parser_backend,
            mineru_backend=self.config.mineru_backend,
            mineru_method=self.config.mineru_method,
            chunk_count=len(chunks),
            modalities=count_modalities(chunks),
        )
        return DocumentParseResult(chunks=chunks, trace_events=trace_events)


class KnowledgeExtractAgent:
    """知识抽取 Agent：负责父子切片、Embedding、Chroma/BM25 索引构建。"""

    name = "KnowledgeExtractAgent"

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def build_index(self, chunks: list[DocumentChunk]) -> KnowledgeBuildResult:
        trace_events: list[dict[str, Any]] = []
        started = start_timer()
        index = VectorIndex(EmbeddingProvider(self.config.embedding_backend), self.config.vector_backend)
        index.build(
            chunks,
            parent_child_enabled=self.config.parent_child_enabled,
            child_chunk_size=self.config.child_chunk_size,
            child_chunk_overlap=self.config.child_chunk_overlap,
        )
        index.save(self.config.index_dir)
        trace_event(
            trace_events,
            self.name,
            started,
            parent_child_enabled=self.config.parent_child_enabled,
            parent_chunks=len(index.parent_chunks),
            search_chunks=len(index.search_chunks),
            embedding_name=index.embedding.name,
            vector_backend=index.backend,
            retrieval="Chroma semantic retrieval + BM25 keyword retrieval + RRF fusion",
        )
        return KnowledgeBuildResult(index=index, trace_events=trace_events)


class QAAgent:
    """问答 Agent：负责路由、改写、混合检索、重排、生成、校验和记忆写入。"""

    name = "QAAgent"

    def __init__(self, index: VectorIndex, config: AppConfig, memory_store: MemoryStore | None = None) -> None:
        self.agent = MiningRagAgent(index, config, memory_store=memory_store)

    def answer(
        self,
        question: str,
        history: list[ConversationTurn | dict] | None = None,
        session_id: str = "",
        memory_scope: str = "default",
    ) -> AgentAnswer:
        answer = self.agent.invoke(
            question,
            history=history,
            session_id=session_id,
            memory_scope=memory_scope,
        )
        answer.trace.insert(0, f"{self.name}: received question and delegated to LangGraph Agentic RAG workflow")
        answer.trace_events.insert(
            0,
            {
                "node": self.name,
                "elapsed_ms": 0.0,
                "question_chars": len(question),
                "responsibility": "router + query rewrite + hybrid retrieval + rerank + generation + verification",
            },
        )
        return answer


class MultiAgentRagSystem:
    """多 Agent 编排器：只包含解析、知识抽取和问答，不包含增量更新 Agent。"""

    def __init__(
        self,
        config: AppConfig,
        index: VectorIndex | None = None,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self.config = config
        self.memory_store = memory_store
        self.index = index or VectorIndex(EmbeddingProvider(config.embedding_backend), config.vector_backend)
        self.doc_parser = DocParserAgent(config)
        self.knowledge_extractor = KnowledgeExtractAgent(config)
        self.qa_agent = QAAgent(self.index, config, memory_store=memory_store)

    @property
    def agent_names(self) -> list[str]:
        return [self.doc_parser.name, self.knowledge_extractor.name, self.qa_agent.name]

    def rebuild_knowledge_base(self, pdf_paths: list[Path]) -> dict[str, Any]:
        trace_id = uuid.uuid4().hex
        parse_result = self.doc_parser.parse(pdf_paths)
        build_result = self.knowledge_extractor.build_index(parse_result.chunks)
        self.index = build_result.index
        self.qa_agent = QAAgent(self.index, self.config, memory_store=self.memory_store)
        trace_events = [*parse_result.trace_events, *build_result.trace_events]
        return {
            "trace_id": trace_id,
            "trace_events": trace_events,
            "index": self.index,
            "uploaded_sources": [path.name for path in pdf_paths],
            "parent_chunks": len(self.index.parent_chunks),
            "search_chunks": len(self.index.search_chunks),
            "modalities": count_modalities(self.index.parent_chunks),
            "agents": self.agent_names,
        }

    def answer(
        self,
        question: str,
        history: list[ConversationTurn | dict] | None = None,
        session_id: str = "",
        memory_scope: str = "default",
    ) -> AgentAnswer:
        return self.qa_agent.answer(
            question,
            history=history,
            session_id=session_id,
            memory_scope=memory_scope,
        )


def count_modalities(chunks: list[DocumentChunk]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.modality] = counts.get(chunk.modality, 0) + 1
    return counts
