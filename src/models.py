from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    source: str
    page: int
    text: str
    modality: str = "text"
    parent_id: str = ""
    chunk_role: str = "parent"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    chunk: DocumentChunk
    score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0
    retrieval_method: str = "vector"
    rerank_score: float | None = None
    reason: str = ""

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score


@dataclass(slots=True)
class ConversationTurn:
    role: str
    content: str


@dataclass(slots=True)
class AgentAnswer:
    answer: str
    route: str
    need_retrieval: bool
    sources: list[RetrievalResult]
    rewritten_query: str
    query_variants: list[str] = field(default_factory=list)
    router_reason: str = ""
    router_confidence: float = 0.0
    verification_passed: bool = True
    verification_reason: str = ""
    retrieval_rounds: int = 0
    trace: list[str] = field(default_factory=list)
    trace_id: str = ""
    trace_events: list[dict[str, Any]] = field(default_factory=list)
    citation_validation: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
