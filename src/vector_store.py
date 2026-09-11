from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .models import DocumentChunk, RetrievalResult
from .parent_child import build_parent_child_chunks
from .text_processing import DOMAIN_TERMS


class EmbeddingProvider:
    def __init__(self, backend: str = "auto", dim: int = 384) -> None:
        self.backend = backend
        self.dim = dim
        self._model = None
        self.name = "hashing"
        self._init_model()

    def _init_model(self) -> None:
        if self.backend in {"auto", "openai"} and os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import OpenAIEmbeddings

                self._model = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
                self.name = "openai"
                return
            except Exception:
                if self.backend == "openai":
                    raise
        if self.backend in {"auto", "sentence-transformers", "sentence_transformers"}:
            try:
                from sentence_transformers import SentenceTransformer

                model_name = os.getenv(
                    "SENTENCE_TRANSFORMER_MODEL",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                )
                self._model = SentenceTransformer(model_name)
                self.name = "sentence-transformers"
                return
            except Exception:
                if self.backend != "auto":
                    raise

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.name == "openai":
            vectors = self._model.embed_documents(texts)
            return _normalize(np.asarray(vectors, dtype=np.float32))
        if self.name == "sentence-transformers":
            vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vectors, dtype=np.float32)
        return _normalize(np.vstack([self._hashing_embed(text) for text in texts]).astype(np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def _hashing_embed(self, text: str) -> np.ndarray:
        tokens = [token.lower() for token in _tokenize(text)]
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big")
            index = raw % self.dim
            sign = 1.0 if (raw >> 8) % 2 == 0 else -1.0
            vector[index] += sign * (1.0 + math.log1p(len(token)))
        return vector


class VectorIndex:
    def __init__(self, embedding: EmbeddingProvider, backend: str = "chroma") -> None:
        self.embedding = embedding
        self.backend = backend.lower()
        if self.backend != "chroma":
            raise ValueError("项目已取消 numpy/FAISS 兜底检索，请使用 VECTOR_BACKEND=chroma。")
        self.chunks: list[DocumentChunk] = []
        self.parent_chunks: list[DocumentChunk] = []
        self.search_chunks: list[DocumentChunk] = []
        self.parent_lookup: dict[str, DocumentChunk] = {}
        self.embeddings: np.ndarray = np.zeros((0, embedding.dim), dtype=np.float32)
        self._chroma_client = None
        self._chroma_collection = None
        self._chroma_path: Path | None = None
        self._tokenized_chunks: list[list[str]] = []
        self._bm25_idf: dict[str, float] = {}
        self._avg_doc_len = 1.0

    def build(
        self,
        chunks: list[DocumentChunk],
        parent_child_enabled: bool = True,
        child_chunk_size: int = 420,
        child_chunk_overlap: int = 80,
    ) -> None:
        if parent_child_enabled:
            self.parent_chunks, self.search_chunks = build_parent_child_chunks(
                chunks,
                child_chunk_size=child_chunk_size,
                child_chunk_overlap=child_chunk_overlap,
            )
        else:
            self.parent_chunks = [ensure_parent_chunk(chunk, index) for index, chunk in enumerate(chunks)]
            self.search_chunks = self.parent_chunks
        self.chunks = self.search_chunks
        self.parent_lookup = {chunk.parent_id or chunk.chunk_id: chunk for chunk in self.parent_chunks}
        self.embeddings = self.embedding.embed_documents([chunk.text for chunk in self.search_chunks])
        self._build_keyword_index()
        self._build_chroma()

    def _build_keyword_index(self) -> None:
        self._tokenized_chunks = [_tokenize(chunk.text) for chunk in self.search_chunks]
        doc_count = max(1, len(self._tokenized_chunks))
        self._avg_doc_len = sum(len(tokens) for tokens in self._tokenized_chunks) / doc_count or 1.0
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokenized_chunks:
            document_frequency.update(set(tokens))
        self._bm25_idf = {
            token: math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            for token, df in document_frequency.items()
        }

    def _build_chroma(self, persist_path: Path | None = None) -> None:
        self._chroma_client = None
        self._chroma_collection = None
        self._chroma_path = persist_path
        if len(self.embeddings) == 0:
            return
        try:
            import chromadb

            if persist_path is not None:
                persist_path.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=str(persist_path))
            else:
                client = chromadb.Client()
            collection = client.get_or_create_collection(
                name="mining_agentic_rag",
                metadata={"hnsw:space": "cosine"},
            )
            existing_count = collection.count()
            if existing_count:
                existing = collection.get(include=[])
                ids = existing.get("ids", [])
                if ids:
                    collection.delete(ids=ids)
            ids = [chunk.chunk_id for chunk in self.search_chunks]
            collection.add(
                ids=ids,
                documents=[chunk.text for chunk in self.search_chunks],
                embeddings=self.embeddings.astype(float).tolist(),
                metadatas=[
                    {
                        "index": index,
                        "source": chunk.source,
                        "page": chunk.page,
                        "modality": chunk.modality,
                    }
                    for index, chunk in enumerate(self.search_chunks)
                ],
            )
            self._chroma_client = client
            self._chroma_collection = collection
        except Exception as exc:
            raise RuntimeError("Chroma 初始化失败，请确认已安装 chromadb 且索引目录可写。") from exc

    def search(self, query: str, top_k: int, keyword_top_k: int | None = None) -> list[RetrievalResult]:
        return self.hybrid_search(query, top_k=top_k, keyword_top_k=keyword_top_k or top_k)

    def hybrid_search(self, query: str, top_k: int, keyword_top_k: int) -> list[RetrievalResult]:
        vector_results = self.vector_search(query, top_k)
        keyword_results = self.keyword_search(query, keyword_top_k)
        return self._fuse_results(vector_results, keyword_results, top_k=max(top_k, keyword_top_k))

    def vector_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if not self.search_chunks:
            return []
        query_vector = self.embedding.embed_query(query).astype(np.float32)
        if self._chroma_collection is None:
            raise RuntimeError("Chroma collection 未初始化，无法执行向量检索。")
        payload = self._chroma_collection.query(
            query_embeddings=[query_vector.astype(float).tolist()],
            n_results=min(top_k, len(self.search_chunks)),
            include=["distances", "metadatas"],
        )
        ids = payload.get("ids", [[]])[0]
        distances = payload.get("distances", [[]])[0]
        metadatas = payload.get("metadatas", [[]])[0]
        results: list[RetrievalResult] = []
        for item_id, distance, metadata in zip(ids, distances, metadatas):
            index = int((metadata or {}).get("index", -1))
            if index < 0:
                index = next(
                    (
                        candidate
                        for candidate, chunk in enumerate(self.search_chunks)
                        if chunk.chunk_id == item_id
                    ),
                    -1,
                )
            if index < 0 or index >= len(self.search_chunks):
                continue
            score = 1.0 - float(distance)
            results.append(
                RetrievalResult(
                    chunk=self.search_chunks[index],
                    score=score,
                    vector_score=score,
                    retrieval_method="vector",
                    reason="vector=chroma",
                )
            )
        return results

    def keyword_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if not self.search_chunks:
            return []
        if not self._tokenized_chunks:
            self._build_keyword_index()
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        query_counts = Counter(query_terms)
        scores: list[tuple[int, float]] = []
        k1 = 1.5
        b = 0.75
        for index, doc_terms in enumerate(self._tokenized_chunks):
            if not doc_terms:
                continue
            term_counts = Counter(doc_terms)
            doc_len = len(doc_terms)
            score = 0.0
            for term, query_weight in query_counts.items():
                freq = term_counts.get(term, 0)
                if freq == 0:
                    continue
                idf = self._bm25_idf.get(term, 0.0)
                denom = freq + k1 * (1 - b + b * doc_len / self._avg_doc_len)
                score += query_weight * idf * (freq * (k1 + 1)) / max(denom, 1e-9)
            if score > 0:
                scores.append((index, score))

        ranked = sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]
        return [
            RetrievalResult(
                chunk=self.search_chunks[index],
                score=float(score),
                keyword_score=float(score),
                retrieval_method="keyword",
                reason="bm25",
            )
            for index, score in ranked
        ]

    def _fuse_results(
        self,
        vector_results: list[RetrievalResult],
        keyword_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        fused: dict[str, RetrievalResult] = {}
        vector_rank: dict[str, int] = {}
        keyword_rank: dict[str, int] = {}
        k = 60.0

        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result.chunk.chunk_id
            vector_rank[chunk_id] = rank
            fused[chunk_id] = RetrievalResult(
                chunk=result.chunk,
                score=1.0 / (k + rank),
                vector_score=result.vector_score or result.score,
                retrieval_method="hybrid",
            )

        for rank, result in enumerate(keyword_results, start=1):
            chunk_id = result.chunk.chunk_id
            keyword_rank[chunk_id] = rank
            if chunk_id not in fused:
                fused[chunk_id] = RetrievalResult(
                    chunk=result.chunk,
                    score=0.0,
                    retrieval_method="hybrid",
                )
            fused_item = fused[chunk_id]
            fused_item.score += 1.0 / (k + rank)
            fused_item.keyword_score = result.keyword_score or result.score

        for chunk_id, result in fused.items():
            methods = []
            if chunk_id in vector_rank:
                methods.append(f"vector_rank={vector_rank[chunk_id]}")
            if chunk_id in keyword_rank:
                methods.append(f"bm25_rank={keyword_rank[chunk_id]}")
            result.reason = ", ".join(methods)

        child_results = sorted(fused.values(), key=lambda item: item.score, reverse=True)[:top_k]
        return self._promote_children_to_parents(child_results, top_k)

    def _promote_children_to_parents(self, child_results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        promoted: dict[str, RetrievalResult] = {}
        for result in child_results:
            child = result.chunk
            parent_id = child.parent_id or child.metadata.get("parent_id") or child.chunk_id
            parent = self.parent_lookup.get(parent_id, child)
            if parent_id not in promoted:
                parent_metadata = dict(parent.metadata)
                parent_metadata.update(
                    {
                        "matched_child_id": child.chunk_id,
                        "matched_child_text": child.text,
                        "matched_child_page": child.page,
                        "matched_child_index": child.metadata.get("child_index", 0),
                        "retrieved_parent_id": parent_id,
                    }
                )
                parent_chunk = DocumentChunk(
                    chunk_id=parent.chunk_id,
                    source=parent.source,
                    page=parent.page,
                    text=parent.text,
                    modality=parent.modality,
                    parent_id=parent.parent_id or parent.chunk_id,
                    chunk_role="parent",
                    metadata=parent_metadata,
                )
                promoted[parent_id] = RetrievalResult(
                    chunk=parent_chunk,
                    score=result.score,
                    vector_score=result.vector_score,
                    keyword_score=result.keyword_score,
                    retrieval_method="parent_child_hybrid",
                    reason=f"parent_recall from child={child.chunk_id}; {result.reason}",
                )
            else:
                current = promoted[parent_id]
                current.score += result.score * 0.5
                current.vector_score = max(current.vector_score, result.vector_score)
                current.keyword_score = max(current.keyword_score, result.keyword_score)
                current.reason = f"{current.reason}; additional_child={child.chunk_id}"
        return sorted(promoted.values(), key=lambda item: item.score, reverse=True)[:top_k]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedding_name": self.embedding.name,
            "backend": self.backend,
            "chunks": [asdict(chunk) for chunk in self.search_chunks],
            "parent_chunks": [asdict(chunk) for chunk in self.parent_chunks],
            "index_mode": "parent_child" if self.parent_chunks != self.search_chunks else "flat",
        }
        (path / "chunks.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(path / "embeddings.npy", self.embeddings)
        if self.backend == "chroma":
            self._build_chroma(path / "chroma")

    def load(self, path: Path) -> bool:
        chunks_path = path / "chunks.json"
        embeddings_path = path / "embeddings.npy"
        if not chunks_path.exists() or not embeddings_path.exists():
            return False
        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.search_chunks = [DocumentChunk(**_normalize_chunk_payload(chunk)) for chunk in payload["chunks"]]
        parent_payload = payload.get("parent_chunks") or payload.get("parents") or []
        if parent_payload:
            self.parent_chunks = [DocumentChunk(**_normalize_chunk_payload(chunk)) for chunk in parent_payload]
        else:
            self.parent_chunks = [ensure_parent_chunk(chunk, index) for index, chunk in enumerate(self.search_chunks)]
        self.parent_lookup = {chunk.parent_id or chunk.chunk_id: chunk for chunk in self.parent_chunks}
        self.chunks = self.search_chunks
        self.embeddings = np.load(embeddings_path).astype(np.float32)
        if self.embeddings.ndim == 2 and self.embeddings.shape[1] > 0:
            self.embedding.dim = int(self.embeddings.shape[1])
        self._build_keyword_index()
        self._build_chroma(path / "chroma")
        return True


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _normalize_chunk_payload(chunk: dict) -> dict:
    normalized = dict(chunk)
    normalized.setdefault("modality", "text")
    normalized.setdefault("parent_id", normalized.get("chunk_id", ""))
    normalized.setdefault("chunk_role", "parent")
    normalized.setdefault("metadata", {})
    return normalized


def ensure_parent_chunk(chunk: DocumentChunk, index: int) -> DocumentChunk:
    parent_id = chunk.parent_id or chunk.chunk_id or f"parent-{index}"
    metadata = dict(chunk.metadata)
    metadata.setdefault("parent_id", parent_id)
    metadata.setdefault("chunk_role", "parent")
    return DocumentChunk(
        chunk_id=parent_id,
        source=chunk.source,
        page=chunk.page,
        text=chunk.text,
        modality=chunk.modality,
        parent_id=parent_id,
        chunk_role="parent",
        metadata=metadata,
    )


def _tokenize(text: str) -> list[str]:
    tokens = [match.group(0).lower() for match in re.finditer(r"[A-Za-z][A-Za-z0-9_\-]+", text)]
    lowered = text.lower()
    for term in DOMAIN_TERMS:
        if term.lower() in lowered:
            tokens.append(term.lower())
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        tokens.append(sequence)
        for size in (2, 3, 4):
            tokens.extend(sequence[index : index + size] for index in range(0, max(0, len(sequence) - size + 1)))
    return tokens


def re_find_tokens(text: str):
    return re.finditer(r"[A-Za-z][A-Za-z0-9_\-]+|[\u4e00-\u9fff]{2,}", text)
