from __future__ import annotations

import os
import re

from .models import RetrievalResult
from .text_processing import DOMAIN_TERMS, domain_term_hits


class Reranker:
    def __init__(self, model_name: str = "") -> None:
        self.model_name = model_name or os.getenv("RERANK_MODEL", "")
        self._model = None
        if self.model_name:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception:
                self._model = None

    def rerank(self, question: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        if not results:
            return []
        if self._model is not None:
            pairs = [(question, result.chunk.text) for result in results]
            scores = self._model.predict(pairs)
            for result, score in zip(results, scores):
                retrieval_reason = result.reason
                result.rerank_score = float(score)
                result.reason = f"{retrieval_reason}; rerank=cross-encoder" if retrieval_reason else "rerank=cross-encoder"
        else:
            query_terms = set(_terms(question))
            for result in results:
                retrieval_reason = result.reason
                chunk_terms = set(_terms(result.chunk.text))
                overlap = len(query_terms & chunk_terms) / max(1, len(query_terms))
                heading = str(result.chunk.metadata.get("heading", "")) + " " + str(result.chunk.metadata.get("caption", ""))
                heading_terms = set(_terms(heading))
                heading_overlap = len(query_terms & heading_terms) / max(1, len(query_terms))
                domain_boost = min(domain_term_hits(result.chunk.text), 6) * 0.025
                phrase_boost = 0.08 if question[:20] and question[:20] in result.chunk.text else 0.0
                modality_boost = {"table": 0.035, "image": 0.025, "page_snapshot": 0.015}.get(
                    result.chunk.modality, 0.0
                )
                bm25_signal = min(result.keyword_score, 8.0) / 8.0
                vector_signal = max(-1.0, min(result.vector_score, 1.0))
                result.rerank_score = (
                    0.34 * result.score
                    + 0.22 * overlap
                    + 0.12 * heading_overlap
                    + 0.14 * bm25_signal
                    + 0.08 * vector_signal
                    + domain_boost
                    + phrase_boost
                    + modality_boost
                )
                rerank_reason = (
                    f"rerank_overlap={overlap:.2f}, heading_overlap={heading_overlap:.2f}, "
                    f"bm25_signal={bm25_signal:.2f}, domain_boost={domain_boost:.2f}, "
                    f"modality_boost={modality_boost:.2f}"
                )
                result.reason = f"{retrieval_reason}; {rerank_reason}" if retrieval_reason else rerank_reason
        return sorted(results, key=lambda item: item.final_score, reverse=True)[:top_k]


def _terms(text: str) -> list[str]:
    tokens = [m.group(0).lower() for m in re.finditer(r"[A-Za-z][A-Za-z0-9_\-]+", text)]
    lowered = text.lower()
    for term in DOMAIN_TERMS:
        if term.lower() in lowered:
            tokens.append(term.lower())
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        for size in (2, 3, 4):
            tokens.extend(sequence[index : index + size] for index in range(0, max(0, len(sequence) - size + 1)))
    return tokens
