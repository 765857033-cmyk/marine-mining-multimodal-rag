from __future__ import annotations

import os

from .models import RetrievalResult


class Reranker:
    def __init__(self, model_name: str = "", backend: str = "auto") -> None:
        self.backend = (backend or os.getenv("RERANK_BACKEND", "auto")).lower().replace("_", "-")
        self.model_name = model_name or os.getenv("RERANK_MODEL", "")
        self._model = None
        self.active_backend = "none"
        self._init_model()

    def _init_model(self) -> None:
        if self.backend in {"none", "off", "disabled"}:
            self.active_backend = "none"
            return
        if self.backend == "rule":
            raise ValueError("项目已取消规则 Rerank 兜底，请使用 RERANK_BACKEND=cross-encoder、bge 或 none。")
        if not self.model_name:
            if self.backend == "auto":
                self.active_backend = "none"
                return
            raise ValueError("启用模型 Rerank 时必须配置 RERANK_MODEL。")

        if self.backend in {"auto", "bge", "bge-reranker"} and "bge-reranker" in self.model_name.lower():
            try:
                from FlagEmbedding import FlagReranker

                self._model = FlagReranker(self.model_name, use_fp16=False)
                self.active_backend = "bge"
                return
            except Exception as exc:
                if self.backend in {"bge", "bge-reranker"}:
                    raise RuntimeError(f"BGE Reranker 加载失败：{self.model_name}") from exc

        if self.backend in {"auto", "cross-encoder", "crossencoder"}:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
                self.active_backend = "cross-encoder"
                return
            except Exception as exc:
                raise RuntimeError(f"CrossEncoder Reranker 加载失败：{self.model_name}") from exc

        raise ValueError(f"不支持的 Rerank 后端：{self.backend}")

    def rerank(self, question: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        if not results:
            return []
        if self.active_backend == "none":
            for result in results:
                retrieval_reason = result.reason
                rerank_reason = "rerank=none"
                result.reason = f"{retrieval_reason}; {rerank_reason}" if retrieval_reason else rerank_reason
            return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

        pairs = [(question, result.chunk.text) for result in results]
        if self.active_backend == "bge":
            scores = self._model.compute_score(pairs)
        else:
            scores = self._model.predict(pairs)
        if isinstance(scores, (float, int)):
            scores = [scores]
        for result, score in zip(results, scores):
            retrieval_reason = result.reason
            result.rerank_score = float(score)
            rerank_reason = f"rerank={self.active_backend}, model={self.model_name}"
            result.reason = f"{retrieval_reason}; {rerank_reason}" if retrieval_reason else rerank_reason
        return sorted(results, key=lambda item: item.final_score, reverse=True)[:top_k]
