from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import MiningRagAgent
from src.config import AppConfig
from src.llm import AnswerGenerator
from src.models import AgentAnswer, RetrievalResult
from src.vector_store import EmbeddingProvider, VectorIndex


DEFAULT_HIT_KS = (1, 3, 5)
REFUSAL_PATTERNS = (
    "证据不足",
    "没有找到足够",
    "无法可靠回答",
    "无法回答",
    "不能确定",
    "not enough evidence",
    "insufficient evidence",
)


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
    return cases


def evaluate(
    cases: list[dict[str, Any]],
    config: AppConfig,
    hit_ks: tuple[int, ...] = DEFAULT_HIT_KS,
    use_llm_judge: bool = False,
) -> dict[str, Any]:
    index = VectorIndex(EmbeddingProvider(config.embedding_backend), config.vector_backend)
    if not index.load(config.index_dir):
        raise RuntimeError(f"Index not found: {config.index_dir}. Build the index in Streamlit or FastAPI first.")
    agent = MiningRagAgent(index, config)
    judge = AnswerGenerator(config) if use_llm_judge else None

    rows = []
    for case in cases:
        question = str(case["question"])
        started = time.perf_counter()
        answer = agent.invoke(question)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        metrics = evaluate_answer(case, answer, hit_ks=hit_ks)
        metrics["llm_judge"] = (
            run_llm_judge(judge, case, answer)
            if judge is not None and judge.backend_name != "fallback"
            else None
        )
        rows.append(
            {
                "question": question,
                "passed": metrics["passed"],
                "latency_ms": latency_ms,
                "answer_preview": answer.answer[:500],
                "route": answer.route,
                "verification_passed": answer.verification_passed,
                "verification_reason": answer.verification_reason,
                "retrieval_rounds": answer.retrieval_rounds,
                **metrics,
                "top_sources": serialize_top_sources(answer.sources, top_n=max(hit_ks) if hit_ks else 5),
            }
        )

    return aggregate(rows, hit_ks)


def evaluate_answer(case: dict[str, Any], answer: AgentAnswer, hit_ks: tuple[int, ...] = DEFAULT_HIT_KS) -> dict[str, Any]:
    expected_sources = normalize_expected_sources(case.get("expected_sources", []))
    expected_terms = [str(item).lower() for item in case.get("expected_terms", []) if str(item).strip()]
    min_term_recall = float(case.get("min_term_recall", 0.6))
    should_refuse = case.get("should_refuse")

    source_text = combined_source_text(answer.sources)
    answer_text = answer.answer.lower()
    source_hit = source_hit_at_k(answer.sources, expected_sources, len(answer.sources)) if expected_sources else True
    hit_at = {
        f"hit_at_{k}": source_hit_at_k(answer.sources, expected_sources, k) if expected_sources else True
        for k in hit_ks
    }
    first_rank = first_relevant_rank(answer.sources, expected_sources)
    mrr = 0.0 if expected_sources and first_rank is None else 1.0 / max(1, first_rank or 1)
    ndcg = ndcg_at_k(answer.sources, expected_sources, max(hit_ks) if hit_ks else len(answer.sources))

    term_hits = sum(1 for term in expected_terms if term in source_text)
    term_recall = term_hits / len(expected_terms) if expected_terms else 1.0
    answer_term_hits = sum(1 for term in expected_terms if term in answer_text)
    answer_term_recall = answer_term_hits / len(expected_terms) if expected_terms else 1.0

    citation_accuracy = citation_accuracy_score(answer.answer, answer.sources, expected_terms)
    faithfulness = faithfulness_score(answer.answer, source_text, expected_terms)
    refusal_correct = refusal_accuracy(answer, should_refuse)

    retrieval_ok = bool(source_hit and term_recall >= min_term_recall)
    answer_ok = bool(answer_term_recall >= float(case.get("min_answer_term_recall", 0.0)))
    verification_ok = bool(answer.verification_passed) if not should_refuse else not answer.verification_passed or is_refusal(answer.answer)
    passed = bool(retrieval_ok and answer_ok and verification_ok and refusal_correct is not False)

    return {
        "source_hit": source_hit,
        **hit_at,
        "first_relevant_rank": first_rank,
        "mrr": mrr,
        "ndcg": ndcg,
        "term_recall": term_recall,
        "answer_term_recall": answer_term_recall,
        "citation_accuracy": citation_accuracy,
        "faithfulness": faithfulness,
        "refusal_correct": refusal_correct,
        "refusal_score": None if refusal_correct is None else float(refusal_correct),
        "passed": passed,
    }


def normalize_expected_sources(raw_sources: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_sources, list):
        raw_sources = [raw_sources]
    normalized: list[dict[str, Any]] = []
    for item in raw_sources:
        if isinstance(item, str):
            text = item.strip().lower()
            if text:
                normalized.append({"source": text, "page": None})
        elif isinstance(item, dict):
            source = str(item.get("source", "")).strip().lower()
            page = normalize_pages(item.get("page"), item.get("pages"))
            if source or page:
                normalized.append({"source": source, "page": page})
    return normalized


def normalize_pages(page: Any, pages: Any) -> set[int] | None:
    raw = pages if pages is not None else page
    if raw is None or raw == "":
        return None
    if not isinstance(raw, list):
        raw = [raw]
    normalized: set[int] = set()
    for item in raw:
        try:
            normalized.add(int(item))
        except Exception:
            continue
    return normalized or None


def source_hit_at_k(results: list[RetrievalResult], expected_sources: list[dict[str, Any]], k: int) -> bool:
    if not expected_sources:
        return True
    return any(result_matches_expected(result, expected_sources) for result in results[:k])


def first_relevant_rank(results: list[RetrievalResult], expected_sources: list[dict[str, Any]]) -> int | None:
    if not expected_sources:
        return 1 if results else None
    for rank, result in enumerate(results, start=1):
        if result_matches_expected(result, expected_sources):
            return rank
    return None


def result_matches_expected(result: RetrievalResult, expected_sources: list[dict[str, Any]]) -> bool:
    source = result.chunk.source.lower()
    page = result.chunk.page
    for expected in expected_sources:
        expected_source = str(expected.get("source", "")).lower()
        expected_pages = expected.get("page")
        source_ok = not expected_source or expected_source in source or source in expected_source
        page_ok = expected_pages is None or page in expected_pages
        if source_ok and page_ok:
            return True
    return False


def ndcg_at_k(results: list[RetrievalResult], expected_sources: list[dict[str, Any]], k: int) -> float:
    if not expected_sources:
        return 1.0 if results else 0.0
    gains = [1.0 if result_matches_expected(result, expected_sources) else 0.0 for result in results[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    idcg = sum(1.0 / math.log2(index + 2) for index in range(min(len(expected_sources), k)))
    return dcg / idcg if idcg else 0.0


def citation_accuracy_score(answer_text: str, results: list[RetrievalResult], expected_terms: list[str]) -> float:
    citations = [int(item) for item in re.findall(r"\[(\d+)\]", answer_text)]
    if not citations:
        return 0.0 if expected_terms else 1.0
    supported = 0
    for citation in citations:
        index = citation - 1
        if 0 <= index < len(results):
            evidence = results[index].chunk.text.lower()
            if not expected_terms or any(term in evidence for term in expected_terms):
                supported += 1
    return supported / max(1, len(citations))


def faithfulness_score(answer_text: str, source_text: str, expected_terms: list[str]) -> float:
    if is_refusal(answer_text):
        return 1.0
    if expected_terms:
        answer_terms = [term for term in expected_terms if term in answer_text.lower()]
        if not answer_terms:
            return 0.0
        supported = sum(1 for term in answer_terms if term in source_text)
        return supported / max(1, len(answer_terms))

    answer_tokens = content_tokens(answer_text)
    if not answer_tokens:
        return 0.0
    source_tokens = set(content_tokens(source_text))
    return min(1.0, sum(1 for token in answer_tokens if token in source_tokens) / max(1, len(answer_tokens)))


def refusal_accuracy(answer: AgentAnswer, should_refuse: Any) -> bool | None:
    if should_refuse is None:
        return None
    refused = is_refusal(answer.answer) or (not answer.verification_passed and not answer.sources)
    return refused if bool(should_refuse) else not refused


def is_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in REFUSAL_PATTERNS)


def combined_source_text(results: list[RetrievalResult]) -> str:
    return "\n".join(
        f"{result.chunk.source} p.{result.chunk.page} {result.chunk.modality}\n{result.chunk.text}"
        for result in results
    ).lower()


def content_tokens(text: str) -> list[str]:
    tokens = [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", text)]
    tokens.extend(re.findall(r"[\u4e00-\u9fff]{2,}", text))
    stopwords = {"the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "以及", "因此", "可以"}
    return [token for token in tokens if token not in stopwords]


def serialize_top_sources(results: list[RetrievalResult], top_n: int) -> list[dict[str, Any]]:
    return [
        {
            "source": result.chunk.source,
            "page": result.chunk.page,
            "modality": result.chunk.modality,
            "score": round(result.final_score, 4),
            "retrieval_method": result.retrieval_method,
        }
        for result in results[:top_n]
    ]


def run_llm_judge(judge: AnswerGenerator, case: dict[str, Any], answer: AgentAnswer) -> dict[str, Any] | None:
    evidence = "\n\n".join(
        f"[{index}] source={result.chunk.source}, page={result.chunk.page}\n{result.chunk.text[:1200]}"
        for index, result in enumerate(answer.sources[:5], start=1)
    )
    prompt = f"""你是 RAG 评测员。请只输出 JSON，不要输出 Markdown。
字段：correctness、faithfulness、citation_accuracy、answer_relevance 均为 0 到 5；should_refuse 为 true/false；reason 为中文简短说明。

问题：{case.get("question")}
期望关键词：{case.get("expected_terms", [])}
期望来源：{case.get("expected_sources", [])}
系统答案：{answer.answer}
检索证据：{evidence or "无"}
"""
    try:
        raw = judge._chat(prompt)
        match = re.search(r"\{.*\}", raw, re.S)
        return json.loads(match.group(0) if match else raw)
    except Exception as exc:
        return {"error": str(exc)}


def aggregate(rows: list[dict[str, Any]], hit_ks: tuple[int, ...]) -> dict[str, Any]:
    refusal_rows = [row for row in rows if row["refusal_correct"] is not None]
    judge_rows = [row for row in rows if isinstance(row.get("llm_judge"), dict) and "error" not in row["llm_judge"]]
    judge_score = None
    if judge_rows:
        judge_score = mean(
            mean(
                [
                    float(row["llm_judge"].get("correctness", 0)),
                    float(row["llm_judge"].get("faithfulness", 0)),
                    float(row["llm_judge"].get("citation_accuracy", 0)),
                    float(row["llm_judge"].get("answer_relevance", 0)),
                ]
            )
            for row in judge_rows
        )

    result: dict[str, Any] = {
        "total": len(rows),
        "accuracy": mean_bool(row["passed"] for row in rows),
        "source_hit_rate": mean_bool(row["source_hit"] for row in rows),
        "avg_term_recall": mean(row["term_recall"] for row in rows),
        "avg_answer_term_recall": mean(row["answer_term_recall"] for row in rows),
        "mrr": mean(row["mrr"] for row in rows),
        "ndcg": mean(row["ndcg"] for row in rows),
        "citation_accuracy": mean(row["citation_accuracy"] for row in rows),
        "faithfulness": mean(row["faithfulness"] for row in rows),
        "refusal_accuracy": mean_bool(row["refusal_correct"] for row in refusal_rows) if refusal_rows else None,
        "verification_pass_rate": mean_bool(row["verification_passed"] for row in rows),
        "avg_retrieval_rounds": mean(row["retrieval_rounds"] for row in rows),
        "avg_latency_ms": mean(row["latency_ms"] for row in rows),
        "llm_judge_score": judge_score,
        "rows": rows,
    }
    for k in hit_ks:
        result[f"hit_at_{k}"] = mean_bool(row[f"hit_at_{k}"] for row in rows)
    return result


def mean(values) -> float:
    items = list(values)
    return sum(items) / max(1, len(items))


def mean_bool(values) -> float:
    items = [bool(value) for value in values]
    return sum(1 for item in items if item) / max(1, len(items))


def parse_hit_ks(raw: str) -> tuple[int, ...]:
    values = [max(1, int(item.strip())) for item in raw.split(",") if item.strip()]
    return tuple(sorted(set(values))) or DEFAULT_HIT_KS


def print_summary(result: dict[str, Any], hit_ks: tuple[int, ...]) -> None:
    print(f"Total: {result['total']}")
    print(f"Accuracy: {result['accuracy']:.2%}")
    for k in hit_ks:
        print(f"Hit@{k}: {result[f'hit_at_{k}']:.2%}")
    print(f"MRR: {result['mrr']:.4f}")
    print(f"nDCG: {result['ndcg']:.4f}")
    print(f"Source hit rate: {result['source_hit_rate']:.2%}")
    print(f"Average term recall: {result['avg_term_recall']:.2%}")
    print(f"Answer term recall: {result['avg_answer_term_recall']:.2%}")
    print(f"Citation accuracy: {result['citation_accuracy']:.2%}")
    print(f"Faithfulness: {result['faithfulness']:.2%}")
    print("Refusal accuracy: n/a" if result["refusal_accuracy"] is None else f"Refusal accuracy: {result['refusal_accuracy']:.2%}")
    print(f"Verification pass rate: {result['verification_pass_rate']:.2%}")
    print(f"Average retrieval rounds: {result['avg_retrieval_rounds']:.2f}")
    print(f"Average latency: {result['avg_latency_ms']:.2f} ms")
    if result["llm_judge_score"] is not None:
        print(f"LLM judge score: {result['llm_judge_score']:.2f}/5")
    for row in result["rows"]:
        mark = "PASS" if row["passed"] else "FAIL"
        rank = row["first_relevant_rank"] if row["first_relevant_rank"] is not None else "-"
        print(
            f"- {mark}: {row['question']} | rank={rank} | term_recall={row['term_recall']:.2%} | "
            f"citation={row['citation_accuracy']:.2%} | faithfulness={row['faithfulness']:.2%} | "
            f"sources={row['top_sources'][:3]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval, citation, faithfulness, and refusal accuracy.")
    parser.add_argument("--cases", default="evaluation/sample_eval.jsonl", help="JSONL evaluation set.")
    parser.add_argument("--index-dir", default=None, help="Override index directory.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    parser.add_argument("--hit-ks", default="1,3,5", help="Comma-separated K values for Hit@K.")
    parser.add_argument("--llm-judge", action="store_true", help="Use configured LLM backend as an optional judge.")
    args = parser.parse_args()

    config = AppConfig.from_env()
    if args.index_dir:
        config.index_dir = Path(args.index_dir)
    hit_ks = parse_hit_ks(args.hit_ks)
    result = evaluate(load_cases(Path(args.cases)), config, hit_ks=hit_ks, use_llm_judge=args.llm_judge)
    print_summary(result, hit_ks)

    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
