from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from .citation import CitationValidation, validate_citations
from .config import AppConfig
from .llm import AnswerGenerator
from .memory import MemoryStore
from .models import AgentAnswer, ConversationTurn, RetrievalResult
from .observability import persist_trace, start_timer, trace_event
from .rerank import Reranker
from .text_processing import DOMAIN_TERMS, domain_term_hits, is_casual_question
from .vector_store import VectorIndex


NODE_ANALYZE = "问题分析"
NODE_QUERY_REWRITE = "查询改写"
NODE_HYBRID_RETRIEVE = "混合检索"
NODE_RERANK = "重排"
NODE_GENERATE = "答案生成"
NODE_CITATION_VALIDATION = "引用校验"
NODE_CITATION_REPAIR = "引用修复"
NODE_SOURCE_VERIFICATION = "证据验证"
NODE_DIRECT = "直接回答"


@dataclass
class AgentState:
    question: str
    session_id: str = ""
    trace_id: str = ""
    history: list[ConversationTurn] = field(default_factory=list)
    memory_context: str = ""
    rewritten_query: str = ""
    query_variants: list[str] = field(default_factory=list)
    route: str = "unknown"
    need_retrieval: bool = True
    retrieved: list[RetrievalResult] = field(default_factory=list)
    reranked: list[RetrievalResult] = field(default_factory=list)
    answer: str = ""
    router_reason: str = ""
    router_confidence: float = 0.0
    verification_passed: bool = True
    verification_reason: str = ""
    retrieval_round: int = 0
    query_history: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    trace_events: list[dict] = field(default_factory=list)
    citation_validation: dict = field(default_factory=dict)
    citation_repair_attempts: int = 0


class MiningRagAgent:
    def __init__(
        self,
        vector_index: VectorIndex,
        config: AppConfig,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self.vector_index = vector_index
        self.config = config
        self.reranker = Reranker(config.rerank_model, config.rerank_backend)
        self.generator = AnswerGenerator(config)
        self.memory_store = memory_store or (MemoryStore(config.state_db_path) if config.memory_enabled else None)
        self._graph = self._build_graph()

    def invoke(
        self,
        question: str,
        history: list[ConversationTurn | dict] | None = None,
        session_id: str = "",
        memory_scope: str = "default",
    ) -> AgentAnswer:
        normalized_history = normalize_history(history or [], self.config.memory_max_turns)
        memory_context_text = ""
        if self.config.memory_enabled and self.memory_store is not None and session_id:
            memory_context = self.memory_store.get_context(
                session_id,
                question,
                max_messages=self.config.memory_max_turns * 2,
                long_term_k=self.config.memory_long_recall_k,
                scope=memory_scope,
            )
            normalized_history = normalize_history(
                [*memory_context.history, *normalized_history],
                self.config.memory_max_turns,
            )
            memory_context_text = memory_context.prompt_text()

        state = AgentState(
            question=question,
            session_id=session_id,
            trace_id=uuid.uuid4().hex,
            history=normalized_history,
            memory_context=memory_context_text,
        )
        if self._graph is not None:
            output = self._graph.invoke(state)
            state = output if isinstance(output, AgentState) else AgentState(**output)
        else:
            state = self._analyze_question(state)
            if state.need_retrieval:
                while True:
                    state = self._rewrite_query(state)
                    state = self._hybrid_retrieve(state)
                    state = self._rerank(state)
                    state = self._generate(state)
                    state = self._validate_citations(state)
                    while (
                        not state.citation_validation.get("passed", False)
                        and state.citation_repair_attempts < self.config.citation_repair_max_attempts
                    ):
                        state = self._repair_citations(state)
                        state = self._validate_citations(state)
                    state = self._verify_sources(state)
                    if state.verification_passed or state.retrieval_round >= self.config.max_retrieval_rounds:
                        break
            else:
                state = self._direct_answer(state)

        answer = AgentAnswer(
            answer=state.answer,
            route=state.route,
            need_retrieval=state.need_retrieval,
            sources=state.reranked,
            rewritten_query=state.rewritten_query,
            query_variants=state.query_variants,
            router_reason=state.router_reason,
            router_confidence=state.router_confidence,
            verification_passed=state.verification_passed,
            verification_reason=state.verification_reason,
            retrieval_rounds=state.retrieval_round,
            trace=state.trace,
            trace_id=state.trace_id,
            trace_events=state.trace_events,
            citation_validation=state.citation_validation,
        )
        if self.config.memory_enabled and self.memory_store is not None and session_id:
            memory_validated = not state.need_retrieval or (
                state.verification_passed and state.citation_validation.get("passed", False)
            )
            self.memory_store.append_turn(
                session_id,
                question,
                state.answer,
                validated=memory_validated,
                max_messages=self.config.memory_max_turns * 2,
                max_chars=self.config.memory_short_max_chars,
                summary_max_chars=self.config.memory_summary_max_chars,
            )
            snapshot = self.memory_store.snapshot(session_id, memory_scope)
            answer.memory = {
                "session_id": session_id,
                "short_term_messages": len(snapshot["short_term"]),
                "summary_chars": len(snapshot["summary"]),
                "long_term_count": len(snapshot["long_term"]),
                "turn_saved": memory_validated,
            }

        if self.config.trace_enabled:
            try:
                persist_trace(
                    self.config.trace_dir,
                    state.trace_id,
                    question=state.question,
                    route=state.route,
                    events=state.trace_events,
                    metadata={
                        "session_id": session_id,
                        "retrieval_rounds": state.retrieval_round,
                        "verification_passed": state.verification_passed,
                        "citation_validation": state.citation_validation,
                        "sources": [
                            {
                                "source": item.chunk.source,
                                "page": item.chunk.page,
                                "modality": item.chunk.modality,
                                "score": item.final_score,
                            }
                            for item in state.reranked
                        ],
                    },
                )
            except OSError:
                pass
        return answer

    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            return None

        try:
            graph = StateGraph(AgentState)
            graph.add_node(NODE_ANALYZE, self._analyze_question)
            graph.add_node(NODE_QUERY_REWRITE, self._rewrite_query)
            graph.add_node(NODE_HYBRID_RETRIEVE, self._hybrid_retrieve)
            graph.add_node(NODE_RERANK, self._rerank)
            graph.add_node(NODE_GENERATE, self._generate)
            graph.add_node(NODE_CITATION_VALIDATION, self._validate_citations)
            graph.add_node(NODE_CITATION_REPAIR, self._repair_citations)
            graph.add_node(NODE_SOURCE_VERIFICATION, self._verify_sources)
            graph.add_node(NODE_DIRECT, self._direct_answer)
            graph.set_entry_point(NODE_ANALYZE)
            graph.add_conditional_edges(
                NODE_ANALYZE,
                lambda s: NODE_QUERY_REWRITE if s.need_retrieval else NODE_DIRECT,
            )
            graph.add_edge(NODE_QUERY_REWRITE, NODE_HYBRID_RETRIEVE)
            graph.add_edge(NODE_HYBRID_RETRIEVE, NODE_RERANK)
            graph.add_edge(NODE_RERANK, NODE_GENERATE)
            graph.add_edge(NODE_GENERATE, NODE_CITATION_VALIDATION)
            graph.add_conditional_edges(
                NODE_CITATION_VALIDATION,
                lambda s: NODE_SOURCE_VERIFICATION
                if s.citation_validation.get("passed", False)
                or s.citation_repair_attempts >= self.config.citation_repair_max_attempts
                else NODE_CITATION_REPAIR,
            )
            graph.add_edge(NODE_CITATION_REPAIR, NODE_CITATION_VALIDATION)
            graph.add_conditional_edges(
                NODE_SOURCE_VERIFICATION,
                lambda s: END
                if s.verification_passed or s.retrieval_round >= self.config.max_retrieval_rounds
                else NODE_QUERY_REWRITE,
            )
            graph.add_edge(NODE_DIRECT, END)
            return graph.compile()
        except Exception:
            return None

    def _analyze_question(self, state: AgentState) -> AgentState:
        started = start_timer()
        question = state.question.strip()
        router_decision = self.generator.route_question(question, state.history, state.memory_context)
        if router_decision is not None:
            state.route = router_decision["route"]
            state.need_retrieval = bool(router_decision["need_retrieval"])
            state.rewritten_query = router_decision.get("rewritten_query") or question
            state.router_reason = router_decision.get("reason", "")
            state.router_confidence = float(router_decision.get("confidence", 0.0))
            self._record(
                state,
                NODE_ANALYZE,
                started,
                "大模型路由："
                f"路径={state.route}，是否检索={state.need_retrieval}，"
                f"置信度={state.router_confidence:.2f}，原因={state.router_reason}",
                router="llm",
                route=state.route,
                need_retrieval=state.need_retrieval,
                confidence=state.router_confidence,
                reason=state.router_reason,
            )
            return state

        if is_casual_question(question):
            state.route = "direct"
            state.need_retrieval = False
            state.router_reason = "规则降级：识别为闲聊或系统能力问题"
        elif domain_term_hits(question) > 0 or len(question) >= 12 or is_followup_question(question):
            state.route = "rag"
            state.need_retrieval = True
            state.router_reason = "规则降级：问题较专业、较长或像上下文追问，需要检索文献"
        else:
            state.route = "direct"
            state.need_retrieval = False
            state.router_reason = "规则降级：问题较短且没有明显专业检索意图"
        state.router_confidence = 0.45
        self._record(
            state,
            NODE_ANALYZE,
            started,
            f"规则路由：路径={state.route}，是否检索={state.need_retrieval}，原因={state.router_reason}",
            router="rule",
            route=state.route,
            need_retrieval=state.need_retrieval,
            confidence=state.router_confidence,
            reason=state.router_reason,
        )
        return state

    def _rewrite_query(self, state: AgentState) -> AgentState:
        started = start_timer()
        seed_question = state.rewritten_query or state.question
        variants = rewrite_queries(
            enrich_followup_question(seed_question, state.history),
            round_index=state.retrieval_round,
            max_queries=self.config.multi_query_count,
        )
        if state.query_history and variants[0] in state.query_history:
            variants.insert(0, broaden_query(variants[0]))
        state.query_variants = dedupe_preserve_order(variants)[: self.config.multi_query_count]
        state.rewritten_query = state.query_variants[0]
        state.query_history.append(state.rewritten_query)
        self._record(
            state,
            NODE_QUERY_REWRITE,
            started,
            f"查询改写：轮次={state.retrieval_round + 1}，候选查询={len(state.query_variants)}，"
            f"主查询={state.rewritten_query}",
            round=state.retrieval_round + 1,
            primary=state.rewritten_query,
            variants=state.query_variants,
        )
        return state

    def _hybrid_retrieve(self, state: AgentState) -> AgentState:
        started = start_timer()
        state.retrieval_round += 1
        query_results = [
            self.vector_index.search(
                query,
                top_k=self.config.top_k,
                keyword_top_k=self.config.keyword_top_k,
            )
            for query in (state.query_variants or [state.rewritten_query])
        ]
        state.retrieved = fuse_query_results(
            query_results,
            top_k=max(self.config.top_k, self.config.keyword_top_k),
        )
        vector_hits = sum(1 for item in state.retrieved if "vector_rank" in item.reason)
        keyword_hits = sum(1 for item in state.retrieved if "bm25_rank" in item.reason)
        self._record(
            state,
            NODE_HYBRID_RETRIEVE,
            started,
            f"混合检索：轮次={state.retrieval_round}，候选证据={len(state.retrieved)}，"
            f"查询数={len(query_results)}，向量命中={vector_hits}，关键词命中={keyword_hits}",
            round=state.retrieval_round,
            candidates=len(state.retrieved),
            query_count=len(query_results),
            vector_hits=vector_hits,
            keyword_hits=keyword_hits,
        )
        return state

    def _rerank(self, state: AgentState) -> AgentState:
        started = start_timer()
        state.reranked = self.reranker.rerank(state.rewritten_query, state.retrieved, self.config.rerank_top_k)
        if state.reranked and state.reranked[0].final_score < self.config.min_relevance:
            message = "重排：低置信度"
        else:
            message = f"重排：保留证据={len(state.reranked)}"
        self._record(
            state,
            NODE_RERANK,
            started,
            message,
            kept=len(state.reranked),
            top_score=state.reranked[0].final_score if state.reranked else 0.0,
            model=self.config.rerank_model or "none",
            backend=self.reranker.active_backend,
        )
        return state

    def _generate(self, state: AgentState) -> AgentState:
        started = start_timer()
        state.answer = self.generator.generate(
            state.question,
            state.reranked,
            state.history,
            state.memory_context,
        )
        self._record(
            state,
            NODE_GENERATE,
            started,
            "答案生成：完成",
            backend=self.generator.backend_name,
            answer_chars=len(state.answer),
            evidence_count=len(state.reranked),
        )
        return state

    def _validate_citations(self, state: AgentState) -> AgentState:
        started = start_timer()
        if not self.config.citation_validation_enabled:
            validation = CitationValidation(passed=True, reason="引用校验已关闭")
        else:
            validation = validate_citations(
                state.answer,
                state.reranked,
                min_coverage=self.config.citation_min_coverage,
            )
        state.citation_validation = validation.to_dict()
        self._record(
            state,
            NODE_CITATION_VALIDATION,
            started,
            f"引用校验：通过={validation.passed}，原因={validation.reason}",
            **state.citation_validation,
        )
        return state

    def _repair_citations(self, state: AgentState) -> AgentState:
        started = start_timer()
        state.citation_repair_attempts += 1
        state.answer = self.generator.repair_citations(
            state.question,
            state.answer,
            state.reranked,
            state.citation_validation.get("reason", ""),
        )
        self._record(
            state,
            NODE_CITATION_REPAIR,
            started,
            f"引用修复：次数={state.citation_repair_attempts}",
            attempt=state.citation_repair_attempts,
            backend=self.generator.backend_name,
        )
        return state

    def _verify_sources(self, state: AgentState) -> AgentState:
        started = start_timer()
        state.verification_passed, state.verification_reason = verify_source_coverage(
            question=state.question,
            results=state.reranked,
            min_relevance=self.config.min_relevance,
            min_keyword_coverage=self.config.min_keyword_coverage,
            min_complex_sources=self.config.min_complex_sources,
        )
        self._record(
            state,
            NODE_SOURCE_VERIFICATION,
            started,
            f"证据验证：通过={state.verification_passed}，原因={state.verification_reason}",
            passed=state.verification_passed,
            reason=state.verification_reason,
            retrieval_round=state.retrieval_round,
        )
        if not state.verification_passed and state.retrieval_round >= self.config.max_retrieval_rounds:
            state.answer = (
                f"{state.answer}\n\n"
                f"> 证据校验提示：{state.verification_reason}。已完成 "
                f"{state.retrieval_round} 轮检索，建议上传更多论文或缩小问题范围。"
            )
        if (
            not state.citation_validation.get("passed", True)
            and state.retrieval_round >= self.config.max_retrieval_rounds
        ):
            state.answer = (
                f"{state.answer}\n\n"
                f"> 引用校验提示：{state.citation_validation.get('reason', '引用未通过校验')}。"
                "请以右侧经过校验的来源列表为准。"
            )
        return state

    def _direct_answer(self, state: AgentState) -> AgentState:
        started = start_timer()
        state.answer = self.generator.direct_answer(state.question)
        self._record(
            state,
            NODE_DIRECT,
            started,
            "直接回答：完成",
            backend=self.generator.backend_name,
            answer_chars=len(state.answer),
        )
        return state

    def _record(
        self,
        state: AgentState,
        node: str,
        started: float,
        message: str,
        **details,
    ) -> None:
        state.trace.append(message)
        trace_event(state.trace_events, node, started, **details)


def rewrite_query(question: str, round_index: int = 0) -> str:
    return rewrite_queries(question, round_index=round_index, max_queries=1)[0]


def rewrite_queries(question: str, round_index: int = 0, max_queries: int = 4) -> list[str]:
    q = re.sub(r"\s+", " ", question).strip()
    expansions = {
        "多金属结核": "多金属结核 polymetallic nodules manganese nodules",
        "富钴结壳": "富钴结壳 cobalt-rich crust ferromanganese crust",
        "热液硫化物": "热液硫化物 seafloor massive sulfide hydrothermal sulfide",
        "赋存环境": "赋存环境 occurrence environment geological setting",
        "控制因素": "控制因素 controlling factors formation mechanism",
        "识别指标": "识别指标 recognition indicators geochemical indicators",
    }
    additions = [value for key, value in expansions.items() if key in q]
    intent_terms = infer_intent_terms(q)
    variants = [
        f"{q} {' '.join(additions)}".strip(),
        f"{q} {' '.join(intent_terms)}".strip(),
        f"{q} source evidence mechanism indicator distribution geochemistry mineralogy".strip(),
    ]
    if round_index > 0:
        variants.extend(
            [
                f"{q} marine minerals deep sea mining metallogenesis",
                f"{q} geochemical indicators geological setting controlling factors",
                f"{q} formation environment occurrence distribution resource assessment",
            ]
        )
    return dedupe_preserve_order([variant for variant in variants if variant])[:max_queries]


def broaden_query(query: str) -> str:
    return (
        f"{query} marine minerals deep sea mineral resources "
        "environmental control geochemistry mineralization evidence"
    )


def infer_intent_terms(question: str) -> list[str]:
    mapping = {
        "因素": ["控制因素", "controlling factors", "formation mechanism"],
        "机制": ["形成机制", "metallogenesis", "formation mechanism"],
        "环境": ["赋存环境", "geological setting", "redox", "sedimentation"],
        "指标": ["识别指标", "geochemical indicators", "mineralogical indicators"],
        "图": ["figure", "image", "diagram", "caption"],
        "表": ["table", "data", "sample", "element concentration"],
        "稀土": ["REE", "rare earth elements", "Ce anomaly", "Y/Ho"],
        "元素": ["geochemistry", "element concentration", "major trace elements"],
    }
    terms: list[str] = []
    for key, values in mapping.items():
        if key in question:
            terms.extend(values)
    return terms or ["marine mineral evidence", "source text", "research paper"]


def normalize_history(history: list[ConversationTurn | dict], max_turns: int) -> list[ConversationTurn]:
    normalized: list[ConversationTurn] = []
    for turn in history[-max(1, max_turns * 2) :]:
        if isinstance(turn, ConversationTurn):
            normalized.append(turn)
            continue
        role = str(turn.get("role", "")).strip()
        content = str(turn.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            normalized.append(ConversationTurn(role=role, content=content))
    return normalized


def is_followup_question(question: str) -> bool:
    return bool(re.search(r"(继续|刚才|上面|上一个|这些|它们|该文|这篇|前面|上述|进一步|再解释)", question))


def enrich_followup_question(question: str, history: list[ConversationTurn]) -> str:
    if not history or not is_followup_question(question):
        return question
    if "上下文问题:" in question:
        return question
    recent_user_questions = [turn.content for turn in history if turn.role == "user"][-2:]
    if not recent_user_questions:
        return question
    return f"{question} 上下文问题: {' '.join(recent_user_questions)}"


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        normalized = re.sub(r"\s+", " ", item).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def fuse_query_results(query_results: list[list[RetrievalResult]], top_k: int) -> list[RetrievalResult]:
    fused: dict[str, RetrievalResult] = {}
    k = 60.0
    for query_index, results in enumerate(query_results, start=1):
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.chunk_id
            contribution = 1.0 / (k + rank)
            if chunk_id not in fused:
                fused[chunk_id] = RetrievalResult(
                    chunk=result.chunk,
                    score=0.0,
                    vector_score=result.vector_score,
                    keyword_score=result.keyword_score,
                    retrieval_method="multi_query_hybrid",
                    reason="",
                )
            fused_item = fused[chunk_id]
            fused_item.score += contribution
            fused_item.vector_score = max(fused_item.vector_score, result.vector_score)
            fused_item.keyword_score = max(fused_item.keyword_score, result.keyword_score)
            fused_item.reason = append_reason(fused_item.reason, f"q{query_index}:{result.reason}")
    return sorted(fused.values(), key=lambda item: item.score, reverse=True)[:top_k]


def append_reason(existing: str, addition: str) -> str:
    if not addition:
        return existing
    if not existing:
        return addition
    if addition in existing:
        return existing
    return f"{existing}; {addition}"


def verify_source_coverage(
    question: str,
    results: list[RetrievalResult],
    min_relevance: float,
    min_keyword_coverage: float = 0.28,
    min_complex_sources: int = 2,
) -> tuple[bool, str]:
    if not results:
        return False, "没有检索到可用文献片段"

    top_score = results[0].final_score
    if top_score < min_relevance:
        return False, f"最高相关性分数 {top_score:.3f} 低于阈值 {min_relevance:.3f}"

    question_terms = set(_terms(question))
    covered_terms: set[str] = set()
    for result in results[:3]:
        covered_terms.update(question_terms & set(_terms(result.chunk.text)))
    coverage = len(covered_terms) / max(1, len(question_terms))
    if question_terms and coverage < min_keyword_coverage:
        return False, f"问题关键词覆盖不足，coverage={coverage:.2f}"

    complex_question = any(word in question for word in ["哪些", "因素", "对比", "机制", "指标", "为什么", "如何"])
    if complex_question and len(results) < min_complex_sources:
        return False, f"复杂科研问题证据偏少，当前 {len(results)} 个，要求至少 {min_complex_sources} 个"

    source_count = len({result.chunk.source for result in results})
    modalities = sorted({result.chunk.modality for result in results})
    return True, f"证据充分，来源数={source_count}，模态={','.join(modalities)}，关键词覆盖={coverage:.2f}"


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
