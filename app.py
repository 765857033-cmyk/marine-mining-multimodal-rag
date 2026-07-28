from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.agent import MiningRagAgent
from src.config import AppConfig
from src.feedback import FeedbackStore
from src.memory import MemoryStore
from src.pdf_ingest import ingest_pdfs
from src.vector_store import EmbeddingProvider, VectorIndex


load_dotenv()

st.set_page_config(page_title="海洋矿产 Agentic RAG", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner=False)
def load_index(index_dir: str, embedding_backend: str, vector_backend: str) -> VectorIndex:
    embedding = EmbeddingProvider(embedding_backend)
    index = VectorIndex(embedding, vector_backend)
    index.load(Path(index_dir))
    return index


@st.cache_resource(show_spinner=False)
def load_memory_store(db_path: str) -> MemoryStore:
    return MemoryStore(Path(db_path))


@st.cache_resource(show_spinner=False)
def load_feedback_store(db_path: str, bad_cases_path: str) -> FeedbackStore:
    return FeedbackStore(Path(db_path), Path(bad_cases_path))


def rebuild_index(files, config: AppConfig) -> VectorIndex:
    config.upload_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths: list[Path] = []
    for uploaded in files:
        target = config.upload_dir / uploaded.name
        target.write_bytes(uploaded.getbuffer())
        pdf_paths.append(target)

    chunks = ingest_pdfs(pdf_paths, config)
    embedding = EmbeddingProvider(config.embedding_backend)
    index = VectorIndex(embedding, config.vector_backend)
    index.build(
        chunks,
        parent_child_enabled=config.parent_child_enabled,
        child_chunk_size=config.child_chunk_size,
        child_chunk_overlap=config.child_chunk_overlap,
    )
    index.save(config.index_dir)
    load_index.clear()
    return index


def render_sources(results) -> None:
    if not results:
        st.info("没有检索来源。")
        return

    rows = []
    for result in results:
        rows.append(
            {
                "source": result.chunk.source,
                "page": result.chunk.page,
                "modality": result.chunk.modality,
                "method": result.retrieval_method,
                "score": round(result.score, 4),
                "vector_score": round(result.vector_score, 4),
                "keyword_score": round(result.keyword_score, 4),
                "rerank_score": round(result.final_score, 4),
                "reason": result.reason,
                "heading": result.chunk.metadata.get("heading", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    for i, result in enumerate(results, start=1):
        title = f"[{i}] {result.chunk.source} p.{result.chunk.page} | {result.chunk.modality}"
        with st.expander(title):
            asset_path = result.chunk.metadata.get("asset_path", "")
            if asset_path and Path(asset_path).exists():
                st.image(asset_path, caption=result.chunk.metadata.get("caption", ""))
            matched_child = result.chunk.metadata.get("matched_child_text", "")
            if matched_child:
                st.caption("命中的子片段")
                st.code(matched_child, language="text")
                st.caption("回召的父文档上下文")
            st.write(result.chunk.text)


def evidence_payload(results) -> list[dict]:
    return [
        {
            "source": result.chunk.source,
            "page": result.chunk.page,
            "modality": result.chunk.modality,
            "text": result.chunk.text[:1200],
            "score": result.final_score,
        }
        for result in results
    ]


def main() -> None:
    config = AppConfig.from_env()
    memory_store = load_memory_store(str(config.state_db_path))
    feedback_store = load_feedback_store(str(config.state_db_path), str(config.feedback_bad_cases_path))
    session_id = st.session_state.setdefault("session_id", uuid.uuid4().hex)

    with st.sidebar:
        st.title("Agentic RAG")
        st.caption("海洋矿产科研文献智能问答")

        config.llm_backend = st.selectbox(
            "回答模型后端",
            options=["auto", "openai", "ocean-gpt-api", "openai-compatible", "none"],
            index=["auto", "openai", "ocean-gpt-api", "openai-compatible", "none"].index(config.llm_backend)
            if config.llm_backend in {"auto", "openai", "ocean-gpt-api", "openai-compatible", "none"}
            else 0,
        )
        config.llm_router_enabled = st.checkbox("启用大模型路由", value=config.llm_router_enabled)
        config.memory_enabled = st.checkbox("启用上下文记忆", value=config.memory_enabled)
        config.memory_max_turns = st.slider("短期记忆轮数", 1, 12, config.memory_max_turns)
        if st.button("清空当前会话记忆", use_container_width=True):
            memory_store.clear_session(session_id)
            st.session_state.pop("last_answer", None)
            st.success("当前会话短期记忆和摘要记忆已清空。")

        with st.expander("长期记忆管理", expanded=False):
            long_memory_text = st.text_area(
                "稳定事实、偏好或项目约束",
                key="long_memory_text",
                height=80,
                placeholder="例如：回答必须使用中文，并优先引用海洋矿产论文。",
            )
            if st.button("保存长期记忆", use_container_width=True):
                try:
                    memory_store.add_long_term(long_memory_text, capacity=config.memory_long_capacity)
                    st.success("长期记忆已保存。")
                except ValueError as exc:
                    st.warning(str(exc))
            long_memories = memory_store.list_long_term()
            if long_memories:
                selected_memory = st.selectbox(
                    "删除长期记忆",
                    options=long_memories,
                    format_func=lambda item: item["content"][:45],
                )
                if st.button("删除选中记忆", use_container_width=True):
                    memory_store.delete_long_term(selected_memory["memory_id"])
                    st.rerun()

        st.subheader("知识库")
        uploads = st.file_uploader("上传论文 PDF", type=["pdf"], accept_multiple_files=True)
        config.parser_backend = st.selectbox(
            "PDF 解析后端",
            options=["mineru", "pymupdf", "auto"],
            index=["mineru", "pymupdf", "auto"].index(config.parser_backend)
            if config.parser_backend in {"mineru", "pymupdf", "auto"}
            else 0,
        )
        config.parser_fallback = st.checkbox("解析失败时回退 PyMuPDF", value=config.parser_fallback)
        if config.parser_backend in {"mineru", "auto"}:
            mineru_backend_options = ["pipeline", "hybrid-engine", "vlm-engine", "vlm-http-client", "hybrid-http-client", ""]
            config.mineru_backend = st.selectbox(
                "MinerU backend",
                options=mineru_backend_options,
                index=mineru_backend_options.index(config.mineru_backend)
                if config.mineru_backend in set(mineru_backend_options)
                else 0,
            )
            config.mineru_method = st.selectbox(
                "MinerU method",
                options=["auto", "txt", "ocr", ""],
                index=["auto", "txt", "ocr", ""].index(config.mineru_method)
                if config.mineru_method in {"auto", "txt", "ocr", ""}
                else 0,
            )
            config.mineru_export_artifacts = st.checkbox("保存 MinerU Markdown/JSON", value=config.mineru_export_artifacts)

        config.parent_child_enabled = st.checkbox("启用父子文档切片", value=config.parent_child_enabled)
        if config.parent_child_enabled:
            config.parent_chunk_size = st.slider("父文档块大小", 800, 4000, config.parent_chunk_size, 100)
            config.parent_chunk_overlap = st.slider("父文档 overlap", 0, 600, config.parent_chunk_overlap, 50)
            config.child_chunk_size = st.slider("子文档块大小", 120, 1000, config.child_chunk_size, 20)
            config.child_chunk_overlap = st.slider("子文档 overlap", 0, 240, config.child_chunk_overlap, 20)
        else:
            config.chunk_size = st.slider("chunk 大小", 400, 1600, config.chunk_size, 50)
            config.chunk_overlap = st.slider("chunk overlap", 0, 300, config.chunk_overlap, 20)

        config.top_k = st.slider("向量召回数量", 3, 30, config.top_k)
        config.keyword_top_k = st.slider("BM25 召回数量", 3, 30, config.keyword_top_k)
        config.rerank_top_k = st.slider("重排保留数量", 1, 10, config.rerank_top_k)
        config.max_retrieval_rounds = st.slider("最大检索轮次", 1, 3, config.max_retrieval_rounds)
        config.multi_query_count = st.slider("多查询改写数量", 1, 6, config.multi_query_count)
        config.min_keyword_coverage = st.slider("证据关键词覆盖阈值", 0.0, 0.8, config.min_keyword_coverage, 0.02)
        config.min_complex_sources = st.slider("复杂问题最少证据数", 1, 5, config.min_complex_sources)

        config.multimodal_enabled = st.checkbox("启用多模态 PDF 解析", value=config.multimodal_enabled)
        config.extract_tables = st.checkbox("抽取表格", value=config.extract_tables, disabled=not config.multimodal_enabled)
        config.extract_images = st.checkbox("抽取图片", value=config.extract_images, disabled=not config.multimodal_enabled)
        config.render_page_snapshots = st.checkbox("渲染整页截图", value=config.render_page_snapshots, disabled=not config.multimodal_enabled)
        config.max_images_per_pdf = st.slider("每篇 PDF 最大图片数", 0, 80, config.max_images_per_pdf)

        build = st.button("解析并建立索引", type="primary", use_container_width=True)

        st.divider()
        st.subheader("运行状态")
        st.write(f"Embedding: `{config.embedding_backend}`")
        st.write(f"LLM backend: `{config.llm_backend}`")
        st.write(f"LLM router: `{config.llm_router_enabled}`")
        st.write(f"Memory turns: `{config.memory_max_turns if config.memory_enabled else 0}`")
        st.write("Memory layers: `short + summary + long-term`")
        st.write(f"Citation validator: `{config.citation_validation_enabled}`")
        st.write(f"Parser: `{config.parser_backend}`")
        st.write("Retrieval: `hybrid(vector + BM25 + RRF)`")
        st.write(f"Parent-child: `{config.parent_child_enabled}`")
        st.write(f"Multimodal: `{config.multimodal_enabled}`")
        st.write(f"Vector backend: `{config.vector_backend}`")
        st.write(f"Index: `{config.index_dir}`")

    if build:
        if not uploads:
            st.warning("请先上传至少一篇 PDF。")
        else:
            with st.spinner("正在解析 PDF、抽取图表、生成证据并建立索引..."):
                index = rebuild_index(uploads, config)
            modality_counts = pd.Series([chunk.modality for chunk in index.parent_chunks]).value_counts().to_dict()
            st.success(
                f"索引完成：父文档 {len(index.parent_chunks)} 个，子文档 {len(index.search_chunks)} 个，"
                f"模态统计：{modality_counts}"
            )
            st.session_state["index_ready"] = True

    st.title("海洋矿产 Agentic RAG 智能问答")
    st.write("上传论文后构建知识库，系统会通过 LLM Router 判断是否检索外部知识库，并展示来源、引用校验、Trace 和反馈闭环。")

    try:
        index = load_index(str(config.index_dir), config.embedding_backend, config.vector_backend)
    except FileNotFoundError:
        index = None

    if index is None:
        st.info("当前还没有可用索引，请先在左侧上传 PDF 并建立索引。")
        return

    st.caption(f"当前索引：父文档 {len(index.parent_chunks)} 个，检索子文档 {len(index.search_chunks)} 个。")
    question = st.text_area(
        "科研问题",
        value="多金属结核成矿受哪些环境因素控制？",
        height=90,
    )
    ask = st.button("开始问答", type="primary")

    if ask and question.strip():
        agent = MiningRagAgent(index, config, memory_store=memory_store)
        with st.spinner("Agent 正在分析问题、检索证据、重排并生成中文回答..."):
            answer = agent.invoke(question.strip(), session_id=session_id)
        st.session_state["last_question"] = question.strip()
        st.session_state["last_answer"] = answer

    answer = st.session_state.get("last_answer")
    if not answer:
        return

    st.subheader("回答")
    st.write(answer.answer)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Route", answer.route)
    col2.metric("检索轮次", answer.retrieval_rounds)
    col3.metric("证据校验", "通过" if answer.verification_passed else "不足")
    col4.metric("Router 置信度", f"{answer.router_confidence:.2f}")

    st.caption(f"Router reason: {answer.router_reason}")
    if answer.rewritten_query:
        st.caption(f"Rewritten query: {answer.rewritten_query}")

    citation = answer.citation_validation or {}
    if citation:
        if citation.get("passed"):
            st.success(f"引用校验通过：{citation.get('reason', '')}")
        else:
            st.warning(f"引用校验未通过：{citation.get('reason', '')}")
            invalid = citation.get("invalid_citations") or []
            if invalid:
                st.error("\n".join(invalid))

    tab_sources, tab_trace, tab_memory, tab_feedback = st.tabs(["来源证据", "Trace Debug", "记忆", "反馈"])

    with tab_sources:
        render_sources(answer.sources)

    with tab_trace:
        st.write(f"trace_id: `{answer.trace_id}`")
        if answer.trace_events:
            st.dataframe(pd.DataFrame(answer.trace_events), use_container_width=True)
            st.json(answer.trace_events)
        elif answer.trace:
            st.code("\n".join(answer.trace), language="text")
        else:
            st.info("当前回答没有 Trace。")

    with tab_memory:
        st.json(answer.memory or {})
        snapshot = memory_store.snapshot(session_id)
        st.write("短期记忆")
        st.json(snapshot.get("short_term", []))
        st.write("摘要记忆")
        st.write(snapshot.get("summary", ""))
        st.write("长期记忆")
        st.json(snapshot.get("long_term", []))

    with tab_feedback:
        rating_label = st.radio("这次回答质量", options=["有帮助", "有问题"], horizontal=True)
        issue_type = st.selectbox(
            "问题类型",
            options=["", "证据不足", "引用错误", "事实错误", "回答不完整", "其他"],
        )
        correction = st.text_area("期望答案或纠错说明", height=80)
        comment = st.text_area("补充备注", height=80)
        if st.button("提交反馈"):
            try:
                item = feedback_store.submit(
                    trace_id=answer.trace_id,
                    session_id=session_id,
                    rating=1 if rating_label == "有帮助" else -1,
                    question=st.session_state.get("last_question", ""),
                    answer=answer.answer,
                    issue_type=issue_type,
                    correction=correction,
                    comment=comment,
                    citations=answer.citation_validation,
                    evidence=evidence_payload(answer.sources),
                )
                st.success(f"反馈已提交：{item.get('feedback_id', '')}")
            except ValueError as exc:
                st.warning(str(exc))

        pending_feedback = feedback_store.list(status="pending", limit=20)
        with st.expander(f"待审核反馈：{len(pending_feedback)}", expanded=False):
            for item in pending_feedback:
                st.write(f"`{item['feedback_id']}` | rating={item['rating']} | {item['issue_type']}")
                st.caption(item.get("question", ""))
                st.write(item.get("comment", ""))
                if st.button("标记为已解决", key=f"resolve-{item['feedback_id']}"):
                    feedback_store.update_status(item["feedback_id"], "resolved")
                    st.rerun()


if __name__ == "__main__":
    main()
