from __future__ import annotations

import tempfile
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

st.set_page_config(page_title="海洋矿产 Agentic RAG", page_icon="search", layout="wide")


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
                "parent_id": result.chunk.metadata.get("retrieved_parent_id", result.chunk.parent_id),
                "matched_child": result.chunk.metadata.get("matched_child_index", ""),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    for i, result in enumerate(results, start=1):
        with st.expander(f"[{i}] {result.chunk.source} p.{result.chunk.page} · {result.chunk.modality}"):
            asset_path = result.chunk.metadata.get("asset_path", "")
            if asset_path and Path(asset_path).exists():
                st.image(asset_path, caption=result.chunk.metadata.get("caption", ""))
            matched_child = result.chunk.metadata.get("matched_child_text", "")
            if matched_child:
                st.caption("命中的子片段")
                st.code(matched_child, language="text")
                st.caption("回召的父文档上下文")
            st.write(result.chunk.text)


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
        config.memory_max_turns = st.slider("记忆轮数", 1, 12, config.memory_max_turns)
        if st.button("清空上下文记忆", use_container_width=True):
            memory_store.clear_session(session_id)
            st.session_state.pop("last_answer", None)
            st.success("当前会话的短期记忆和摘要记忆已清空。")
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
        config.parent_child_enabled = st.checkbox("启用父子文档切片", value=config.parent_child_enabled)
        if config.parent_child_enabled:
            config.parent_chunk_size = st.slider("父文档块大小", 800, 4000, config.parent_chunk_size, 100)
            config.parent_chunk_overlap = st.slider("父文档 overlap", 0, 600, config.parent_chunk_overlap, 50)
            config.child_chunk_size = st.slider("子文档块大小", 120, 1000, config.child_chunk_size, 20)
            config.child_chunk_overlap = st.slider("子文档 overlap", 0, 240, config.child_chunk_overlap, 20)
        config.parser_backend = st.selectbox(
            "PDF 解析后端",
            options=["mineru", "pymupdf", "docling", "auto"],
            index=["mineru", "pymupdf", "docling", "auto"].index(config.parser_backend)
            if config.parser_backend in {"mineru", "pymupdf", "docling", "auto"}
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
            config.mineru_export_artifacts = st.checkbox(
                "保存 MinerU Markdown/JSON", value=config.mineru_export_artifacts
            )
        if config.parser_backend in {"docling", "auto"}:
            config.docling_use_hybrid_chunker = st.checkbox(
                "Docling HybridChunker", value=config.docling_use_hybrid_chunker
            )
            config.docling_do_table_structure = st.checkbox(
                "Docling 表格结构识别", value=config.docling_do_table_structure
            )
            config.docling_do_ocr = st.checkbox("Docling OCR", value=config.docling_do_ocr)
            config.docling_export_artifacts = st.checkbox(
                "保存 Docling Markdown/JSON", value=config.docling_export_artifacts
            )
        config.chunk_size = config.parent_chunk_size if config.parent_child_enabled else st.slider("chunk 大小", 400, 1600, config.chunk_size, 50)
        config.chunk_overlap = config.parent_chunk_overlap if config.parent_child_enabled else st.slider("chunk overlap", 0, 300, config.chunk_overlap, 20)
        config.top_k = st.slider("向量召回数量", 3, 30, config.top_k)
        config.keyword_top_k = st.slider("BM25 召回数量", 3, 30, config.keyword_top_k)
        config.rerank_top_k = st.slider("重排保留", 1, 10, config.rerank_top_k)
        config.max_retrieval_rounds = st.slider("最大检索轮次", 1, 3, config.max_retrieval_rounds)
        config.multi_query_count = st.slider("多查询改写数量", 1, 6, config.multi_query_count)
        config.min_keyword_coverage = st.slider("证据关键词覆盖阈值", 0.0, 0.8, config.min_keyword_coverage, 0.02)
        config.min_complex_sources = st.slider("复杂问题最少证据数", 1, 5, config.min_complex_sources)
        config.multimodal_enabled = st.checkbox("启用多模态 PDF 解析", value=config.multimodal_enabled)
        config.extract_tables = st.checkbox("抽取表格", value=config.extract_tables, disabled=not config.multimodal_enabled)
        config.extract_images = st.checkbox("抽取图片", value=config.extract_images, disabled=not config.multimodal_enabled)
        config.render_page_snapshots = st.checkbox("渲染整页截图", value=config.render_page_snapshots, disabled=not config.multimodal_enabled)
        config.max_images_per_pdf = st.slider("每篇 PDF 最多图片数", 0, 80, config.max_images_per_pdf)
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
        st.write(f"Retrieval: `hybrid(vector + BM25 + RRF)`")
        st.write(f"Parent-child: `{config.parent_child_enabled}`")
        st.write(f"Multimodal: `{config.multimodal_enabled}`")
        st.write(f"Vector backend: `{config.vector_backend}`")
        st.write(f"Index: `{config.index_dir}`")

    if build:
        if not uploads:
            st.warning("请先上传至少一篇 PDF。")
        else:
            with st.spinner("正在解析 PDF、清洗文本、抽取图表、生成多模态证据、向量化并建立索引..."):
                index = rebuild_index(uploads, config)
            modality_counts = pd.Series([chunk.modality for chunk in index.parent_chunks]).value_counts().to_dict()
            st.success(
                f"索引完成：父文档 {len(index.parent_chunks)} 个，子文档 {len(index.search_chunks)} 个，"
                f"模态统计：{modality_counts}"
            )

    index = load_index(str(config.index_dir), config.embedding_backend, config.vector_backend)
    agent = MiningRagAgent(index, config, memory_store=memory_store)

    st.title("基于 Agentic RAG 的海洋矿产科研文献智能问答系统")
    st.write("上传海洋矿产相关论文后，可围绕多金属结核、富钴结壳、热液硫化物、赋存环境、控制因素、识别指标、图件和表格等问题进行带来源问答。")

    left, right = st.columns([0.58, 0.42], gap="large")
    with left:
        memory_snapshot = memory_store.snapshot(session_id)
        if config.memory_enabled and (
            memory_snapshot["short_term"] or memory_snapshot["summary"] or memory_snapshot["long_term"]
        ):
            with st.expander("上下文记忆", expanded=False):
                st.caption(
                    f"短期消息 {len(memory_snapshot['short_term'])} 条 · "
                    f"摘要 {len(memory_snapshot['summary'])} 字 · "
                    f"长期记忆 {len(memory_snapshot['long_term'])} 条"
                )
                if memory_snapshot["summary"]:
                    st.markdown("**摘要记忆**")
                    st.write(memory_snapshot["summary"])
                for turn in memory_snapshot["short_term"]:
                    speaker = "用户" if turn["role"] == "user" else "助手"
                    validated = "已校验" if turn["validated"] else "未写入 Agent 记忆"
                    st.markdown(f"**{speaker}（{validated}）：** {turn['content']}")
                if memory_snapshot["long_term"]:
                    st.markdown("**长期记忆**")
                    for item in memory_snapshot["long_term"]:
                        st.write(f"- {item['content']}")
        question = st.text_area(
            "科研问题",
            value="多金属结核成矿受哪些环境因素控制？请给出文献依据。",
            height=120,
        )
        ask = st.button("开始问答", type="primary")

        if ask:
            with st.spinner("Agent 正在分析问题、路由、检索、rerank 并生成答案..."):
                answer = agent.invoke(
                    question,
                    session_id=session_id if config.memory_enabled else "",
                )
            st.session_state["last_answer"] = answer
            st.session_state["last_question"] = question

        answer = st.session_state.get("last_answer")
        if answer:
            st.subheader("回答")
            st.markdown(answer.answer)
            with st.form(f"feedback_form_{answer.trace_id}"):
                st.caption("你的反馈会绑定本次 Trace；负反馈和纠错会自动进入离线评测坏例集。")
                rating_label = st.radio(
                    "回答是否有帮助",
                    options=["有帮助", "需要改进"],
                    horizontal=True,
                )
                issue_type = st.selectbox(
                    "问题类型",
                    options=["", "检索不相关", "引用错误", "事实错误", "回答不完整", "证据不足却作答", "其他"],
                )
                correction = st.text_area("正确答案或纠正内容", height=80)
                comment = st.text_input("补充说明")
                submit_feedback = st.form_submit_button("提交反馈")
                if submit_feedback:
                    evidence = [
                        {
                            "source": item.chunk.source,
                            "page": item.chunk.page,
                            "modality": item.chunk.modality,
                            "chunk_id": item.chunk.chunk_id,
                        }
                        for item in answer.sources
                    ]
                    feedback_store.submit(
                        trace_id=answer.trace_id,
                        session_id=session_id,
                        rating=1 if rating_label == "有帮助" else -1,
                        question=st.session_state.get("last_question", question),
                        answer=answer.answer,
                        issue_type=issue_type,
                        correction=correction,
                        comment=comment,
                        citations=answer.citation_validation,
                        evidence=evidence,
                    )
                    st.success("反馈已记录，并已绑定本次 Agent Trace。")

    with right:
        answer = st.session_state.get("last_answer")
        st.subheader("Agent 过程")
        if answer:
            st.write(f"Trace ID: `{answer.trace_id}`")
            st.write(f"Route: `{answer.route}`")
            st.write(f"Router confidence: `{answer.router_confidence:.2f}`")
            st.caption(answer.router_reason)
            st.write(f"Retrieval rounds: `{answer.retrieval_rounds}`")
            st.write(f"Source verification: `{answer.verification_passed}`")
            st.caption(answer.verification_reason)
            citation = answer.citation_validation
            if citation:
                st.write(f"Citation validation: `{citation.get('passed', False)}`")
                st.caption(citation.get("reason", ""))
                st.progress(float(citation.get("coverage", 0.0)), text="事实性结论引用覆盖率")
                if citation.get("invalid_citations"):
                    st.error("；".join(citation["invalid_citations"]))
            st.write(f"查询：`{answer.rewritten_query}`")
            if answer.query_variants:
                with st.expander("查询改写"):
                    for query in answer.query_variants:
                        st.code(query, language="text")
            if answer.trace_events:
                trace_rows = [
                    {
                        "step": event["step"],
                        "node": event["node"],
                        "status": event["status"],
                        "duration_ms": event["duration_ms"],
                        "summary": answer.trace[index],
                    }
                    for index, event in enumerate(answer.trace_events)
                ]
                st.dataframe(pd.DataFrame(trace_rows), use_container_width=True, hide_index=True)
                with st.expander("Trace 详细数据"):
                    for event in answer.trace_events:
                        st.markdown(f"**{event['step']}. {event['node']}**")
                        st.json(event["details"])
            if answer.memory:
                st.caption(
                    f"记忆写入={answer.memory.get('turn_saved')} · "
                    f"短期消息={answer.memory.get('short_term_messages', 0)} · "
                    f"摘要={answer.memory.get('summary_chars', 0)} 字 · "
                    f"长期={answer.memory.get('long_term_count', 0)} 条"
                )
            st.subheader("来源与 Rerank")
            render_sources(answer.sources)
            pending_feedback = feedback_store.list(status="pending", limit=20)
            with st.expander(f"反馈审核队列（{len(pending_feedback)}）", expanded=False):
                if pending_feedback:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "id": item["feedback_id"][:8],
                                    "rating": item["rating"],
                                    "issue": item["issue_type"],
                                    "question": item["question"][:50],
                                    "created_at": item["created_at"],
                                }
                                for item in pending_feedback
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                    selected_feedback = st.selectbox(
                        "选择反馈",
                        pending_feedback,
                        format_func=lambda item: f"{item['feedback_id'][:8]} · {item['question'][:35]}",
                    )
                    review_status = st.selectbox("审核结果", ["resolved", "approved", "rejected"])
                    if st.button("更新审核状态"):
                        feedback_store.update_status(selected_feedback["feedback_id"], review_status)
                        st.rerun()
                else:
                    st.caption("当前没有待审核反馈。")
        else:
            st.info("完成一次问答后，这里会展示路由、检索片段和 rerank 排序。")


if __name__ == "__main__":
    main()
