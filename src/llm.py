from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass

from .config import AppConfig
from .models import ConversationTurn, RetrievalResult


@dataclass(slots=True)
class LlmEndpoint:
    backend: str
    model: str
    api_key: str
    base_url: str = ""


class AnswerGenerator:
    def __init__(self, config: AppConfig | str = "gpt-4o-mini") -> None:
        if isinstance(config, AppConfig):
            self.config = config
        else:
            self.config = AppConfig(openai_model=config)
        self.endpoint = resolve_llm_endpoint(self.config)
        self._client = None
        if self.endpoint is not None:
            try:
                from openai import OpenAI

                kwargs = {"api_key": self.endpoint.api_key, "timeout": self.config.llm_timeout}
                if self.endpoint.base_url:
                    kwargs["base_url"] = self.endpoint.base_url
                self._client = OpenAI(**kwargs)
            except Exception:
                self._client = None

    def direct_answer(self, question: str) -> str:
        if self._client is not None and self.endpoint is not None:
            prompt = (
                "你是海洋矿产科研文献问答助手。用户问题无需检索即可回答时，"
                "请用中文简洁回答，并说明若涉及论文证据需要上传文献。\n\n"
                f"问题：{question}"
            )
            return self._chat(prompt)
        return "我可以帮助解析海洋矿产科研 PDF、检索相关片段、重排证据并生成带来源的回答。请先上传论文 PDF 并建立索引。"

    def route_question(
        self,
        question: str,
        history: list[ConversationTurn] | None = None,
        memory_context: str = "",
    ) -> dict | None:
        if not self.config.llm_router_enabled or self._client is None or self.endpoint is None:
            return None
        history_text = format_history(history or [], max_turns=self.config.memory_max_turns)
        prompt = f"""你是海洋矿产科研文献问答系统的 Router。
请判断用户问题是否必须使用外部论文知识库检索后才能可靠回答。

必须返回严格 JSON，不要输出 Markdown，不要解释 JSON 之外的内容。
JSON 字段：
- route: "rag" 或 "direct"
- need_retrieval: true 或 false
- rewritten_query: 如果需要检索，给出适合中英文论文检索的查询；否则给出原问题
- reason: 中文说明为什么这样路由
- confidence: 0 到 1 的数字

判定规则：
1. 涉及某篇/上传论文、文献依据、source、页码、表格、图片、图件、数据、具体数值、专业结论来源时，必须 route="rag"。
2. 海洋矿产专业问题，如多金属结核、富钴结壳、热液硫化物、成矿机制、控制因素、元素含量、稀土、Ce异常、地球化学指标，通常 route="rag"。
3. 闲聊、系统能力介绍、无需文献支撑的一般说明，可以 route="direct"。
4. 如果用户说“继续刚才”“上一个问题”“这些来源”，要结合对话历史判断，通常需要检索或沿用文献上下文。

最近对话：
{history_text or "无"}

摘要与长期记忆：
{memory_context or "无"}

用户问题：
{question}
"""
        try:
            raw = self._chat(prompt)
            return parse_router_json(raw)
        except Exception:
            return None

    def generate(
        self,
        question: str,
        results: list[RetrievalResult],
        history: list[ConversationTurn] | None = None,
        memory_context: str = "",
    ) -> str:
        if not results:
            return "当前知识库中没有找到足够相关的论文片段。可以尝试上传更多 PDF，或换一种更具体的问法。"

        context = "\n\n".join(
            f"[{i}] 来源：{r.chunk.source}，页码：{r.chunk.page}，模态：{r.chunk.modality}\n{r.chunk.text[:1400]}"
            for i, r in enumerate(results, start=1)
        )
        if self._client is not None and self.endpoint is not None:
            history_text = format_history(history or [], max_turns=self.config.memory_max_turns)
            prompt = f"""你是面向海洋矿产科研论文的 Agentic RAG 问答系统。
请只基于给定文献片段回答。无论文献片段是中文还是英文，最终回答必须全部使用中文表达。
要求：
1. 先给出直接答案；
2. 每个事实性或专业结论句末都必须标注来源，例如 [1]；只能使用下方已提供的引用编号；
3. 最后列出 source 文档名、页码和相关片段摘要，文档名和页码必须与片段元数据完全一致；
4. 如果证据来自 table/image/page_snapshot，要说明它是表格证据还是视觉证据；
5. 如果证据不足，明确说明不足。
6. 不要直接输出英文原句；必要的专业术语可以保留英文缩写，并在中文后用括号说明，例如 稀土元素（REE）。
7. 可以参考最近对话理解用户追问，但结论必须由当前给定文献片段支撑。

最近对话：
{history_text or "无"}

摘要与长期记忆（只用于理解追问，不可作为论文证据）：
{memory_context or "无"}

问题：
{question}

文献片段：
{context}
"""
            return self._chat(prompt)

        bullets = []
        for i, result in enumerate(results[:4], start=1):
            summary = _summarize_sentence(result.chunk.text, question)
            bullets.append(f"- {summary} [{i}]")
        source_lines = [
            f"[{i}] {result.chunk.source} 第 {result.chunk.page} 页，模态={_modality_zh(result.chunk.modality)}，相关性={result.final_score:.3f}"
            for i, result in enumerate(results[:4], start=1)
        ]
        return "基于当前检索到的文献片段，可得到以下要点：\n" + "\n".join(bullets) + "\n\n来源：\n" + "\n".join(source_lines)

    def repair_citations(
        self,
        question: str,
        answer: str,
        results: list[RetrievalResult],
        validation_reason: str,
    ) -> str:
        if self._client is None or self.endpoint is None:
            return answer
        evidence_map = "\n".join(
            f"[{index}] {item.chunk.source} 第 {item.chunk.page} 页，模态={item.chunk.modality}"
            for index, item in enumerate(results, start=1)
        )
        prompt = f"""请修复下面回答中的引用问题，不能增加任何新事实。
每个事实性结论必须在句末使用 [数字] 引用；引用编号、文档名和页码只能来自证据映射。
删除无法由证据支持的结论。最终输出必须全部为中文，不要解释修复过程。

问题：{question}
校验失败原因：{validation_reason}

证据映射：
{evidence_map}

待修复回答：
{answer}
"""
        return self._chat(prompt)

    def _chat(self, prompt: str) -> str:
        assert self._client is not None
        assert self.endpoint is not None
        response = self._client.chat.completions.create(
            model=self.endpoint.model,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是严谨的海洋矿产科研问答助手。所有最终回答必须使用中文，"
                        "不得编造来源，不得把英文文献原句直接当作中文答案。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    @property
    def backend_name(self) -> str:
        return self.endpoint.backend if self.endpoint is not None and self._client is not None else "fallback"


def resolve_llm_endpoint(config: AppConfig) -> LlmEndpoint | None:
    backend = config.llm_backend.lower()
    if backend == "auto":
        if os.getenv("OCEANGPT_API_KEY") and config.oceangpt_base_url:
            backend = "ocean-gpt-api"
        elif os.getenv("OPENAI_API_KEY"):
            backend = "openai"
        elif os.getenv("CUSTOM_LLM_API_KEY") and config.custom_llm_base_url:
            backend = "openai-compatible"
        else:
            return None

    if backend in {"none", "fallback", "off"}:
        return None

    if backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return None
        return LlmEndpoint(
            backend="openai",
            model=config.openai_model,
            api_key=api_key,
            base_url=config.openai_base_url,
        )

    if backend in {"ocean-gpt", "oceangpt", "ocean-gpt-api"}:
        api_key = os.getenv("OCEANGPT_API_KEY", "")
        if not api_key or not config.oceangpt_base_url:
            return None
        return LlmEndpoint(
            backend="ocean-gpt-api",
            model=config.oceangpt_model,
            api_key=api_key,
            base_url=config.oceangpt_base_url,
        )

    if backend in {"openai-compatible", "custom"}:
        api_key = os.getenv("CUSTOM_LLM_API_KEY", "")
        if not api_key or not config.custom_llm_base_url or not config.custom_llm_model:
            return None
        return LlmEndpoint(
            backend="openai-compatible",
            model=config.custom_llm_model,
            api_key=api_key,
            base_url=config.custom_llm_base_url,
        )

    return None


def parse_router_json(raw: str) -> dict | None:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            text = match.group(0)
    data = json.loads(text)
    route = str(data.get("route", "")).lower()
    need_retrieval = bool(data.get("need_retrieval", route == "rag"))
    if route not in {"rag", "direct"}:
        route = "rag" if need_retrieval else "direct"
    confidence = data.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0
    return {
        "route": route,
        "need_retrieval": route == "rag" or need_retrieval,
        "rewritten_query": str(data.get("rewritten_query", "")).strip(),
        "reason": str(data.get("reason", "")).strip(),
        "confidence": max(0.0, min(1.0, confidence)),
    }


def format_history(history: list[ConversationTurn], max_turns: int = 6) -> str:
    if not history:
        return ""
    selected = history[-max(1, max_turns * 2) :]
    lines = []
    for turn in selected:
        role = "用户" if turn.role == "user" else "助手"
        content = re.sub(r"\s+", " ", turn.content).strip()
        lines.append(f"{role}: {content[:500]}")
    return "\n".join(lines)


def _summarize_sentence(text: str, question: str) -> str:
    sentences = re.split(r"(?<=[。！？.!?])\s+|\n+", text)
    q_terms = set(re.findall(r"[A-Za-z][A-Za-z0-9_\-]+|[\u4e00-\u9fff]{2,}", question.lower()))
    best = ""
    best_score = -1
    for sentence in sentences:
        terms = set(re.findall(r"[A-Za-z][A-Za-z0-9_\-]+|[\u4e00-\u9fff]{2,}", sentence.lower()))
        score = len(q_terms & terms)
        if score > best_score and 30 <= len(sentence) <= 320:
            best = sentence.strip()
            best_score = score
    return _to_chinese_evidence_summary(best or text[:260].replace("\n", " "))


def _to_chinese_evidence_summary(sentence: str) -> str:
    sentence = sentence.strip()
    if not sentence:
        return "该证据片段与问题相关，但内容较短，需要结合更多来源判断。"
    if _looks_chinese(sentence):
        return sentence

    lowered = sentence.lower()
    glossary = {
        "polymetallic nodules": "多金属结核",
        "manganese nodules": "锰结核",
        "cobalt-rich crust": "富钴结壳",
        "ferromanganese crust": "铁锰结壳",
        "hydrothermal sulfide": "热液硫化物",
        "seafloor massive sulfide": "海底块状硫化物",
        "redox": "氧化还原条件",
        "sedimentation rate": "沉积速率",
        "metal source": "金属来源",
        "geochemical": "地球化学",
        "mineralogy": "矿物学",
        "rare earth elements": "稀土元素",
        "ree": "稀土元素（REE）",
        "ce anomaly": "铈异常",
        "controlling factors": "控制因素",
        "formation mechanism": "形成机制",
        "geological setting": "地质背景",
        "sample": "样品",
        "station": "站位",
        "element concentration": "元素含量",
    }
    hits = []
    for key, value in glossary.items():
        if key in lowered and value not in hits:
            hits.append(value)

    if hits:
        return "英文文献证据表明，该片段涉及" + "、".join(hits[:8]) + "等信息，可作为回答该问题的依据。"

    topic_words = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{3,}", sentence)
    topic_words = [word for word in topic_words if word.lower() not in {"this", "that", "with", "from", "were", "have", "been", "also"}]
    if topic_words:
        return "英文文献证据显示，该片段围绕" + "、".join(topic_words[:6]) + "等关键词展开，需要结合来源上下文解释。"
    return "英文文献证据与问题相关，但未识别出足够明确的专业关键词。"


def _looks_chinese(text: str) -> bool:
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    return chinese_chars >= max(4, latin_chars // 3)


def _modality_zh(modality: str) -> str:
    return {
        "text": "文本",
        "table": "表格",
        "image": "图片",
        "page_snapshot": "页面截图",
    }.get(modality, modality)
