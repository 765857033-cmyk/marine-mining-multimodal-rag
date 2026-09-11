from __future__ import annotations

import base64
import os
import re
from hashlib import blake2b
from pathlib import Path
from typing import Any

from .config import AppConfig
from .text_processing import normalize_text


FIGURE_CAPTION_PATTERN = re.compile(
    r"(?im)^\s*((fig\.?|figure|图|图版|plate|table|表)\s*[\dIVXivx\-\.]+[^\n]{0,260})"
)


def summarize_mineru_image(
    asset_path: Path,
    source: str,
    page: int,
    caption: str,
    nearby_text: str,
    config: AppConfig,
) -> str:
    return summarize_image(
        image_path=asset_path,
        source=source,
        page=page,
        modality="image",
        caption=caption,
        page_text=nearby_text,
        config=config,
    )


def summarize_image(
    image_path: Path,
    source: str,
    page: int,
    modality: str,
    caption: str,
    page_text: str,
    config: AppConfig,
) -> str:
    endpoint = resolve_vision_endpoint(config)
    if endpoint is None:
        return fallback_visual_summary(source, page, modality, caption, page_text, image_path)
    try:
        from openai import OpenAI

        client_kwargs: dict[str, Any] = {"api_key": endpoint["api_key"], "timeout": config.llm_timeout}
        if endpoint["base_url"]:
            client_kwargs["base_url"] = endpoint["base_url"]
        client = OpenAI(**client_kwargs)
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        prompt = (
            "你是海洋矿产科研论文的多模态解析助手。请阅读图片或页面截图，"
            "提取对问答检索有价值的信息：图/表类型、坐标轴或图例含义、样品或站位、"
            "元素/矿物/地球化学指标、趋势、结论和任何可引用证据。"
            "如果图片只是装饰或无法判断，请明确说明。"
            f"\nsource={source}, page={page}, modality={modality}"
            f"\n附近 caption：{caption or '无'}"
            f"\n页面文本摘录：{page_text[:1200]}"
        )
        response = client.chat.completions.create(
            model=endpoint["model"],
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ],
                }
            ],
        )
        content = response.choices[0].message.content or ""
        return content[: config.max_image_summary_chars].strip()
    except Exception as exc:
        return fallback_visual_summary(source, page, modality, caption, page_text, image_path, error=str(exc))


def resolve_vision_endpoint(config: AppConfig) -> dict[str, str] | None:
    backend = config.llm_backend.lower()
    if backend == "auto":
        if os.getenv("OPENAI_API_KEY"):
            backend = "openai"
        elif os.getenv("CUSTOM_LLM_API_KEY") and config.custom_llm_base_url:
            backend = "openai-compatible"
        elif os.getenv("OCEANGPT_API_KEY") and config.oceangpt_base_url:
            backend = "ocean-gpt-api"
        else:
            return None

    if backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return None
        return {"api_key": api_key, "base_url": config.openai_base_url, "model": config.vision_model}

    if backend in {"openai-compatible", "custom"}:
        api_key = os.getenv("CUSTOM_LLM_API_KEY", "")
        if not api_key or not config.custom_llm_base_url:
            return None
        return {
            "api_key": api_key,
            "base_url": config.custom_llm_base_url,
            "model": config.vision_model or config.custom_llm_model,
        }

    if backend in {"ocean-gpt", "oceangpt", "ocean-gpt-api"}:
        api_key = os.getenv("OCEANGPT_API_KEY", "")
        if not api_key or not config.oceangpt_base_url:
            return None
        return {
            "api_key": api_key,
            "base_url": config.oceangpt_base_url,
            "model": config.vision_model or config.oceangpt_model,
        }

    return None


def fallback_visual_summary(
    source: str,
    page: int,
    modality: str,
    caption: str,
    page_text: str,
    image_path: Path,
    error: str = "",
) -> str:
    caption_text = caption or "未识别到明确图题或表题"
    surrounding = extract_visual_context(page_text)
    error_note = f"；视觉模型未启用或调用失败：{error}" if error else "；未启用视觉模型"
    return (
        f"多模态视觉证据来自 {source} 第 {page} 页，类型为 {modality}，图片文件为 {image_path.name}"
        f"{error_note}。可检索信息主要来自图题/表题和页面邻近文本：{caption_text}。"
        f"页面相关上下文：{surrounding}"
    )


def build_visual_chunk_text(
    source: str,
    page: int,
    modality: str,
    caption: str,
    summary: str,
    width: int,
    height: int,
    asset_path: Path,
) -> str:
    return (
        f"多模态证据 | modality={modality} | source={source} | page={page}\n"
        f"caption: {caption or '无明确 caption'}\n"
        f"image_size: {width}x{height}\n"
        f"asset_path: {asset_path}\n\n"
        f"视觉/版面摘要：{summary}"
    )


def rows_to_markdown(rows: list[list[Any]]) -> str:
    normalized = [["" if cell is None else str(cell).replace("\n", " ").strip() for cell in row] for row in rows]
    normalized = [row for row in normalized if any(cell for cell in row)]
    if not normalized:
        return ""
    width = max(len(row) for row in normalized)
    normalized = [row + [""] * (width - len(row)) for row in normalized]
    header = normalized[0]
    body = normalized[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def extract_captions(page_text: str) -> list[str]:
    captions = [normalize_text(match.group(1)) for match in FIGURE_CAPTION_PATTERN.finditer(page_text)]
    return [caption for caption in captions if 4 <= len(caption) <= 280]


def nearest_caption(captions: list[str], image_index: int) -> str:
    if not captions:
        return ""
    return captions[min(image_index - 1, len(captions) - 1)]


def extract_visual_context(page_text: str) -> str:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    selected = [
        line
        for line in lines
        if re.search(r"(fig\.?|figure|图|图版|plate|table|表|sample|station|element|oxide|REE|稀土|元素|样品|站位)", line, re.I)
    ]
    if not selected:
        selected = lines[:6]
    return normalize_text(" ".join(selected[:8]))[:900]


def make_multimodal_id(source: str, page: int, modality: str, index: int, text: str) -> str:
    digest = blake2b(f"{source}:{page}:{modality}:{index}:{text[:100]}".encode("utf-8"), digest_size=8).hexdigest()
    return f"{source}:{page}:{modality}:{index}:{digest}"
