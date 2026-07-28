from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .models import RetrievalResult


@dataclass(slots=True)
class CitationValidation:
    passed: bool
    citation_count: int = 0
    valid_citations: list[int] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    uncited_claims: list[str] = field(default_factory=list)
    claim_count: int = 0
    cited_claim_count: int = 0
    coverage: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


_CITATION_RE = re.compile(r"\[(\d+)\]")
_SOURCE_LINE_RE = re.compile(
    r"^\s*\[(\d+)\]\s+(.+?)\s+(?:第\s*)?(\d+)\s*页",
    re.IGNORECASE,
)
_SENTENCE_RE = re.compile(r".+?[。！？!?；;](?:\s*\[\d+\])*|.+$")


def validate_citations(
    answer: str,
    results: list[RetrievalResult],
    min_coverage: float = 0.6,
) -> CitationValidation:
    if not results:
        return CitationValidation(
            passed=False,
            reason="没有检索证据，无法校验引用",
        )

    citation_indexes = [int(value) for value in _CITATION_RE.findall(answer)]
    valid_indexes = sorted({index for index in citation_indexes if 1 <= index <= len(results)})
    invalid = [f"[{index}] 超出证据范围 1-{len(results)}" for index in citation_indexes if index not in valid_indexes]

    for line in answer.splitlines():
        match = _SOURCE_LINE_RE.match(line)
        if not match:
            continue
        index = int(match.group(1))
        if not 1 <= index <= len(results):
            continue
        expected = results[index - 1].chunk
        mentioned_source = _normalize_source(match.group(2))
        mentioned_page = int(match.group(3))
        if _normalize_source(expected.source) not in mentioned_source and mentioned_source not in _normalize_source(expected.source):
            invalid.append(f"[{index}] 文档名与检索证据不一致")
        if mentioned_page != expected.page:
            invalid.append(f"[{index}] 页码 {mentioned_page} 与检索页码 {expected.page} 不一致")

    claims = _extract_claims(answer)
    cited_claims = [claim for claim in claims if _CITATION_RE.search(claim)]
    uncited_claims = [claim for claim in claims if not _CITATION_RE.search(claim)]
    coverage = len(cited_claims) / max(1, len(claims))

    if not citation_indexes:
        reason = "回答包含专业结论，但没有任何引用标记"
        passed = False
    elif invalid:
        reason = "存在无法映射到检索证据的文档、页码或引用编号"
        passed = False
    elif claims and coverage < min_coverage:
        reason = f"事实性结论引用覆盖率 {coverage:.0%} 低于阈值 {min_coverage:.0%}"
        passed = False
    else:
        reason = f"引用均可映射到检索证据，事实性结论覆盖率 {coverage:.0%}"
        passed = True

    return CitationValidation(
        passed=passed,
        citation_count=len(citation_indexes),
        valid_citations=valid_indexes,
        invalid_citations=list(dict.fromkeys(invalid)),
        uncited_claims=uncited_claims[:8],
        claim_count=len(claims),
        cited_claim_count=len(cited_claims),
        coverage=coverage,
        reason=reason,
    )


def _extract_claims(answer: str) -> list[str]:
    main_answer = re.split(r"\n\s*(?:来源|参考来源|证据来源)\s*[：:]", answer, maxsplit=1)[0]
    claims: list[str] = []
    for line in main_answer.splitlines():
        for part in _SENTENCE_RE.findall(line):
            claim = re.sub(r"^[\s#>*\-•\d.、]+", "", part).strip()
            plain = _CITATION_RE.sub("", claim).strip()
            if len(plain) < 12:
                continue
            if any(marker in plain for marker in ["证据不足", "无法确定", "建议上传", "当前知识库中没有"]):
                continue
            claims.append(claim)
    return claims


def _normalize_source(value: str) -> str:
    return re.sub(r"[\s`*_]+", "", value).lower()
