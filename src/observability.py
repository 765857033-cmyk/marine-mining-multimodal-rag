from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


def start_timer() -> float:
    return perf_counter()


def trace_event(
    events: list[dict[str, Any]],
    node: str,
    started_at: float,
    *,
    status: str = "ok",
    **details: Any,
) -> dict[str, Any]:
    event = {
        "step": len(events) + 1,
        "node": node,
        "status": status,
        "duration_ms": round((perf_counter() - started_at) * 1000, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": _safe_value(details),
    }
    events.append(event)
    return event


def persist_trace(
    trace_dir: Path,
    trace_id: str,
    *,
    question: str,
    route: str,
    events: list[dict],
    metadata: dict,
) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    target = trace_dir / f"{trace_id}.json"
    payload = {
        "trace_id": trace_id,
        "question": question,
        "route": route,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "events": events,
        "metadata": _safe_value(metadata),
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def list_traces(trace_dir: Path, limit: int = 50) -> list[dict]:
    if not trace_dir.exists():
        return []
    records = []
    for path in sorted(trace_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records.append(
            {
                "trace_id": payload.get("trace_id", path.stem),
                "question": payload.get("question", ""),
                "route": payload.get("route", ""),
                "created_at": payload.get("created_at", ""),
                "event_count": len(payload.get("events", [])),
            }
        )
    return records


def load_trace(trace_dir: Path, trace_id: str) -> dict | None:
    if not trace_id or any(char not in "0123456789abcdef-" for char in trace_id.lower()):
        return None
    target = trace_dir / f"{trace_id}.json"
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in value]
    return str(value)
