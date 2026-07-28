from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FeedbackStore:
    def __init__(self, db_path: Path | str, bad_cases_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.bad_cases_path = Path(bad_cases_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.bad_cases_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    session_id TEXT NOT NULL DEFAULT '',
                    rating INTEGER NOT NULL,
                    issue_type TEXT NOT NULL DEFAULT '',
                    correction TEXT NOT NULL DEFAULT '',
                    comment TEXT NOT NULL DEFAULT '',
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    citations_json TEXT NOT NULL DEFAULT '{}',
                    evidence_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_feedback_trace ON feedback(trace_id);
                """
            )

    def submit(
        self,
        *,
        trace_id: str,
        session_id: str,
        rating: int,
        question: str,
        answer: str,
        issue_type: str = "",
        correction: str = "",
        comment: str = "",
        citations: dict | None = None,
        evidence: list[dict] | None = None,
    ) -> dict:
        if rating not in {-1, 1}:
            raise ValueError("rating 只能是 -1 或 1")
        feedback_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        evidence = evidence or []
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback(
                    feedback_id, trace_id, session_id, rating, issue_type, correction,
                    comment, question, answer, citations_json, evidence_json,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    feedback_id,
                    trace_id,
                    session_id,
                    rating,
                    issue_type.strip(),
                    correction.strip(),
                    comment.strip(),
                    question.strip(),
                    answer.strip(),
                    json.dumps(citations or {}, ensure_ascii=False),
                    json.dumps(evidence, ensure_ascii=False),
                    now,
                    now,
                ),
            )
        if rating < 0 or correction.strip():
            self._append_bad_case(
                {
                    "feedback_id": feedback_id,
                    "trace_id": trace_id,
                    "question": question.strip(),
                    "actual_answer": answer.strip(),
                    "expected_answer": correction.strip(),
                    "expected_terms": [],
                    "expected_sources": [
                        {"source": item.get("source", ""), "page": item.get("page", 0)}
                        for item in evidence
                        if item.get("source")
                    ],
                    "should_refuse": issue_type == "证据不足却作答",
                    "issue_type": issue_type.strip(),
                    "review_status": "pending",
                }
            )
        return self.get(feedback_id) or {}

    def list(self, status: str | None = None, limit: int = 100) -> list[dict]:
        sql = "SELECT * FROM feedback"
        params: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(limit, 500)))
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._decode(row) for row in rows]

    def get(self, feedback_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM feedback WHERE feedback_id = ?",
                (feedback_id,),
            ).fetchone()
        return self._decode(row) if row else None

    def update_status(self, feedback_id: str, status: str) -> dict:
        if status not in {"pending", "approved", "rejected", "resolved"}:
            raise ValueError("不支持的审核状态")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE feedback SET status = ?, updated_at = ? WHERE feedback_id = ?",
                (status, datetime.now(timezone.utc).isoformat(), feedback_id),
            )
        if cursor.rowcount == 0:
            raise KeyError(feedback_id)
        return self.get(feedback_id) or {}

    def _decode(self, row: sqlite3.Row) -> dict:
        item = dict(row)
        item["citations"] = json.loads(item.pop("citations_json"))
        item["evidence"] = json.loads(item.pop("evidence_json"))
        return item

    def _append_bad_case(self, record: dict) -> None:
        existing_ids = set()
        if self.bad_cases_path.exists():
            for line in self.bad_cases_path.read_text(encoding="utf-8").splitlines():
                try:
                    existing_ids.add(json.loads(line).get("feedback_id"))
                except json.JSONDecodeError:
                    continue
        if record["feedback_id"] in existing_ids:
            return
        with self.bad_cases_path.open("a", encoding="utf-8") as writer:
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
