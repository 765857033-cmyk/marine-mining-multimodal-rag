from __future__ import annotations

import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .models import ConversationTurn


@dataclass(slots=True)
class MemoryContext:
    history: list[ConversationTurn] = field(default_factory=list)
    summary: str = ""
    long_term: list[str] = field(default_factory=list)

    def prompt_text(self) -> str:
        sections = []
        if self.summary:
            sections.append(f"会话摘要记忆：\n{self.summary}")
        if self.long_term:
            sections.append("长期记忆：\n" + "\n".join(f"- {item}" for item in self.long_term))
        return "\n\n".join(sections)


class MemoryStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
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
                CREATE TABLE IF NOT EXISTS memory_sessions (
                    session_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    validated INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_messages_session
                    ON memory_messages(session_id, id);
                CREATE TABLE IF NOT EXISTS long_memories (
                    memory_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(scope, content)
                );
                """
            )

    def get_context(
        self,
        session_id: str,
        query: str,
        max_messages: int,
        long_term_k: int,
        scope: str = "default",
    ) -> MemoryContext:
        self.ensure_session(session_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content FROM memory_messages
                WHERE session_id = ? AND validated = 1
                ORDER BY id DESC LIMIT ?
                """,
                (session_id, max(1, max_messages)),
            ).fetchall()
            summary_row = connection.execute(
                "SELECT summary FROM memory_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            memories = connection.execute(
                """
                SELECT content, importance, updated_at FROM long_memories
                WHERE scope = ?
                ORDER BY importance DESC, updated_at DESC
                """,
                (scope,),
            ).fetchall()

        history = [ConversationTurn(row["role"], row["content"]) for row in reversed(rows)]
        ranked = sorted(
            memories,
            key=lambda row: (
                _lexical_overlap(query, row["content"]),
                float(row["importance"]),
                row["updated_at"],
            ),
            reverse=True,
        )
        selected = [row["content"] for row in ranked[: max(0, long_term_k)]]
        return MemoryContext(
            history=history,
            summary=(summary_row["summary"] if summary_row else ""),
            long_term=selected,
        )

    def append_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        validated: bool,
        max_messages: int,
        max_chars: int,
        summary_max_chars: int,
    ) -> None:
        self.ensure_session(session_id)
        if not validated:
            return
        now = _utc_now()
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO memory_messages(session_id, role, content, validated, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (session_id, "user", question, 1, now),
                    (session_id, "assistant", answer, 1, now),
                ],
            )
            connection.execute(
                "UPDATE memory_sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
        self._compact(session_id, max_messages, max_chars, summary_max_chars)

    def add_long_term(
        self,
        content: str,
        scope: str = "default",
        importance: float = 1.0,
        capacity: int = 64,
    ) -> str:
        normalized = re.sub(r"\s+", " ", content).strip()
        if not normalized:
            raise ValueError("长期记忆内容不能为空")
        now = _utc_now()
        memory_id = uuid.uuid4().hex
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT memory_id FROM long_memories WHERE scope = ? AND content = ?",
                (scope, normalized),
            ).fetchone()
            if existing:
                connection.execute(
                    "UPDATE long_memories SET importance = ?, updated_at = ? WHERE memory_id = ?",
                    (importance, now, existing["memory_id"]),
                )
                return str(existing["memory_id"])
            connection.execute(
                """
                INSERT INTO long_memories(memory_id, scope, content, importance, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (memory_id, scope, normalized, importance, now, now),
            )
            overflow = connection.execute(
                """
                SELECT memory_id FROM long_memories WHERE scope = ?
                ORDER BY importance DESC, updated_at DESC
                LIMIT -1 OFFSET ?
                """,
                (scope, max(1, capacity)),
            ).fetchall()
            if overflow:
                connection.executemany(
                    "DELETE FROM long_memories WHERE memory_id = ?",
                    [(row["memory_id"],) for row in overflow],
                )
        return memory_id

    def list_long_term(self, scope: str = "default", limit: int = 64) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT memory_id, content, importance, created_at, updated_at
                FROM long_memories WHERE scope = ?
                ORDER BY importance DESC, updated_at DESC LIMIT ?
                """,
                (scope, max(1, limit)),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_long_term(self, memory_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM long_memories WHERE memory_id = ?", (memory_id,))
        return cursor.rowcount > 0

    def snapshot(self, session_id: str, scope: str = "default") -> dict:
        self.ensure_session(session_id)
        with self._connect() as connection:
            session = connection.execute(
                "SELECT summary FROM memory_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            rows = connection.execute(
                """
                SELECT role, content, validated, created_at FROM memory_messages
                WHERE session_id = ? ORDER BY id
                """,
                (session_id,),
            ).fetchall()
        return {
            "session_id": session_id,
            "short_term": [dict(row) for row in rows],
            "summary": session["summary"] if session else "",
            "long_term": self.list_long_term(scope),
        }

    def clear_session(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM memory_messages WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM memory_sessions WHERE session_id = ?", (session_id,))

    def ensure_session(self, session_id: str) -> None:
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO memory_sessions(session_id, summary, created_at, updated_at)
                VALUES (?, '', ?, ?)
                """,
                (session_id, now, now),
            )

    def _compact(self, session_id: str, max_messages: int, max_chars: int, summary_max_chars: int) -> None:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, role, content FROM memory_messages
                WHERE session_id = ? AND validated = 1 ORDER BY id
                """,
                (session_id,),
            ).fetchall()
            kept = list(rows)
            evicted = []
            while kept and (
                len(kept) > max(2, max_messages)
                or sum(len(row["content"]) for row in kept) > max(500, max_chars)
            ):
                pair_size = 2 if len(kept) >= 2 else 1
                for _ in range(pair_size):
                    evicted.append(kept.pop(0))
            if not evicted:
                return
            previous = connection.execute(
                "SELECT summary FROM memory_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            summary = _merge_summary(
                previous["summary"] if previous else "",
                evicted,
                max(300, summary_max_chars),
            )
            ids = [row["id"] for row in evicted]
            placeholders = ",".join("?" for _ in ids)
            connection.execute(
                f"DELETE FROM memory_messages WHERE id IN ({placeholders})",
                ids,
            )
            connection.execute(
                "UPDATE memory_sessions SET summary = ?, updated_at = ? WHERE session_id = ?",
                (summary, _utc_now(), session_id),
            )


def _merge_summary(previous: str, rows: list[sqlite3.Row], max_chars: int) -> str:
    additions = []
    for row in rows:
        role = "用户" if row["role"] == "user" else "助手"
        content = re.sub(r"\s+", " ", row["content"]).strip()
        additions.append(f"{role}：{content[:500]}")
    combined = "\n".join(part for part in [previous.strip(), *additions] if part)
    return combined[-max_chars:]


def _lexical_overlap(query: str, text: str) -> float:
    query_terms = _memory_terms(query)
    if not query_terms:
        return 0.0
    return len(query_terms & _memory_terms(text)) / len(query_terms)


def _memory_terms(text: str) -> set[str]:
    terms = set(re.findall(r"[a-z][a-z0-9_-]{1,}", text.lower()))
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        terms.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return terms


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
