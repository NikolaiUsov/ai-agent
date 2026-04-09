from __future__ import annotations
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Tuple


Role = Literal["human", "ai"]


@dataclass(frozen=True)
class HistoryMessage:
    role: Role
    content: str
    created_at: int


_BASE_DIR = Path(__file__).resolve().parents[1]          
_CHAT_HISTORY_DIR = _BASE_DIR / "chat_history"         
_CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)    
_DB_PATH = _CHAT_HISTORY_DIR / "chat_history.sqlite3" 


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT    NOT NULL,
            role       TEXT    NOT NULL CHECK (role IN ('human', 'ai')),
            content    TEXT    NOT NULL,
            created_at INTEGER NOT NULL
        );
        """
    ) 
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_session_created
        ON messages(session_id, created_at, id);
        """
    )


def append_turn(*, session_id: str, user_text: str, ai_text: str, keep_last_pairs: int = 10) -> None:
    """
    Добавляет один "ход" диалога: сообщение пользователя + ответ ассистента.
    Автоматически подрезает историю до последних `keep_last_pairs` пар (2*N сообщений).
    """
    session_id = (session_id or "").strip()
    if not session_id:
        raise ValueError("session_id пустой")
    now = int(time.time())
    keep_last_messages = max(0, int(keep_last_pairs)) * 2
    with _connect() as conn:
        _ensure_schema(conn)
        conn.execute(
            "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, 'human', ?, ?)",
            (session_id, user_text, now),
        )
        conn.execute(
            "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, 'ai', ?, ?)",
            (session_id, ai_text, now),
        )
        if keep_last_messages > 0:
            # Оставляем только последние 2*N сообщений (по времени/ид).
            conn.execute(
                """
                DELETE FROM messages
                WHERE session_id = ?
                  AND id NOT IN (
                    SELECT id FROM messages
                    WHERE session_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                  );
                """,
                (session_id, session_id, keep_last_messages),
            )


def load_messages(*, session_id: str, limit_pairs: int = 10) -> List[HistoryMessage]:
    """
    Возвращает историю для `session_id` в хронологическом порядке.
    По умолчанию — последние 10 пар (20 сообщений).
    """
    session_id = (session_id or "").strip()
    if not session_id:
        return []
    limit_messages = max(0, int(limit_pairs)) * 2
    if limit_messages == 0:
        return []
    with _connect() as conn:
        _ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?;
            """,
            (session_id, limit_messages),
        ).fetchall()
    # На чтении берем "с хвоста" (DESC) и разворачиваем обратно.
    rows = list(reversed(rows))
    return [
        HistoryMessage(role=row["role"], content=row["content"], created_at=row["created_at"])
        for row in rows
    ]


def clear_history(*, session_id: str) -> None:
    session_id = (session_id or "").strip()
    if not session_id:
        return
    with _connect() as conn:
        _ensure_schema(conn)
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
