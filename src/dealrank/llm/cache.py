"""A small SQLite store of LLM responses, keyed by the exact request."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dealrank.llm.base import Completion


class ResponseCache:
    """Reuse responses to identical requests, and keep every fresh one.

    The ``responses`` table keeps the first response seen for each request
    key and serves cache hits. The ``events`` table appends every fresh
    response, so repeated timing runs never overwrite earlier ones.
    """

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self._local = threading.local()
        with self._conn() as conn:
            for table in ("responses", "events"):
                key = "key TEXT PRIMARY KEY" if table == "responses" else "id INTEGER PRIMARY KEY, key TEXT"
                conn.execute(
                    f"CREATE TABLE IF NOT EXISTS {table} ({key}, model TEXT, text TEXT,"
                    " prompt_tokens INT, completion_tokens INT, latency_s REAL,"
                    " created_at REAL, meta TEXT)"
                )
            conn.execute("PRAGMA journal_mode=WAL")

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=60)
            self._local.conn = conn
        return conn

    def get(self, key: str) -> Completion | None:
        from dealrank.llm.base import Completion

        row = (
            self._conn()
            .execute(
                "SELECT text, prompt_tokens, completion_tokens, latency_s, meta FROM responses WHERE key=?",
                (key,),
            )
            .fetchone()
        )
        if row is None:
            return None
        return Completion(
            text=row[0],
            prompt_tokens=row[1],
            completion_tokens=row[2],
            latency_s=row[3],
            meta=json.loads(row[4] or "{}"),
        )

    def put(self, key: str, model: str, completion: Completion) -> None:
        values = (
            key,
            model,
            completion.text,
            completion.prompt_tokens,
            completion.completion_tokens,
            completion.latency_s,
            time.time(),
            json.dumps(completion.meta, default=str),
        )
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO events (key, model, text, prompt_tokens, completion_tokens,"
                " latency_s, created_at, meta) VALUES (?,?,?,?,?,?,?,?)",
                values,
            )
            conn.execute("INSERT OR IGNORE INTO responses VALUES (?,?,?,?,?,?,?,?)", values)
