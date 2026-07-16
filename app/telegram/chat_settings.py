"""Per-chat default options for the Telegram bot, persisted in SQLite.

Stores each chat's preferred preset / model / render factor so ``/settings`` can
change them and photo jobs pick them up (CLAUDE.md §5).
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_settings (
    chat_id       INTEGER PRIMARY KEY,
    preset        TEXT NOT NULL DEFAULT 'full',
    model         TEXT NOT NULL DEFAULT 'artistic',
    render_factor INTEGER
);
"""


@dataclass
class ChatSettings:
    preset: str = "full"
    model: str = "artistic"
    render_factor: int | None = None


class ChatSettingsStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_SCHEMA)

    def get(self, chat_id: int) -> ChatSettings:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT preset, model, render_factor FROM chat_settings WHERE chat_id=?",
                (chat_id,),
            ).fetchone()
        if row is None:
            return ChatSettings()
        return ChatSettings(row["preset"], row["model"], row["render_factor"])

    def set(self, chat_id: int, **fields: object) -> ChatSettings:
        current = self.get(chat_id)
        merged = ChatSettings(
            preset=str(fields.get("preset", current.preset)),
            model=str(fields.get("model", current.model)),
            render_factor=fields.get("render_factor", current.render_factor),  # type: ignore[arg-type]
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO chat_settings (chat_id, preset, model, render_factor)
                   VALUES (?,?,?,?)
                   ON CONFLICT(chat_id) DO UPDATE SET
                       preset=excluded.preset, model=excluded.model,
                       render_factor=excluded.render_factor""",
                (chat_id, merged.preset, merged.model, merged.render_factor),
            )
        return merged
