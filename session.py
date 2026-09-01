"""Local, persistent conversation sessions for the coding agent."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent import AgentError


SESSION_VERSION = 1
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MESSAGE_ROLES = {"system", "user", "assistant", "tool"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _title_from_messages(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") != "user" or not isinstance(message.get("content"), str):
            continue
        title = " ".join(message["content"].split())
        if title:
            return title[:60]
    return "New session"


@dataclass
class AgentSession:
    id: str
    workspace: str
    created_at: str
    updated_at: str
    title: str = "New session"
    messages: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentSession":
        if payload.get("version") != SESSION_VERSION:
            raise AgentError(f"unsupported session version: {payload.get('version')!r}")
        session_id = payload.get("id")
        workspace = payload.get("workspace")
        created_at = payload.get("created_at")
        updated_at = payload.get("updated_at")
        title = payload.get("title")
        messages = payload.get("messages")
        if not isinstance(session_id, str) or not SESSION_ID_PATTERN.fullmatch(session_id):
            raise AgentError("invalid session id")
        if not isinstance(workspace, str) or not workspace:
            raise AgentError("invalid session workspace")
        if not isinstance(created_at, str) or not isinstance(updated_at, str):
            raise AgentError("invalid session timestamps")
        if not isinstance(title, str) or not isinstance(messages, list):
            raise AgentError("invalid session content")
        if any(not isinstance(message, dict) or message.get("role") not in MESSAGE_ROLES for message in messages):
            raise AgentError("invalid message in session")
        if messages and messages[0].get("role") != "system":
            raise AgentError("session history must start with a system message")
        return cls(session_id, workspace, created_at, updated_at, title, messages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SESSION_VERSION,
            "id": self.id,
            "workspace": self.workspace,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "title": self.title,
            "messages": self.messages,
        }


class SessionStore:
    """Store independent sessions as human-readable JSON files."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()

    def _path(self, session_id: str) -> Path:
        if not SESSION_ID_PATTERN.fullmatch(session_id):
            raise AgentError("invalid session id")
        return self.directory / f"{session_id}.json"

    def create(self, workspace: str | Path) -> AgentSession:
        now = _utc_now()
        workspace_path = str(Path(workspace).resolve())
        for _ in range(10):
            session_id = f"{datetime.now():%Y%m%d}-{uuid.uuid4().hex[:8]}"
            if not self._path(session_id).exists():
                session = AgentSession(session_id, workspace_path, now, now)
                self.save(session)
                return session
        raise AgentError("could not allocate a unique session id")

    def save(self, session: AgentSession) -> None:
        path = self._path(session.id)
        self.directory.mkdir(parents=True, exist_ok=True)
        session.updated_at = _utc_now()
        session.title = _title_from_messages(session.messages)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(session.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
            try:
                path.chmod(0o600)
            except OSError:
                pass
        except OSError as exc:
            raise AgentError(f"could not save session {session.id}: {exc}") from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def load(self, session_id: str) -> AgentSession:
        path = self._path(session_id)
        if not path.is_file():
            raise AgentError(f"session not found: {session_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AgentError(f"could not read session {session_id}: {exc}") from exc
        if not isinstance(payload, dict):
            raise AgentError(f"invalid session file: {session_id}")
        session = AgentSession.from_dict(payload)
        if session.id != session_id:
            raise AgentError(f"session id does not match filename: {session_id}")
        return session

    def list_sessions(self, workspace: str | Path | None = None) -> list[AgentSession]:
        if not self.directory.is_dir():
            return []
        expected_workspace = os.path.normcase(str(Path(workspace).resolve())) if workspace is not None else None
        sessions: list[AgentSession] = []
        for path in self.directory.glob("*.json"):
            try:
                session = self.load(path.stem)
            except AgentError:
                continue
            if expected_workspace is None or os.path.normcase(session.workspace) == expected_workspace:
                sessions.append(session)
        return sorted(sessions, key=lambda item: item.updated_at, reverse=True)

    def latest(self, workspace: str | Path) -> AgentSession | None:
        sessions = self.list_sessions(workspace)
        return sessions[0] if sessions else None
