"""Persistent, conflict-aware rollback for agent-owned file edits."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


UNDO_VERSION = 1
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class UndoError(RuntimeError):
    """A rollback operation could not be completed safely."""


class UndoConflict(UndoError):
    """A file changed after the recorded agent edit."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass
class PendingSnapshot:
    id: str
    task_id: str
    tool: str
    path: str
    existed_before: bool
    content_before: bytes | None
    mode_before: int | None
    hash_before: str | None


@dataclass
class UndoAction:
    id: str
    task_id: str
    tool: str
    path: str
    existed_before: bool
    mode_before: int | None
    hash_before: str | None
    hash_after: str
    created_at: str
    status: str = "active"
    undone_at: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UndoAction":
        try:
            action = cls(**value)
        except TypeError as exc:
            raise UndoError(f"invalid undo action: {exc}") from exc
        if action.status not in {"active", "undone"}:
            raise UndoError(f"invalid undo action status: {action.status}")
        return action


@dataclass
class UndoTask:
    id: str
    prompt: str
    created_at: str
    status: str = "running"
    finished_at: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UndoTask":
        try:
            return cls(**value)
        except TypeError as exc:
            raise UndoError(f"invalid undo task: {exc}") from exc


class UndoManager:
    """Store pre-edit bytes and restore only unchanged agent outputs.

    Snapshots live outside the model-visible workspace. Each session owns an
    independent index and blob directory, so one session cannot undo another
    session's operations accidentally.
    """

    def __init__(self, directory: str | Path, session_id: str, workspace: str | Path,
                 max_snapshot_bytes: int = 20 * 1024 * 1024):
        if not SESSION_ID_PATTERN.fullmatch(session_id):
            raise UndoError("invalid session id")
        if max_snapshot_bytes < 1:
            raise UndoError("max snapshot size must be positive")
        self.workspace = Path(workspace).resolve()
        self.directory = (Path(directory).resolve() / session_id).resolve()
        self.index_path = self.directory / "index.json"
        self.session_id = session_id
        self.max_snapshot_bytes = max_snapshot_bytes
        self.tasks: list[UndoTask] = []
        self.actions: list[UndoAction] = []
        self.current_task_id: str | None = None
        self._load()
        if not self.index_path.exists():
            self._save()

    def _resolve_workspace_path(self, relative: str) -> Path:
        candidate = (self.workspace / relative).resolve()
        try:
            candidate.relative_to(self.workspace)
        except ValueError as exc:
            raise UndoError("snapshot path escapes workspace") from exc
        return candidate

    def _blob_path(self, action_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", action_id):
            raise UndoError("invalid undo action id")
        return self.directory / f"{action_id}.bin"

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise UndoError(f"could not read undo index: {exc}") from exc
        if payload.get("version") != UNDO_VERSION:
            raise UndoError(f"unsupported undo version: {payload.get('version')!r}")
        if payload.get("session_id") != self.session_id:
            raise UndoError("undo index belongs to another session")
        stored_workspace = payload.get("workspace")
        if not isinstance(stored_workspace, str) or os.path.normcase(stored_workspace) != os.path.normcase(str(self.workspace)):
            raise UndoError("undo index belongs to another workspace")
        tasks = payload.get("tasks")
        actions = payload.get("actions")
        if not isinstance(tasks, list) or not isinstance(actions, list):
            raise UndoError("invalid undo index")
        self.tasks = [UndoTask.from_dict(item) for item in tasks if isinstance(item, dict)]
        self.actions = [UndoAction.from_dict(item) for item in actions if isinstance(item, dict)]
        # A process may have stopped before finish_task. Starting the next task
        # will mark an old running task as interrupted.
        running = [task.id for task in self.tasks if task.status == "running"]
        self.current_task_id = running[-1] if running else None

    def _save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": UNDO_VERSION,
            "session_id": self.session_id,
            "workspace": str(self.workspace),
            "tasks": [asdict(task) for task in self.tasks],
            "actions": [asdict(action) for action in self.actions],
        }
        temporary = self.directory / f".{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.index_path)
            for path in (self.directory, self.index_path):
                try:
                    path.chmod(0o700 if path.is_dir() else 0o600)
                except OSError:
                    pass
        except OSError as exc:
            raise UndoError(f"could not save undo index: {exc}") from exc
        finally:
            temporary.unlink(missing_ok=True)

    def begin_task(self, prompt: str) -> str:
        if self.current_task_id:
            self.finish_task(self.current_task_id, "interrupted")
        task_id = uuid.uuid4().hex
        self.tasks.append(UndoTask(task_id, " ".join(prompt.split())[:200], _utc_now()))
        self.current_task_id = task_id
        self._save()
        return task_id

    def finish_task(self, task_id: str | None, status: str) -> None:
        if not task_id:
            return
        for task in self.tasks:
            if task.id == task_id:
                task.status = status
                task.finished_at = _utc_now()
                if self.current_task_id == task_id:
                    self.current_task_id = None
                self._save()
                return
        raise UndoError(f"undo task not found: {task_id}")

    def capture(self, file: str | Path, tool: str) -> PendingSnapshot:
        if not self.current_task_id:
            raise UndoError("no active task for reversible edit")
        path = Path(file).resolve()
        try:
            relative = str(path.relative_to(self.workspace))
        except ValueError as exc:
            raise UndoError("snapshot path escapes workspace") from exc
        existed = path.exists()
        if existed and not path.is_file():
            raise UndoError(f"cannot snapshot non-file path: {relative}")
        content = path.read_bytes() if existed else None
        if content is not None and len(content) > self.max_snapshot_bytes:
            raise UndoError(
                f"file is too large for rollback ({len(content)} bytes > {self.max_snapshot_bytes}); edit was not applied"
            )
        mode = stat.S_IMODE(path.stat().st_mode) if existed else None
        return PendingSnapshot(
            id=uuid.uuid4().hex,
            task_id=self.current_task_id,
            tool=tool,
            path=relative,
            existed_before=existed,
            content_before=content,
            mode_before=mode,
            hash_before=_sha256(content) if content is not None else None,
        )

    def commit(self, pending: PendingSnapshot) -> UndoAction | None:
        path = self._resolve_workspace_path(pending.path)
        if not path.is_file():
            raise UndoError(f"edited file is missing after {pending.tool}: {pending.path}")
        content_after = path.read_bytes()
        hash_after = _sha256(content_after)
        if pending.existed_before and hash_after == pending.hash_before:
            return None
        action = UndoAction(
            id=pending.id,
            task_id=pending.task_id,
            tool=pending.tool,
            path=pending.path,
            existed_before=pending.existed_before,
            mode_before=pending.mode_before,
            hash_before=pending.hash_before,
            hash_after=hash_after,
            created_at=_utc_now(),
        )
        blob = self._blob_path(pending.id)
        try:
            if pending.existed_before and pending.content_before is not None:
                blob.parent.mkdir(parents=True, exist_ok=True)
                temporary = blob.with_name(f".{blob.name}.{uuid.uuid4().hex}.tmp")
                try:
                    temporary.write_bytes(pending.content_before)
                    temporary.replace(blob)
                    try:
                        blob.chmod(0o600)
                    except OSError:
                        pass
                finally:
                    temporary.unlink(missing_ok=True)
            self.actions.append(action)
            self._save()
        except Exception as exc:
            if self.actions and self.actions[-1] is action:
                self.actions.pop()
            try:
                self._restore_pending(pending)
                blob.unlink(missing_ok=True)
            except Exception as restore_exc:
                raise UndoError(
                    f"could not save rollback checkpoint and could not restore {pending.path}: {restore_exc}"
                ) from exc
            raise UndoError(f"edit was restored because its rollback checkpoint could not be saved: {exc}") from exc
        return action

    def _restore_pending(self, pending: PendingSnapshot) -> None:
        path = self._resolve_workspace_path(pending.path)
        if pending.existed_before:
            if pending.content_before is None:
                raise UndoError(f"missing pending snapshot data for {pending.path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            self._replace_bytes(path, pending.content_before, pending.mode_before)
        else:
            path.unlink(missing_ok=True)

    @staticmethod
    def _replace_bytes(path: Path, content: bytes, mode: int | None) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.undo")
        try:
            temporary.write_bytes(content)
            if mode is not None:
                try:
                    temporary.chmod(mode)
                except OSError:
                    pass
            temporary.replace(path)
            if mode is not None:
                try:
                    path.chmod(mode)
                except OSError:
                    pass
        finally:
            temporary.unlink(missing_ok=True)

    def active_actions(self, task_id: str | None = None) -> list[UndoAction]:
        return [
            action for action in self.actions
            if action.status == "active" and (task_id is None or action.task_id == task_id)
        ]

    def latest_task_with_actions(self) -> UndoTask | None:
        task_ids = {action.task_id for action in self.active_actions()}
        return next((task for task in reversed(self.tasks) if task.id in task_ids), None)

    def _verify_current(self, action: UndoAction) -> Path:
        path = self._resolve_workspace_path(action.path)
        if not path.is_file():
            raise UndoConflict(f"cannot undo {action.path}: file was deleted or replaced")
        current_hash = _sha256(path.read_bytes())
        if current_hash != action.hash_after:
            raise UndoConflict(f"cannot undo {action.path}: file changed after the agent edit")
        return path

    def _restore(self, action: UndoAction) -> None:
        path = self._verify_current(action)
        if action.existed_before:
            blob = self._blob_path(action.id)
            if not blob.is_file():
                raise UndoError(f"snapshot data is missing for {action.path}")
            content = blob.read_bytes()
            if _sha256(content) != action.hash_before:
                raise UndoError(f"snapshot data is corrupt for {action.path}")
            self._replace_bytes(path, content, action.mode_before)
        else:
            path.unlink()
        action.status = "undone"
        action.undone_at = _utc_now()
        self._save()

    def undo_last(self) -> UndoAction:
        active = self.active_actions()
        if not active:
            raise UndoError("there are no reversible agent edits")
        action = active[-1]
        self._restore(action)
        return action

    def rollback_latest_task(self) -> tuple[UndoTask, list[UndoAction]]:
        task = self.latest_task_with_actions()
        if task is None:
            raise UndoError("there is no task with reversible agent edits")
        actions = self.active_actions(task.id)
        # Only the newest action for each path can match the current file. This
        # preflight prevents a partial rollback when a user edited any file.
        newest_by_path: dict[str, UndoAction] = {}
        for action in actions:
            newest_by_path[action.path] = action
        for action in newest_by_path.values():
            self._verify_current(action)
        restored: list[UndoAction] = []
        for action in reversed(actions):
            self._restore(action)
            restored.append(action)
        task.status = "rolled_back"
        task.finished_at = _utc_now()
        self._save()
        return task, restored
