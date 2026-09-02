"""Interactive terminal entry point for the coding agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, TextIO

from agent import AgentConfig, AgentError, CodingAgent, AgentEvent, make_agent
from session import AgentSession, SessionStore
from undo import UndoManager


COMMAND_HELP = """Session commands:
  /new                 start a new session
  /sessions            list saved sessions
  /switch <session-id> resume another session
  /undo                undo the latest agent file edit
  /rollback            undo all file edits from the latest changed task
  /checkpoints         list reversible file edits
  /help                show these commands
  /exit                 save and exit"""
DEFAULT_SESSION_DIR = Path.home() / ".coding-agent" / "sessions"
DEFAULT_UNDO_DIR = Path.home() / ".coding-agent" / "snapshots"


def format_sessions(sessions: list[AgentSession], current_id: str | None = None) -> str:
    if not sessions:
        return "(no saved sessions)"
    lines = []
    for session in sessions:
        marker = "*" if session.id == current_id else " "
        workspace = Path(session.workspace).name or session.workspace
        lines.append(f"{marker} {session.id}  {session.updated_at}  [{workspace}] {session.title}")
    return "\n".join(lines)


def _require_same_workspace(session: AgentSession, root: Path) -> None:
    if os.path.normcase(session.workspace) != os.path.normcase(str(root)):
        raise AgentError(
            f"session {session.id} belongs to {session.workspace}; current workspace is {root}"
        )


class Renderer:
    """Translate structured agent events into compact, terminal-friendly output."""

    COLORS = {"red": "\x1b[31m", "green": "\x1b[32m", "yellow": "\x1b[33m", "cyan": "\x1b[36m"}
    RESET = "\x1b[0m"

    def __init__(self, verbosity: int = 0, quiet: bool = False, stream: TextIO | None = None,
                 color: bool | None = None):
        self.verbosity = verbosity
        self.quiet = quiet
        self.stream = stream or sys.stdout
        self.color = bool(color) if color is not None else bool(
            getattr(self.stream, "isatty", lambda: False)() and not os.environ.get("NO_COLOR")
        )
        self._status_active = False
        self._status_width = 0
        self._answer_started = False

    def _paint(self, text: str, color: str) -> str:
        return f"{self.COLORS[color]}{text}{self.RESET}" if self.color else text

    def _write(self, text: str) -> None:
        # Windows terminals can still expose a legacy GBK/CP936 stream. Keep the
        # richer glyphs when supported, but never let a status icon crash a run.
        encoding = getattr(self.stream, "encoding", None)
        if encoding:
            replacements = {"⠋": "...", "✓": "+", "✗": "x", "⏱": "!", "→": "->", "·": "-", "…": "..."}
            text = "".join(replacements.get(char, char) for char in text)
            try:
                text.encode(encoding)
            except UnicodeEncodeError:
                text = text.encode(encoding, errors="replace").decode(encoding)
        self.stream.write(text)
        self.stream.flush()

    def _line(self, text: str) -> None:
        self._write(text.rstrip("\n") + "\n")

    @staticmethod
    def _one_line(value: Any) -> str:
        return " ".join(str(value or "").split())

    def _clear_status(self) -> None:
        if self._status_active:
            if self.color or getattr(self.stream, "isatty", lambda: False)():
                self._write("\r\x1b[2K")
            else:
                self._write("\r" + (" " * self._status_width) + "\r")
            self._status_active = False
            self._status_width = 0

    def _finish_answer_line(self) -> None:
        if self._answer_started:
            self._write("\n")
            self._answer_started = False

    @staticmethod
    def _short_path(value: Any) -> str:
        return str(value or ".").replace("\\", "/")

    @staticmethod
    def _size(value: Any) -> str:
        try:
            amount = int(value)
        except (TypeError, ValueError):
            amount = 0
        if amount < 1024:
            return f"{amount}B"
        if amount < 1024 * 1024:
            return f"{amount / 1024:.1f}KB"
        return f"{amount / (1024 * 1024):.1f}MB"

    @staticmethod
    def _duration(seconds: Any) -> str:
        try:
            value = float(seconds)
        except (TypeError, ValueError):
            return ""
        return f", {value:.1f}s" if value > 1 else ""

    def _action(self, name: str, arguments: Any) -> str:
        args = arguments if isinstance(arguments, dict) else {}
        if name == "read_file":
            start = args.get("start_line", 1)
            end = args.get("end_line") or "end"
            return f"read_file  {self._short_path(args.get('path'))}:{start}-{end}"
        if name == "search_files":
            query = json.dumps(str(args.get("query", "")), ensure_ascii=False)
            return f"search_files  {query} in {self._short_path(args.get('path', '.'))}"
        if name == "list_files":
            return f"list_files  {self._short_path(args.get('path', '.'))}"
        if name == "write_file":
            content = args.get("content", "")
            return f"write_file  {self._short_path(args.get('path'))} ({self._size(len(str(content).encode('utf-8')))})"
        if name == "apply_patch":
            return f"apply_patch  {self._short_path(args.get('path'))}"
        if name == "run_command":
            return f"run_command  {str(args.get('command', '')).strip()}"
        return f"{name}  {json.dumps(args, ensure_ascii=False, separators=(',', ':'))}"

    def approval_prompt(self, action: str) -> str:
        if action.startswith("run command"):
            return f"? Run (rollback not guaranteed): {action.split(':', 1)[1].strip()} [y/N] "
        if action.startswith("patch "):
            return f"? Apply patch: {action[6:]} [y/N] "
        if action.startswith("write "):
            return f"? Write: {action[6:]} [y/N] "
        return f"? {action} [y/N] "

    def _render_diff(self, diff: str) -> None:
        if not diff:
            return
        for line in diff.rstrip("\n").splitlines():
            if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
                painted = self._paint(line, "cyan")
            elif line.startswith("+"):
                painted = self._paint(line, "green")
            elif line.startswith("-"):
                painted = self._paint(line, "red")
            else:
                painted = line
            self._line(f"   {painted}")

    @staticmethod
    def _command_output(result: str, limit: int | None = None) -> list[str]:
        lines = result.splitlines()
        if lines and lines[0].startswith("exit_code="):
            lines = lines[1:]
        return lines[-limit:] if limit and len(lines) > limit else lines

    @staticmethod
    def _search_count(result: str) -> str:
        if result == "(no matches)":
            return "0 matches"
        count = sum(1 for line in result.splitlines() if line and not line.startswith("(results truncated)"))
        suffix = "+" if "(results truncated)" in result else ""
        return f"{count}{suffix} matches"

    def _tool_end(self, data: dict[str, Any]) -> None:
        name = str(data.get("name", "tool"))
        arguments = data.get("arguments", {})
        result = str(data.get("result", ""))
        display_result = str(data.get("full_result") or result)
        ok = bool(data.get("ok"))
        elapsed = self._duration(data.get("elapsed"))
        if ok:
            marker = self._paint("✓", "green")
        elif result.startswith("TIMEOUT"):
            marker = self._paint("⏱", "yellow")
        else:
            marker = self._paint("✗", "red")
        suffix = elapsed
        if name == "search_files" and ok:
            suffix += f", {self._search_count(result)}"
        if name == "run_command":
            match = result.split("\n", 1)[0] if result else ""
            if match.startswith("exit_code="):
                suffix += f", {match.replace('exit_code=', 'exit ')}"
        label = self._action(name, arguments)
        if not ok:
            reason = self._one_line(data.get("error") or result.split("\n", 1)[0] or "tool failed")
            self._line(f"{marker} {label}{suffix} - {reason}")
        else:
            self._line(f"{marker} {label}{suffix}")

        if name == "run_command" and result and not result.startswith(("DENIED", "ERROR", "TIMEOUT")):
            lines = self._command_output(display_result, None if self.verbosity else 20)
            if lines:
                for line in lines:
                    self._line(f"   {line}")
        elif self.verbosity and display_result and name in {"read_file", "search_files", "list_files"}:
            for line in display_result.splitlines():
                self._line(f"   {line}")

    def on_event(self, event: AgentEvent) -> None:
        if not isinstance(event, tuple) or len(event) != 2:
            return
        kind, payload = event
        if kind == "thinking":
            self._clear_status()
            self._finish_answer_line()
            status = "⠋ Thinking…"
            self._write("\r" + status)
            self._status_width = len(status)
            self._status_active = True
        elif kind == "tool_call":
            if self.verbosity >= 2:
                self._clear_status()
                self._finish_answer_line()
                self._line("· tool_call " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        elif kind in {"model_request", "model_response", "model_chunk"}:
            if self.verbosity >= 2:
                self._clear_status()
                self._finish_answer_line()
                self._line(f"· {kind} " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        elif kind == "tool_start":
            data = payload if isinstance(payload, dict) else {}
            self._clear_status()
            self._finish_answer_line()
            if self.quiet:
                return
            action = self._action(str(data.get("name", "tool")), data.get("arguments", {}))
            preview = data.get("preview")
            if isinstance(preview, dict) and preview.get("kind") == "diff":
                self._line("→ " + action)
                self._render_diff(str(preview.get("diff", "")))
            else:
                status = "⠋ " + action
                self._write("\r" + status)
                self._status_width = len(status)
                self._status_active = True
        elif kind == "tool_end":
            self._clear_status()
            if not self.quiet:
                self._tool_end(payload if isinstance(payload, dict) else {})
        elif kind == "assistant_delta":
            self._clear_status()
            if isinstance(payload, str) and payload:
                self._write(payload)
                self._answer_started = True
        elif kind == "run_end":
            self._clear_status()
            self._finish_answer_line()
            data = payload if isinstance(payload, dict) else {}
            if not self.quiet:
                steps = data.get("steps", "?")
                elapsed = float(data.get("elapsed", 0) or 0)
                usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
                total = usage.get("total_tokens")
                if not isinstance(total, (int, float)):
                    total = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
                tokens = f" · {int(total):,} tokens" if total else ""
                self._line(f"本轮 {steps} 步{tokens} · {elapsed:.1f}s")
            self._answer_started = False

    def error(self, error: BaseException) -> None:
        self._clear_status()
        self._line(self._paint(f"Error: {self._one_line(error)}", "red"))
        if self.verbosity >= 2:
            traceback.print_exc(file=self.stream)


def _make_session_agent(
    session: AgentSession,
    store: SessionStore,
    root: Path,
    config: AgentConfig,
    approve: Callable[[str], bool],
    audit_path: str | Path | None,
    api_key: str | None,
    undo_directory: str | Path,
) -> CodingAgent:
    def checkpoint(messages: list[dict[str, Any]]) -> None:
        session.messages = messages
        store.save(session)

    undo_manager = UndoManager(undo_directory, session.id, root)
    agent = make_agent(
        root,
        config=config,
        approve=approve,
        audit_path=audit_path,
        api_key=api_key,
        messages=session.messages,
        on_history_change=checkpoint,
        undo_manager=undo_manager,
    )
    checkpoint(agent.messages)
    return agent


def main() -> int:
    parser = argparse.ArgumentParser(description="A small local coding agent")
    parser.add_argument("task", nargs="?", help="initial programming task; more tasks can follow interactively")
    parser.add_argument("--root", default="./working_directory", help="workspace root")
    parser.add_argument("--model", default=os.environ.get("CODING_AGENT_MODEL", "gpt-5.6-luna"))
    parser.add_argument(
        "--base-url",
        default=(os.environ.get("CODING_AGENT_BASE_URL") or os.environ.get("RIGHTAPI_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://rightapi.ai/codex/v1"),
    )
    parser.add_argument(
        "--api-key",
        default=(os.environ.get("CODING_AGENT_API_KEY") or os.environ.get("RIGHTAPI_API_KEY") or os.environ.get("OPENAI_API_KEY")),
        help="API key (prefer an environment variable)",
    )
    parser.add_argument(
        "--api-mode",
        choices=("auto", "chat", "responses"),
        default=os.environ.get("CODING_AGENT_API_MODE", "auto"),
        help="relay protocol (default: auto)",
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--request-timeout", type=int, default=120, help="model request timeout in seconds")
    parser.add_argument("--audit-log", help="write local JSONL audit records to this workspace-relative path")
    parser.add_argument(
        "--session-dir",
        default=os.environ.get("CODING_AGENT_SESSION_DIR", str(DEFAULT_SESSION_DIR)),
        help="directory for local session JSON files",
    )
    parser.add_argument(
        "--undo-dir",
        default=os.environ.get("CODING_AGENT_UNDO_DIR", str(DEFAULT_UNDO_DIR)),
        help="directory for rollback indexes and file snapshots",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="show tool results; repeat for raw tool calls")
    parser.add_argument("-q", "--quiet", action="store_true", help="hide tool summaries and show the answer stream")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    session_group = parser.add_mutually_exclusive_group()
    session_group.add_argument("--session", help="resume a saved session by id")
    session_group.add_argument("--continue-session", action="store_true", help="resume the latest session for this workspace")
    session_group.add_argument("--list-sessions", action="store_true", help="list saved sessions and exit")
    parser.add_argument("--once", action="store_true", help="run one task and exit instead of prompting again")
    args = parser.parse_args()

    renderer = Renderer(args.verbose or 0, args.quiet, color=False if args.no_color else None)
    store = SessionStore(args.session_dir)
    if args.list_sessions:
        print(format_sessions(store.list_sessions()))
        return 0
    root = Path(args.root).resolve()
    if not root.is_dir():
        parser.error(f"workspace is not a directory: {root}")

    def approve(action: str) -> bool:
        answer = input(renderer.approval_prompt(action)).strip().lower()
        return answer in {"y", "yes"}

    config = AgentConfig(
        model=args.model,
        base_url=args.base_url,
        max_steps=args.max_steps,
        api_mode=args.api_mode,
        request_timeout=args.request_timeout,
    )
    try:
        if args.session:
            session = store.load(args.session)
            _require_same_workspace(session, root)
            session_status = "resumed"
        elif args.continue_session:
            session = store.latest(root) or store.create(root)
            session_status = "resumed" if session.messages else "new"
        else:
            session = store.create(root)
            session_status = "new"
        agent = _make_session_agent(
            session, store, root, config, approve, args.audit_log, args.api_key, args.undo_dir
        )
    except Exception as exc:
        renderer.error(exc)
        return 1

    if not args.quiet:
        print(f"Session {session.id} ({session_status})")
    task = args.task
    while True:
        if task is None:
            try:
                task = input(f"[{session.id}] Task> ").strip()
            except EOFError:
                print()
                return 0
        else:
            task = task.strip()

        if not task:
            if args.once:
                parser.error("task cannot be empty")
            task = None
            continue
        if task in {"/exit", "/quit"}:
            return 0
        if task == "/help":
            print(COMMAND_HELP)
        elif task == "/sessions":
            print(format_sessions(store.list_sessions(), session.id))
        elif task == "/new":
            try:
                session = store.create(root)
                agent = _make_session_agent(
                    session, store, root, config, approve, args.audit_log, args.api_key, args.undo_dir
                )
                if not args.quiet:
                    print(f"Session {session.id} (new)")
            except Exception as exc:
                renderer.error(exc)
        elif task.startswith("/switch"):
            session_id = task.removeprefix("/switch").strip()
            if not session_id:
                print("Usage: /switch <session-id>")
            else:
                try:
                    candidate = store.load(session_id)
                    _require_same_workspace(candidate, root)
                    agent = _make_session_agent(
                        candidate, store, root, config, approve, args.audit_log, args.api_key, args.undo_dir
                    )
                    session = candidate
                    if not args.quiet:
                        print(f"Session {session.id} (resumed)")
                except Exception as exc:
                    renderer.error(exc)
        elif task == "/undo":
            try:
                action = agent.workspace.undo_last()
                agent.record_control_event(
                    f"Undid the latest {action.tool} edit to {action.path}. Reread the file before editing it again."
                )
                print(f"Undid {action.tool}: {action.path} [{action.id[:8]}]")
            except Exception as exc:
                renderer.error(exc)
        elif task == "/rollback":
            try:
                changed_task, actions = agent.workspace.rollback_latest_task()
                paths = list(dict.fromkeys(action.path for action in actions))
                agent.record_control_event(
                    "Rolled back the latest changed task and restored: " + ", ".join(paths)
                )
                print(
                    f"Rolled back {len(actions)} edit(s) from task {changed_task.id[:8]}: "
                    + ", ".join(paths)
                )
            except Exception as exc:
                renderer.error(exc)
        elif task == "/checkpoints":
            actions = agent.workspace.undo_checkpoints()
            if not actions:
                print("(no reversible agent edits)")
            else:
                for action in reversed(actions):
                    print(
                        f"{action.id[:8]}  task {action.task_id[:8]}  "
                        f"{action.tool}  {action.path}"
                    )
        elif task.startswith("/"):
            print("Unknown command. Type /help for session commands.")
        else:
            try:
                agent.run(task, on_event=renderer.on_event)
            except KeyboardInterrupt:
                renderer._clear_status()
                renderer._line("Interrupted; session checkpoint was preserved.")
                if args.once:
                    return 130
            except Exception as exc:
                renderer.error(exc)
                if args.once:
                    return 1
            else:
                if args.once:
                    return 0
        if args.once:
            return 0
        task = None


if __name__ == "__main__":
    raise SystemExit(main())
