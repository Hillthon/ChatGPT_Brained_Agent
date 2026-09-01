"""Command-line entry point for the coding agent."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable

from agent import AgentConfig, AgentError, CodingAgent, make_agent
from session import AgentSession, SessionStore


COMMAND_HELP = """Session commands:
  /new                 start a new session
  /sessions            list saved sessions
  /switch <session-id> resume another session
  /help                show these commands
  /exit                 save and exit"""
DEFAULT_SESSION_DIR = Path.home() / ".coding-agent" / "sessions"


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


def _make_session_agent(
    session: AgentSession,
    store: SessionStore,
    root: Path,
    config: AgentConfig,
    approve: Callable[[str], bool],
    audit_path: str | Path | None,
    api_key: str | None,
) -> CodingAgent:
    def checkpoint(messages: list[dict[str, Any]]) -> None:
        session.messages = messages
        store.save(session)

    agent = make_agent(
        root,
        config=config,
        approve=approve,
        audit_path=audit_path,
        api_key=api_key,
        messages=session.messages,
        on_history_change=checkpoint,
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
    session_group = parser.add_mutually_exclusive_group()
    session_group.add_argument("--session", help="resume a saved session by id")
    session_group.add_argument("--continue-session", action="store_true", help="resume the latest session for this workspace")
    session_group.add_argument("--list-sessions", action="store_true", help="list saved sessions and exit")
    parser.add_argument("--once", action="store_true", help="run one task and exit instead of prompting again")
    args = parser.parse_args()

    store = SessionStore(args.session_dir)
    if args.list_sessions:
        print(format_sessions(store.list_sessions()))
        return 0
    root = Path(args.root).resolve()
    if not root.is_dir():
        parser.error(f"workspace is not a directory: {root}")

    def approve(action: str) -> bool:
        answer = input(f"Approve {action}? [y/N] ").strip().lower()
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
            session, store, root, config, approve, args.audit_log, args.api_key
        )
    except AgentError as exc:
        parser.error(str(exc))

    print(f"Session {session.id} ({session_status}). Type /help for session commands.")
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
                    session, store, root, config, approve, args.audit_log, args.api_key
                )
                print(f"Started session {session.id}.")
            except AgentError as exc:
                print(f"Error: {exc}")
        elif task.startswith("/switch"):
            session_id = task.removeprefix("/switch").strip()
            if not session_id:
                print("Usage: /switch <session-id>")
            else:
                try:
                    candidate = store.load(session_id)
                    _require_same_workspace(candidate, root)
                    agent = _make_session_agent(
                        candidate, store, root, config, approve, args.audit_log, args.api_key
                    )
                    session = candidate
                    print(f"Resumed session {session.id}: {session.title}")
                except AgentError as exc:
                    print(f"Error: {exc}")
        elif task.startswith("/"):
            print("Unknown command. Type /help for session commands.")
        else:
            try:
                print(agent.run(task, on_event=lambda event: print(f"[{event}]")))
            except KeyboardInterrupt:
                print("\nInterrupted; session checkpoint was preserved.")
                if args.once:
                    return 130
            except AgentError as exc:
                print(f"Error: {exc}")
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
