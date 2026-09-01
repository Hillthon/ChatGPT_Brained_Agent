"""Command-line entry point for the coding agent."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from agent import AgentConfig, make_agent


def main() -> int:
    parser = argparse.ArgumentParser(description="A small local coding agent")
    parser.add_argument("task", nargs="?", help="programming task; omit to read stdin")
    parser.add_argument("--root", default="./working_directory", help="workspace root (default: current directory)")
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
    args = parser.parse_args()
    task = args.task or input("Task> ").strip()
    if not task:
        parser.error("task cannot be empty")

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
        agent = make_agent(
            Path(args.root),
            config=config,
            approve=approve,
            audit_path=args.audit_log,
            api_key=args.api_key,
        )
    except Exception as exc:
        parser.error(str(exc))
    print(agent.run(task, on_event=lambda event: print(f"[{event}]")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
