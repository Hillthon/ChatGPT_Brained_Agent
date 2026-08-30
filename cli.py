"""Command-line entry point for the coding agent."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from agent import AgentConfig, make_agent


def main() -> int:
    parser = argparse.ArgumentParser(description="A small local coding agent")
    parser.add_argument("task", nargs="?", help="programming task; omit to read stdin")
    parser.add_argument("--root", default=".", help="workspace root (default: current directory)")
    parser.add_argument("--model", default=os.environ.get("CODING_AGENT_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--audit-log", help="write local JSONL audit records to this workspace-relative path")
    args = parser.parse_args()
    task = args.task or input("Task> ").strip()
    if not task:
        parser.error("task cannot be empty")

    def approve(action: str) -> bool:
        answer = input(f"Approve {action}? [y/N] ").strip().lower()
        return answer in {"y", "yes"}

    config = AgentConfig(model=args.model, base_url=args.base_url, max_steps=args.max_steps)
    try:
        agent = make_agent(Path(args.root), config=config, approve=approve, audit_path=args.audit_log)
    except Exception as exc:
        parser.error(str(exc))
    print(agent.run(task, on_event=lambda event: print(f"[{event}]")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
