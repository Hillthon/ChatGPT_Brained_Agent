"""A small, framework-free coding agent.

The model only proposes tool calls. Files and commands are always handled by
this process, which keeps the important execution logic local and inspectable.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class AgentError(RuntimeError):
    """An expected, user-facing agent error."""


class ModelError(AgentError):
    """The model endpoint could not be called or parsed."""


@dataclass
class AgentConfig:
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    max_steps: int = 20
    max_context_chars: int = 80_000
    command_timeout: int = 30
    max_output_chars: int = 12_000


class OpenAICompatibleClient:
    """Minimal Chat Completions client using only the Python standard library."""

    def __init__(self, config: AgentConfig, api_key: str | None = None):
        self.config = config
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ModelError("OPENAI_API_KEY is not set")

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        body = {"model": self.config.model, "messages": messages, "tools": tools, "tool_choice": "auto"}
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelError(f"model request failed: {exc}") from exc
        try:
            return payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelError(f"unexpected model response: {payload!r}") from exc


def _json_schema(type_: str, description: str, **properties: Any) -> dict[str, Any]:
    required = [name for name, schema in properties.items() if "default" not in schema]
    return {"type": "function", "function": {"name": type_, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required, "additionalProperties": False,
    }}}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _json_schema("list_files", "List files under a workspace-relative directory.", path={"type": "string", "description": "Relative directory", "default": "."}),
    _json_schema("read_file", "Read a UTF-8 text file with line numbers.", path={"type": "string"}, start_line={"type": "integer", "minimum": 1, "default": 1}, end_line={"type": ["integer", "null"], "minimum": 1, "default": None}),
    _json_schema("search_files", "Search text in UTF-8 files below a directory.", query={"type": "string"}, path={"type": "string", "default": "."}),
    _json_schema("write_file", "Create or replace a UTF-8 file. Use only when the user requested a change.", path={"type": "string"}, content={"type": "string"}),
    _json_schema("apply_patch", "Apply a unified diff to one workspace-relative file.", path={"type": "string"}, patch={"type": "string"}),
    _json_schema("run_command", "Run a project command in the workspace after confirmation.", command={"type": "string"}, timeout={"type": "integer", "minimum": 1, "maximum": 120, "default": 30}),
]


class Workspace:
    """Local tools with path and command guardrails."""

    BLOCKED_COMMANDS = re.compile(
        r"(?:^|[\s;&|])(?:rm|del|erase|rmdir|format|shutdown|reboot|diskpart|remove-item)(?:\s|$)|"
        r"(?:^|[\s;&|])git\s+reset\s+--hard(?:\s|$)", re.I
    )

    def __init__(self, root: str | Path, approve: Callable[[str], bool] | None = None,
                 config: AgentConfig | None = None, audit_path: str | Path | None = None):
        self.root = Path(root).resolve()
        self.approve = approve or (lambda _: False)
        self.config = config or AgentConfig()
        self.audit_path = self.resolve(audit_path) if audit_path else None

    def _audit(self, event: str, **data: Any) -> None:
        if not self.audit_path:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **data}
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def resolve(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise AgentError("path escapes workspace") from exc
        return candidate

    def list_files(self, path: str = ".") -> str:
        directory = self.resolve(path)
        if not directory.is_dir():
            raise AgentError(f"not a directory: {path}")
        entries = []
        for item in sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            if item.name in {".git", ".venv", "node_modules", "__pycache__"}:
                continue
            entries.append(f"{item.relative_to(self.root)}{'/' if item.is_dir() else ''}")
        return "\n".join(entries) or "(empty)"

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        file = self.resolve(path)
        if not file.is_file():
            raise AgentError(f"not a file: {path}")
        try:
            lines = file.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise AgentError("only UTF-8 text files are supported") from exc
        first = max(1, start_line)
        last = end_line or len(lines)
        if first > last:
            return ""
        shown = [f"{i:>5} | {lines[i - 1]}" for i in range(first, min(last, len(lines)) + 1)]
        return "\n".join(shown)[: self.config.max_output_chars]

    def search_files(self, query: str, path: str = ".") -> str:
        base = self.resolve(path)
        if not base.is_dir():
            raise AgentError(f"not a directory: {path}")
        results: list[str] = []
        for file in base.rglob("*"):
            if not file.is_file() or any(part in {".git", ".venv", "node_modules", "__pycache__"} for part in file.parts):
                continue
            try:
                for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
                    if query.lower() in line.lower():
                        results.append(f"{file.relative_to(self.root)}:{number}: {line.strip()}")
                        if len(results) >= 100:
                            return "\n".join(results) + "\n(results truncated)"
            except (UnicodeDecodeError, OSError):
                continue
        return "\n".join(results) or "(no matches)"

    def write_file(self, path: str, content: str) -> str:
        file = self.resolve(path)
        prompt = f"write {file.relative_to(self.root)} ({len(content)} chars)"
        if not self.approve(prompt):
            self._audit("denied", action=prompt)
            return "DENIED: user did not approve file write"
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content, encoding="utf-8")
        self._audit("write_file", path=str(file.relative_to(self.root)), chars=len(content))
        return f"wrote {file.relative_to(self.root)} ({len(content)} chars)"

    def apply_patch(self, path: str, patch: str) -> str:
        file = self.resolve(path)
        if not file.is_file():
            raise AgentError(f"not a file: {path}")
        old = file.read_text(encoding="utf-8").splitlines(keepends=True)
        new = self._apply_unified_diff(old, patch)
        if not new or new == old:
            raise AgentError("patch did not change file")
        if not self.approve(f"patch {file.relative_to(self.root)}"):
            self._audit("denied", action=f"patch {file.relative_to(self.root)}")
            return "DENIED: user did not approve patch"
        file.write_text("".join(new), encoding="utf-8")
        self._audit("apply_patch", path=str(file.relative_to(self.root)))
        return f"patched {file.relative_to(self.root)}"

    @staticmethod
    def _apply_unified_diff(old: list[str], patch: str) -> list[str]:
        """Apply a single-file unified diff while checking every context line."""
        if patch.lstrip().startswith("```"):
            fenced = patch.lstrip().splitlines(keepends=True)
            if len(fenced) < 3 or not fenced[-1].strip().startswith("```"):
                raise AgentError("invalid patch: unclosed code fence")
            patch = "".join(fenced[1:-1])
        lines = patch.splitlines(keepends=True)
        hunk_indexes = [i for i, line in enumerate(lines) if line.startswith("@@")]
        if not hunk_indexes:
            raise AgentError("invalid patch: missing @@ hunk header")
        result: list[str] = []
        old_cursor = 0
        for index, hunk_start in enumerate(hunk_indexes):
            header = lines[hunk_start]
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", header)
            if not match:
                raise AgentError(f"invalid patch hunk header: {header.strip()}")
            source_start = int(match.group(1)) - 1
            if source_start < old_cursor or source_start > len(old):
                raise AgentError("patch context is outside the current file")
            result.extend(old[old_cursor:source_start])
            body_end = hunk_indexes[index + 1] if index + 1 < len(hunk_indexes) else len(lines)
            cursor = source_start
            for line in lines[hunk_start + 1:body_end]:
                if line.startswith("\\ No newline"):
                    continue
                if not line:
                    continue
                marker, content = line[0], line[1:]
                if marker == " ":
                    if cursor >= len(old) or old[cursor] != content:
                        raise AgentError("patch context does not match file")
                    result.append(content)
                    cursor += 1
                elif marker == "-":
                    if cursor >= len(old) or old[cursor] != content:
                        raise AgentError("patch removal does not match file")
                    cursor += 1
                elif marker == "+":
                    result.append(content)
                else:
                    raise AgentError(f"invalid patch line: {line.strip()}")
            old_cursor = cursor
        result.extend(old[old_cursor:])
        return result

    def run_command(self, command: str, timeout: int | None = None) -> str:
        command = command.strip()
        if not command:
            raise AgentError("command is empty")
        if self.BLOCKED_COMMANDS.search(command):
            self._audit("blocked", action=command, reason="destructive command")
            return "DENIED: destructive command is blocked"
        # Keep shell use explicit and visible; approval is required for every invocation.
        if not self.approve(f"run command: {command}"):
            self._audit("denied", action=command)
            return "DENIED: user did not approve command"
        try:
            result = subprocess.run(command, cwd=self.root, shell=True, capture_output=True, text=True, errors="replace",
                                    timeout=min(timeout or self.config.command_timeout, 120))
        except subprocess.TimeoutExpired as exc:
            return f"TIMEOUT after {exc.timeout}s"
        output = (result.stdout + ("\n" + result.stderr if result.stderr else "")).strip()
        output = output[-self.config.max_output_chars:]
        self._audit("run_command", command=command, exit_code=result.returncode)
        return f"exit_code={result.returncode}\n{output}" if output else f"exit_code={result.returncode}"

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        methods = {"list_files": self.list_files, "read_file": self.read_file, "search_files": self.search_files,
                   "write_file": self.write_file, "apply_patch": self.apply_patch, "run_command": self.run_command}
        if name not in methods:
            raise AgentError(f"unknown tool: {name}")
        return methods[name](**arguments)


SYSTEM_PROMPT = """You are a careful coding agent working in a local workspace.
Inspect before editing. Use tools for all file reads and changes; do not pretend a change happened.
Explain a concise plan, make the smallest useful edits, and run relevant tests when approved.
Never request secrets, access paths outside the workspace, or destructive commands.
When the task is complete, summarize changed files and verification. If blocked, say exactly why."""


class CodingAgent:
    def __init__(self, client: OpenAICompatibleClient, workspace: Workspace, config: AgentConfig | None = None):
        self.client = client
        self.workspace = workspace
        self.config = config or client.config
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _trim_context(self) -> None:
        serialized = json.dumps(self.messages, ensure_ascii=False)
        if len(serialized) <= self.config.max_context_chars:
            return
        # Preserve system prompt and the newest turns. This is deterministic and avoids silently
        # dropping the current tool result, which is more useful than old transcript detail.
        system = self.messages[0]
        kept: list[dict[str, Any]] = []
        size = len(json.dumps(system, ensure_ascii=False))
        index = len(self.messages)
        while index > 1:
            # Keep an assistant tool-call message together with all immediately following
            # tool results. This preserves the Chat Completions message contract when trimming.
            start = index - 1
            if self.messages[start].get("role") == "tool":
                while start > 1 and self.messages[start - 1].get("role") == "tool":
                    start -= 1
                if start > 1 and self.messages[start - 1].get("role") == "assistant" and self.messages[start - 1].get("tool_calls"):
                    start -= 1
            elif self.messages[start].get("role") == "assistant" and self.messages[start].get("tool_calls"):
                start = max(1, start)
            group = self.messages[start:index]
            cost = len(json.dumps(group, ensure_ascii=False))
            if size + cost > self.config.max_context_chars:
                break
            kept[0:0] = group
            size += cost
            index = start
        self.messages = [system] + kept

    def run(self, task: str, on_event: Callable[[str], None] | None = None) -> str:
        emit = on_event or (lambda _: None)
        self.messages.append({"role": "user", "content": task})
        for step in range(1, self.config.max_steps + 1):
            self._trim_context()
            emit(f"step {step}/{self.config.max_steps}: thinking")
            assistant = self.client.complete(self.messages, TOOL_SCHEMAS)
            tool_calls = assistant.get("tool_calls") or []
            content = assistant.get("content") or ""
            self.messages.append({"role": "assistant", "content": content, **({"tool_calls": tool_calls} if tool_calls else {})})
            if not tool_calls:
                return content.strip() or "完成。"
            for call in tool_calls:
                function = call.get("function", {})
                name = function.get("name", "")
                raw_args = function.get("arguments", {})
                try:
                    arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must be an object")
                    emit(f"tool {name}: {arguments}")
                    result = self.workspace.execute(name, arguments)
                except Exception as exc:  # tool failures are recoverable model observations
                    result = f"ERROR: {exc}"
                    self.workspace._audit("tool_error", tool=name, error=str(exc))
                self.messages.append({"role": "tool", "tool_call_id": call.get("id", name), "content": result})
        return "达到最大步骤数，任务尚未确认完成。请检查工作区后继续。"


def make_agent(root: str | Path, config: AgentConfig | None = None, approve: Callable[[str], bool] | None = None,
               audit_path: str | Path | None = None) -> CodingAgent:
    cfg = config or AgentConfig()
    return CodingAgent(OpenAICompatibleClient(cfg), Workspace(root, approve=approve, config=cfg, audit_path=audit_path), cfg)
