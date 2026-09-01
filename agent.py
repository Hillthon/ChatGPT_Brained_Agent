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
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


class AgentError(RuntimeError):
    """An expected, user-facing agent error."""


class ModelError(AgentError):
    """The model endpoint could not be called or parsed."""


@dataclass
class AgentConfig:
    model: str = field(default_factory=lambda: os.environ.get("CODING_AGENT_MODEL", "gpt-5.6-luna"))
    base_url: str = field(default_factory=lambda: (
        os.environ.get("CODING_AGENT_BASE_URL")
        or os.environ.get("RIGHTAPI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://rightapi.ai/codex/v1"
    ))
    max_steps: int = 100
    max_context_chars: int = 80_000
    command_timeout: int = 30
    max_output_chars: int = 12_000
    api_mode: str = field(default_factory=lambda: os.environ.get("CODING_AGENT_API_MODE", "auto"))
    request_timeout: int = 120


class OpenAICompatibleClient:
    """OpenAI-compatible client using only the Python standard library.

    Relay services generally expose Chat Completions or Responses under a
    configurable path such as ``/codex/v1``. The agent keeps one internal
    tool-call shape and translates Responses payloads at this boundary.
    """

    def __init__(self, config: AgentConfig, api_key: str | None = None):
        self.config = config
        self.api_key = (
            api_key
            or os.environ.get("CODING_AGENT_API_KEY")
            or os.environ.get("RIGHTAPI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ModelError("set OPENAI_API_KEY, RIGHTAPI_API_KEY, or CODING_AGENT_API_KEY")
        mode = self.config.api_mode.strip().lower()
        if mode in {"completions", "chat_completions"}:
            mode = "chat"
        if mode not in {"auto", "chat", "responses"}:
            raise ModelError(f"unsupported API mode: {self.config.api_mode!r} (use auto, chat, or responses)")
        self.api_mode = mode

    @staticmethod
    def _endpoint(base_url: str, suffix: str) -> str:
        """Build an endpoint without duplicating a suffix supplied by a user."""
        base = base_url.strip().rstrip("/")
        if not base:
            raise ModelError("base URL is empty")
        return base if base.endswith(suffix) else base + suffix

    @staticmethod
    def _response_error(exc: urllib.error.HTTPError) -> str:
        try:
            raw = exc.read().decode("utf-8", errors="replace").strip()
            if raw:
                try:
                    payload = json.loads(raw)
                    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
                        error = payload["error"]
                        return str(error.get("message") or error.get("detail") or raw)
                    return raw
                except json.JSONDecodeError:
                    return raw
        except OSError:
            pass
        return str(exc.reason or exc)

    def _request(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "coding-agent/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.request_timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            error = ModelError(f"model request failed (HTTP {exc.code}): {self._response_error(exc)}")
            error.status_code = exc.code  # type: ignore[attr-defined]
            raise error from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ModelError(f"model request failed: {exc}") from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ModelError(f"model returned invalid JSON: {raw[:500]!r}") from exc
        if not isinstance(payload, dict):
            raise ModelError(f"unexpected model response: {payload!r}")
        return payload

    @staticmethod
    def _normalise_chat_message(message: dict[str, Any]) -> dict[str, Any]:
        normalised = dict(message)
        content = normalised.get("content")
        if isinstance(content, list):
            normalised["content"] = "".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        return normalised

    def _complete_chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        endpoint = self._endpoint(self.config.base_url, "/chat/completions")
        payload = self._request(endpoint, {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        })
        try:
            message = payload["choices"][0]["message"]
            if not isinstance(message, dict):
                raise TypeError("message is not an object")
            return self._normalise_chat_message(message)
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelError(f"unexpected Chat Completions response: {payload!r}") from exc

    @staticmethod
    def _responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for tool in tools:
            function = tool.get("function", {})
            converted.append({
                "type": "function",
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {"type": "object", "properties": {}}),
            })
        return converted

    @staticmethod
    def _responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "assistant" and message.get("tool_calls"):
                for call in message["tool_calls"]:
                    function = call.get("function", {})
                    converted.append({
                        "type": "function_call",
                        "call_id": call.get("id", function.get("name", "call")),
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments", "{}"),
                    })
                if message.get("content"):
                    converted.append({"role": "assistant", "content": message["content"]})
            elif role == "tool":
                converted.append({
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id", ""),
                    "output": message.get("content", ""),
                })
            else:
                converted.append({"role": role, "content": message.get("content", "")})
        return converted

    @staticmethod
    def _responses_message(payload: dict[str, Any]) -> dict[str, Any]:
        output = payload.get("output")
        if not isinstance(output, list):
            output_text = payload.get("output_text")
            if isinstance(output_text, str):
                return {"role": "assistant", "content": output_text}
            raise ModelError(f"unexpected Responses response: {payload!r}")
        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                content = item.get("content", [])
                if isinstance(content, str):
                    content_parts.append(content)
                else:
                    for part in content:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            content_parts.append(part["text"])
            elif item.get("type") == "output_text" and isinstance(item.get("text"), str):
                content_parts.append(item["text"])
            elif item.get("type") == "function_call":
                tool_calls.append({
                    "id": item.get("call_id") or item.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                })
        if not content_parts and isinstance(payload.get("output_text"), str):
            content_parts.append(payload["output_text"])
        message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    def _complete_responses(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        endpoint = self._endpoint(self.config.base_url, "/responses")
        payload = self._request(endpoint, {
            "model": self.config.model,
            "input": self._responses_input(messages),
            "tools": self._responses_tools(tools),
            "store": False,
        })
        return self._responses_message(payload)

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        if self.api_mode == "responses":
            return self._complete_responses(messages, tools)
        if self.api_mode == "chat":
            return self._complete_chat(messages, tools)
        try:
            return self._complete_chat(messages, tools)
        except ModelError as exc:
            # A 404/405 indicates a relay that exposes only Responses at this
            # base path. Do not replay auth, quota, or malformed requests.
            if getattr(exc, "status_code", None) not in {404, 405}:
                raise
            return self._complete_responses(messages, tools)


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
    def __init__(self, client: OpenAICompatibleClient, workspace: Workspace, config: AgentConfig | None = None,
                 messages: list[dict[str, Any]] | None = None,
                 on_history_change: Callable[[list[dict[str, Any]]], None] | None = None):
        self.client = client
        self.workspace = workspace
        self.config = config or client.config
        self.messages = deepcopy(messages) if messages else [{"role": "system", "content": SYSTEM_PROMPT}]
        if self.messages[0].get("role") != "system":
            raise AgentError("session history must start with a system message")
        self.on_history_change = on_history_change

    def _checkpoint(self) -> None:
        if self.on_history_change:
            self.on_history_change(deepcopy(self.messages))

    def _append_message(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        self._checkpoint()

    def _trim_context(self) -> None:
        serialized = json.dumps(self.messages, ensure_ascii=False)
        if len(serialized) <= self.config.max_context_chars:
            return
        # Keep complete user turns so trimming never separates a request from its response or
        # an assistant tool call from its results. The newest turn is kept even if it alone is
        # larger than the target, so the current user request is never silently discarded.
        system = self.messages[0]
        turns: list[list[dict[str, Any]]] = []
        for message in self.messages[1:]:
            if message.get("role") == "user" or not turns:
                turns.append([])
            turns[-1].append(message)
        kept_turns: list[list[dict[str, Any]]] = []
        size = len(json.dumps(system, ensure_ascii=False))
        for turn in reversed(turns):
            cost = len(json.dumps(turn, ensure_ascii=False))
            if kept_turns and size + cost > self.config.max_context_chars:
                break
            kept_turns.insert(0, turn)
            size += cost
        self.messages = [system] + [message for turn in kept_turns for message in turn]
        self._checkpoint()

    def run(self, task: str, on_event: Callable[[str], None] | None = None) -> str:
        emit = on_event or (lambda _: None)
        self._append_message({"role": "user", "content": task})
        for step in range(1, self.config.max_steps + 1):
            self._trim_context()
            emit(f"step {step}/{self.config.max_steps}: thinking")
            assistant = self.client.complete(self.messages, TOOL_SCHEMAS)
            tool_calls = assistant.get("tool_calls") or []
            content = assistant.get("content") or ""
            self._append_message({"role": "assistant", "content": content, **({"tool_calls": tool_calls} if tool_calls else {})})
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
                self._append_message({"role": "tool", "tool_call_id": call.get("id", name), "content": result})
        return "达到最大步骤数，任务尚未确认完成。请检查工作区后继续。"


def make_agent(root: str | Path, config: AgentConfig | None = None, approve: Callable[[str], bool] | None = None,
               audit_path: str | Path | None = None, api_key: str | None = None,
               messages: list[dict[str, Any]] | None = None,
               on_history_change: Callable[[list[dict[str, Any]]], None] | None = None) -> CodingAgent:
    cfg = config or AgentConfig()
    client = OpenAICompatibleClient(cfg, api_key=api_key)
    workspace = Workspace(root, approve=approve, config=cfg, audit_path=audit_path)
    return CodingAgent(client, workspace, cfg, messages=messages, on_history_change=on_history_change)
