"""A small, framework-free coding agent.

The model only proposes tool calls. Files and commands are always handled by
this process, which keeps the important execution logic local and inspectable.
"""

from __future__ import annotations

import hashlib
import json
import base64
import difflib
import io
import os
import re
import shutil
import stat
import subprocess
import time
import mimetypes
import urllib.error
import urllib.request
import zipfile
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator
import xml.etree.ElementTree as ET


AgentEvent = tuple[str, Any]
EventHandler = Callable[[AgentEvent], None]


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
    max_steps: int = 1000
    max_context_chars: int = 80_0000
    command_timeout: int = 30
    max_output_chars: int = 12_0000
    max_image_bytes: int = 2_000_000
    max_image_dimension: int = 2048
    max_image_pages: int = 4
    max_context_tokens: int = 32_000
    max_output_tokens: int = 4_000
    tool_result_max_chars: int = 6_000
    task_summary_max_chars: int = 2_000
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
        self.last_usage: dict[str, Any] = {}
        self.total_usage: dict[str, int] = {}
        self.debug_handler: Callable[[str, Any], None] | None = None

    def _debug(self, kind: str, payload: Any) -> None:
        if self.debug_handler:
            self.debug_handler(kind, payload)

    def _record_usage(self, payload: dict[str, Any]) -> None:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return
        self.last_usage = dict(usage)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                self.total_usage[key] = self.total_usage.get(key, 0) + int(value)

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
        self._debug("model_request", {"endpoint": endpoint, "body": body})
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
        self._record_usage(payload)
        self._debug("model_response", payload)
        return payload

    @staticmethod
    def _stream_request(endpoint: str, body: dict[str, Any], api_key: str, timeout: int):
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream, application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "coding-agent/1.0",
            },
            method="POST",
        )
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            error = ModelError(f"model request failed (HTTP {exc.code}): {OpenAICompatibleClient._response_error(exc)}")
            error.status_code = exc.code  # type: ignore[attr-defined]
            raise error from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ModelError(f"model request failed: {exc}") from exc

    @staticmethod
    def _sse_payloads(response) -> Iterator[dict[str, Any]]:
        """Yield JSON payloads from SSE, while accepting a non-SSE JSON fallback."""
        pending: list[str] = []
        non_sse: list[str] = []

        def consume(data: str) -> dict[str, Any] | None:
            if data.strip() == "[DONE]":
                return None
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ModelError(f"model returned invalid SSE JSON: {data[:500]!r}") from exc
            if not isinstance(payload, dict):
                raise ModelError(f"unexpected streamed model response: {payload!r}")
            return payload

        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n") if isinstance(raw_line, bytes) else str(raw_line).rstrip("\r\n")
            if line.startswith((":", "event:", "id:", "retry:")):
                continue
            if line.startswith("data:"):
                pending.append(line[5:].lstrip())
                continue
            if line == "" and pending:
                data = "\n".join(pending)
                pending.clear()
                payload = consume(data)
                if payload is None:
                    return
                yield payload
            elif not pending and line.strip():
                if line.strip() == "[DONE]":
                    return
                non_sse.append(line.strip())
        if pending:
            payload = consume("\n".join(pending))
            if payload is not None:
                yield payload
        if non_sse:
            payload = consume("\n".join(non_sse))
            if payload is not None:
                yield payload

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
            "max_tokens": self.config.max_output_tokens,
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
        def convert_content(content: Any) -> Any:
            if not isinstance(content, list):
                return content
            converted: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                kind = part.get("type")
                if kind == "text":
                    converted.append({"type": "input_text", "text": part.get("text", "")})
                elif kind == "image_url":
                    image = part.get("image_url")
                    if isinstance(image, dict):
                        item: dict[str, Any] = {"type": "input_image", "image_url": image.get("url", "")}
                        if image.get("detail"):
                            item["detail"] = image["detail"]
                        converted.append(item)
            return converted

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
                converted.append({"role": role, "content": convert_content(message.get("content", ""))})
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
            "max_output_tokens": self.config.max_output_tokens,
        })
        return self._responses_message(payload)

    def _complete_stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_delta: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        endpoint = self._endpoint(self.config.base_url, "/chat/completions")
        body = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": self.config.max_output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        self._debug("model_request", {"endpoint": endpoint, "body": body})
        response = self._stream_request(endpoint, body, self.api_key, self.config.request_timeout)
        content_parts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        try:
            for payload in self._sse_payloads(response):
                self._record_usage(payload)
                self._debug("model_chunk", payload)
                choices = payload.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                if isinstance(choice.get("message"), dict):
                    direct = self._normalise_chat_message(choice["message"])
                    text = direct.get("content") or ""
                    if text:
                        content_parts.append(text)
                        if on_delta:
                            on_delta(text)
                    for index, call in enumerate(direct.get("tool_calls") or []):
                        calls[index] = call
                    continue
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    continue
                text = delta.get("content")
                if isinstance(text, str) and text:
                    content_parts.append(text)
                    if on_delta:
                        on_delta(text)
                for fallback_index, call in enumerate(delta.get("tool_calls") or []):
                    if not isinstance(call, dict):
                        continue
                    index = call.get("index", fallback_index)
                    try:
                        index = int(index)
                    except (TypeError, ValueError):
                        index = fallback_index
                    entry = calls.setdefault(index, {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    })
                    if call.get("id"):
                        entry["id"] = call["id"]
                    function = call.get("function") or {}
                    if isinstance(function, dict):
                        if isinstance(function.get("name"), str):
                            entry["function"]["name"] += function["name"]
                        if isinstance(function.get("arguments"), str):
                            entry["function"]["arguments"] += function["arguments"]
        finally:
            getattr(response, "close", lambda: None)()
        message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
        if calls:
            message["tool_calls"] = [calls[index] for index in sorted(calls)]
        return message

    def _complete_stream_responses(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_delta: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        endpoint = self._endpoint(self.config.base_url, "/responses")
        body = {
            "model": self.config.model,
            "input": self._responses_input(messages),
            "tools": self._responses_tools(tools),
            "store": False,
            "max_output_tokens": self.config.max_output_tokens,
            "stream": True,
        }
        self._debug("model_request", {"endpoint": endpoint, "body": body})
        response = self._stream_request(endpoint, body, self.api_key, self.config.request_timeout)
        content_parts: list[str] = []
        calls: dict[str, dict[str, Any]] = {}
        final_response: dict[str, Any] | None = None
        try:
            for payload in self._sse_payloads(response):
                self._record_usage(payload)
                self._debug("model_chunk", payload)
                event_type = payload.get("type")
                if event_type == "response.output_text.delta":
                    delta = payload.get("delta")
                    if isinstance(delta, str):
                        content_parts.append(delta)
                        if on_delta:
                            on_delta(delta)
                elif event_type == "response.output_item.added":
                    item = payload.get("item")
                    if isinstance(item, dict) and item.get("type") == "function_call":
                        key = str(item.get("call_id") or item.get("id") or len(calls))
                        calls[key] = {
                            "id": item.get("call_id") or item.get("id", key),
                            "type": "function",
                            "function": {
                                "name": item.get("name") or "",
                                "arguments": item.get("arguments") or "",
                            },
                        }
                elif event_type == "response.function_call_arguments.delta":
                    key = str(payload.get("call_id") or payload.get("item_id") or len(calls))
                    if key not in calls and len(calls) == 1:
                        key = next(iter(calls))
                    entry = calls.setdefault(key, {
                        "id": payload.get("call_id", key),
                        "type": "function",
                        "function": {"name": payload.get("name") or "", "arguments": ""},
                    })
                    delta = payload.get("delta") or payload.get("arguments")
                    if isinstance(delta, str):
                        entry["function"]["arguments"] += delta
                elif event_type == "response.completed" and isinstance(payload.get("response"), dict):
                    final_response = payload["response"]
                elif isinstance(payload.get("output"), list):
                    # Some OpenAI-compatible relays ignore stream=true and return
                    # one ordinary Responses payload instead of SSE events.
                    final_response = payload
                elif isinstance(payload.get("output_text"), str) and not content_parts:
                    content_parts.append(payload["output_text"])
                    if on_delta:
                        on_delta(payload["output_text"])
        finally:
            getattr(response, "close", lambda: None)()
        if final_response:
            final_message = self._responses_message(final_response)
            if not content_parts and final_message.get("content"):
                content = final_message["content"]
                content_parts.append(content)
                if on_delta:
                    on_delta(content)
            if not calls:
                for call in final_message.get("tool_calls") or []:
                    calls[str(call.get("id", len(calls)))] = call
        message = {"role": "assistant", "content": "".join(content_parts)}
        if calls:
            message["tool_calls"] = list(calls.values())
        return message

    def complete_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_delta: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Complete a turn while forwarding assistant text deltas as they arrive."""
        if self.api_mode == "responses":
            return self._complete_stream_responses(messages, tools, on_delta)
        if self.api_mode == "chat":
            return self._complete_stream_chat(messages, tools, on_delta)
        try:
            return self._complete_stream_chat(messages, tools, on_delta)
        except ModelError as exc:
            if getattr(exc, "status_code", None) not in {404, 405}:
                raise
            return self._complete_stream_responses(messages, tools, on_delta)

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
    _json_schema("read_image", "Read a workspace image so a vision-capable model can inspect it.", path={"type": "string"}, detail={"type": "string", "enum": ["low", "auto", "high"], "default": "auto"}),
    _json_schema("read_pdf", "Extract text from a workspace PDF and optionally attach rendered pages for visual analysis.", path={"type": "string"}, start_page={"type": "integer", "minimum": 1, "default": 1}, end_page={"type": ["integer", "null"], "minimum": 1, "default": None}, include_images={"type": "boolean", "default": False}),
    _json_schema("read_docx", "Extract paragraphs and tables from a workspace DOCX and optionally attach embedded images.", path={"type": "string"}, start_paragraph={"type": "integer", "minimum": 1, "default": 1}, end_paragraph={"type": ["integer", "null"], "minimum": 1, "default": None}, include_images={"type": "boolean", "default": False}),
    _json_schema("search_files", "Search text in UTF-8 files below a directory.", query={"type": "string"}, path={"type": "string", "default": "."}),
    _json_schema("write_file", "Create or replace a UTF-8 file. Use only when the user requested a change.", path={"type": "string"}, content={"type": "string"}),
    _json_schema("apply_patch", "Apply a single-file unified diff to one workspace-relative file. Prefer complete ---/+++ and @@ -start,count +start,count @@ headers; Codex-style *** Begin Patch wrappers are also accepted.", path={"type": "string"}, patch={"type": "string"}),
    _json_schema("run_command", "Run a project command in the workspace after confirmation.", command={"type": "string"}, timeout={"type": "integer", "minimum": 1, "maximum": 120, "default": 30}),
    _json_schema("verify_task", "Run the project's tests, checks, or build command. A zero exit code is required before finishing a task that changed files.", command={"type": "string"}, timeout={"type": "integer", "minimum": 1, "maximum": 120, "default": 30}),
    _json_schema("finish_task", "Submit a task completion summary after verification. Do not use this before verify_task succeeds when files were changed.", summary={"type": "string"}),
]


class Workspace:
    """Local tools with path and command guardrails."""

    BLOCKED_COMMANDS = re.compile(
        r"(?:^|[\s;&|])(?:rm|del|erase|rmdir|format|shutdown|reboot|diskpart|remove-item)(?:\s|$)|"
        r"(?:^|[\s;&|])git\s+reset\s+--hard(?:\s|$)", re.I
    )

    def __init__(self, root: str | Path, approve: Callable[[str], bool] | None = None,
                 config: AgentConfig | None = None, audit_path: str | Path | None = None,
                 undo_manager: Any | None = None):
        self.root = Path(root).resolve()
        self.approve = approve or (lambda _: False)
        self.config = config or AgentConfig()
        self.audit_path = self.resolve(audit_path) if audit_path else None
        self.last_full_result: str | None = None
        self.last_images: list[dict[str, Any]] = []
        self.undo_manager = undo_manager

    def begin_undo_task(self, prompt: str) -> str | None:
        return self.undo_manager.begin_task(prompt) if self.undo_manager else None

    def finish_undo_task(self, task_id: str | None, status: str) -> None:
        if self.undo_manager:
            self.undo_manager.finish_task(task_id, status)

    def undo_last(self) -> Any:
        if not self.undo_manager:
            raise AgentError("undo is not configured for this agent")
        action = self.undo_manager.undo_last()
        self._audit("undo", action_id=action.id, task_id=action.task_id, path=action.path, tool=action.tool)
        return action

    def rollback_latest_task(self) -> tuple[Any, list[Any]]:
        if not self.undo_manager:
            raise AgentError("undo is not configured for this agent")
        task, actions = self.undo_manager.rollback_latest_task()
        self._audit("rollback_task", task_id=task.id, paths=[action.path for action in actions])
        return task, actions

    def undo_checkpoints(self) -> list[Any]:
        return self.undo_manager.active_actions() if self.undo_manager else []

    @staticmethod
    def _write_text_atomic(file: Path, content: str) -> None:
        mode = stat.S_IMODE(file.stat().st_mode) if file.exists() else None
        temporary = file.with_name(f".{file.name}.{os.urandom(8).hex()}.tmp")
        try:
            temporary.write_text(content, encoding="utf-8")
            if mode is not None:
                try:
                    temporary.chmod(mode)
                except OSError:
                    pass
            temporary.replace(file)
            if mode is not None:
                try:
                    file.chmod(mode)
                except OSError:
                    pass
        finally:
            temporary.unlink(missing_ok=True)

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
        result = "\n".join(entries) or "(empty)"
        self.last_full_result = result
        return result

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
        result = "\n".join(shown)
        self.last_full_result = result
        return result[: self.config.max_output_chars]

    @staticmethod
    def _mime_type(path: str, fallback: str = "application/octet-stream") -> str:
        return mimetypes.guess_type(path)[0] or fallback

    def _image_attachment(self, raw: bytes, name: str, detail: str = "auto") -> dict[str, Any]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise AgentError("image support requires Pillow; install it with: python -m pip install pillow") from exc
        try:
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                width, height = image.size
                image_format = (image.format or "PNG").upper()
                if len(raw) > self.config.max_image_bytes or max(width, height) > self.config.max_image_dimension:
                    converted = image.convert("RGB") if image.mode not in {"RGB", "L"} else image.copy()
                    converted.thumbnail((self.config.max_image_dimension, self.config.max_image_dimension))
                    output = io.BytesIO()
                    converted.save(output, format="JPEG", quality=85, optimize=True)
                    raw = output.getvalue()
                    image_format = "JPEG"
                mime = "image/jpeg" if image_format == "JPEG" else self._mime_type(name, "image/png")
        except Exception as exc:
            raise AgentError(f"unsupported or invalid image: {name}") from exc
        if len(raw) > self.config.max_image_bytes:
            raise AgentError(f"image is too large after compression: {name}")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}",
                "detail": detail,
            },
            "name": name,
        }

    def read_image(self, path: str, detail: str = "auto") -> str:
        """Load one workspace image and make it available to a vision-capable model."""
        file = self.resolve(path)
        if not file.is_file():
            raise AgentError(f"not a file: {path}")
        if file.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}:
            raise AgentError("read_image supports PNG, JPEG, WEBP, GIF, BMP, and TIFF files")
        if detail not in {"low", "auto", "high"}:
            raise AgentError("detail must be low, auto, or high")
        try:
            raw = file.read_bytes()
        except OSError as exc:
            raise AgentError(f"could not read image: {exc}") from exc
        attachment = self._image_attachment(raw, file.name, detail)
        self.last_images = [attachment]
        result = f"image_loaded: {file.relative_to(self.root)} ({attachment['image_url']['url'].split(';', 1)[0]})"
        self.last_full_result = result
        return result

    def _render_pdf_images(self, file: Path, start_page: int, end_page: int, detail: str = "auto") -> list[dict[str, Any]]:
        pages = list(range(start_page, min(end_page, start_page + self.config.max_image_pages - 1) + 1))
        attachments: list[dict[str, Any]] = []
        try:
            import fitz  # PyMuPDF
        except ImportError:
            fitz = None
        if fitz is not None:
            try:
                document = fitz.open(str(file))
                try:
                    for page in pages:
                        pixmap = document.load_page(page - 1).get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                        attachments.append(self._image_attachment(pixmap.tobytes("png"), f"{file.name} page {page}.png", detail))
                finally:
                    document.close()
                return attachments
            except Exception as exc:
                raise AgentError(f"could not render PDF visually: {exc}") from exc
        renderer = shutil.which("pdftoppm")
        if not renderer:
            raise AgentError("visual PDF reading requires PyMuPDF or the pdftoppm command")
        import tempfile
        with tempfile.TemporaryDirectory(prefix="coding-agent-pdf-") as directory:
            output_dir = Path(directory)
            for page in pages:
                prefix = output_dir / f"page-{page}"
                completed = subprocess.run(
                    [renderer, "-f", str(page), "-l", str(page), "-png", "-singlefile", str(file), str(prefix)],
                    capture_output=True, text=True, errors="replace", timeout=self.config.command_timeout,
                )
                rendered = prefix.with_suffix(".png")
                if completed.returncode != 0 or not rendered.is_file():
                    detail_text = (completed.stderr or completed.stdout or "renderer failed").strip()
                    raise AgentError(f"could not render PDF page {page}: {detail_text}")
                attachments.append(self._image_attachment(rendered.read_bytes(), f"{file.name} page {page}.png", detail))
        return attachments

    def read_pdf(self, path: str, start_page: int = 1, end_page: int | None = None, include_images: bool = False) -> str:
        """Extract text from a PDF without passing binary document data to the model."""
        file = self.resolve(path)
        if file.suffix.lower() != ".pdf":
            raise AgentError("read_pdf only supports .pdf files")
        if not file.is_file():
            raise AgentError(f"not a file: {path}")
        if start_page < 1 or (end_page is not None and end_page < 1):
            raise AgentError("page numbers are 1-based")
        if end_page is not None and start_page > end_page:
            raise AgentError("start_page must not be greater than end_page")
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise AgentError("PDF support requires pypdf; install it with: python -m pip install pypdf") from exc
        try:
            with file.open("rb") as stream:
                reader = PdfReader(stream)
                if reader.is_encrypted:
                    try:
                        if not reader.decrypt(""):
                            raise AgentError("encrypted PDF cannot be opened without a password")
                    except AgentError:
                        raise
                    except Exception as exc:
                        raise AgentError("encrypted PDF cannot be opened without a password") from exc
                page_count = len(reader.pages)
                last_page = end_page or page_count
                if start_page > page_count or last_page > page_count:
                    raise AgentError(f"page range {start_page}-{last_page} is outside PDF ({page_count} pages)")
                sections: list[str] = []
                for number in range(start_page, last_page + 1):
                    text = reader.pages[number - 1].extract_text() or ""
                    sections.append(f"[Page {number}]\n{text.strip()}".rstrip())
        except AgentError:
            raise
        except Exception as exc:
            # pypdf raises several parser-specific exception classes across versions.
            raise AgentError(f"could not read PDF: {exc}") from exc
        result = "\n\n".join(sections)
        has_text = any(not section.rstrip().endswith(f"[Page {number}]") for number, section in zip(range(start_page, last_page + 1), sections))
        self.last_images = []
        if include_images or not has_text:
            try:
                self.last_images = self._render_pdf_images(file, start_page, last_page)
            except AgentError as exc:
                if not has_text:
                    raise AgentError("PDF contains no extractable text and visual rendering is unavailable; install PyMuPDF or pdftoppm")
                if include_images:
                    raise exc
        if not result and not self.last_images:
            raise AgentError("PDF contains no extractable text; OCR is not available")
        if self.last_images:
            result = (result + "\n\n" if result else "") + f"[Visual pages attached: {len(self.last_images)}]"
        self.last_full_result = result
        return result[: self.config.max_output_chars]

    @staticmethod
    def _docx_local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    @classmethod
    def _docx_paragraph_text(cls, element: ET.Element) -> str:
        parts: list[str] = []
        for node in element.iter():
            kind = cls._docx_local_name(node.tag)
            if kind == "t":
                parts.append(node.text or "")
            elif kind == "br":
                parts.append("\n")
        return "".join(parts)

    def read_docx(self, path: str, start_paragraph: int = 1, end_paragraph: int | None = None, include_images: bool = False) -> str:
        """Extract document-level paragraphs and tables from a DOCX OOXML package."""
        file = self.resolve(path)
        if file.suffix.lower() != ".docx":
            raise AgentError("read_docx only supports .docx files; convert legacy .doc files to DOCX or PDF")
        if not file.is_file():
            raise AgentError(f"not a file: {path}")
        if start_paragraph < 1 or (end_paragraph is not None and end_paragraph < 1):
            raise AgentError("paragraph numbers are 1-based")
        if end_paragraph is not None and start_paragraph > end_paragraph:
            raise AgentError("start_paragraph must not be greater than end_paragraph")
        try:
            with zipfile.ZipFile(file) as package:
                try:
                    document_xml = package.read("word/document.xml")
                except KeyError as exc:
                    raise AgentError("DOCX is missing word/document.xml") from exc
                media = [
                    (name, package.read(name))
                    for name in package.namelist()
                    if include_images and name.startswith("word/media/") and not name.endswith("/")
                ]
            root = ET.fromstring(document_xml)
        except AgentError:
            raise
        except (OSError, zipfile.BadZipFile, ET.ParseError) as exc:
            raise AgentError(f"could not read DOCX: {exc}") from exc

        body = next((element for element in root.iter() if self._docx_local_name(element.tag) == "body"), None)
        if body is None:
            raise AgentError("DOCX is missing a document body")
        sections: list[str] = []
        paragraph_number = 0
        table_number = 0
        for child in list(body):
            kind = self._docx_local_name(child.tag)
            if kind == "p":
                paragraph_number += 1
                if start_paragraph <= paragraph_number and (end_paragraph is None or paragraph_number <= end_paragraph):
                    text = self._docx_paragraph_text(child)
                    sections.append(f"[Paragraph {paragraph_number}]\n{text}".rstrip())
            elif kind == "tbl":
                table_number += 1
                rows: list[str] = []
                for row in child:
                    if self._docx_local_name(row.tag) != "tr":
                        continue
                    cells: list[str] = []
                    for cell in row:
                        if self._docx_local_name(cell.tag) == "tc":
                            paragraphs = [
                                self._docx_paragraph_text(paragraph)
                                for paragraph in cell.iter()
                                if self._docx_local_name(paragraph.tag) == "p"
                            ]
                            cell_text = "\n".join(paragraph for paragraph in paragraphs if paragraph)
                            cells.append(cell_text)
                    if cells:
                        rows.append(" | ".join(cells))
                if rows:
                    sections.append(f"[Table {table_number}]\n" + "\n".join(rows))
        self.last_images = []
        if include_images:
            for name, raw in media[: self.config.max_image_pages]:
                try:
                    self.last_images.append(self._image_attachment(raw, Path(name).name))
                except AgentError:
                    continue
        if not sections and not self.last_images:
            raise AgentError("DOCX contains no extractable paragraphs, tables, or supported images")
        result = "\n\n".join(sections)
        if self.last_images:
            result += ("\n\n" if result else "") + f"[Embedded images attached: {len(self.last_images)}]"
        self.last_full_result = result
        return result[: self.config.max_output_chars]

    def search_files(self, query: str, path: str = ".") -> str:
        base = self.resolve(path)
        if not base.is_dir():
            raise AgentError(f"not a directory: {path}")
        results: list[str] = []
        limit_reached = False
        for file in base.rglob("*"):
            if not file.is_file() or any(part in {".git", ".venv", "node_modules", "__pycache__"} for part in file.parts):
                continue
            try:
                for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
                    if query.lower() in line.lower():
                        results.append(f"{file.relative_to(self.root)}:{number}: {line.strip()}")
                        if len(results) >= 100:
                            limit_reached = True
                            break
            except (UnicodeDecodeError, OSError):
                continue
            if limit_reached:
                break
        result = "\n".join(results) or "(no matches)"
        if limit_reached:
            result += "\n(results truncated)"
        self.last_full_result = result
        return result

    def preview(self, name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Build a read-only preview for an edit before the approval callback runs."""
        if name == "apply_patch":
            patch = arguments.get("patch")
            if not isinstance(patch, str):
                return None
            if patch.lstrip().startswith("```"):
                fenced = patch.lstrip().splitlines(keepends=True)
                if len(fenced) >= 3 and fenced[-1].strip().startswith("```"):
                    patch = "".join(fenced[1:-1])
            return {"kind": "diff", "path": arguments.get("path", ""), "diff": patch}
        if name != "write_file" or not isinstance(arguments.get("content"), str):
            return None
        relative = arguments.get("path", "")
        file = self.resolve(relative)
        old: list[str] = []
        if file.exists():
            if not file.is_file():
                raise AgentError(f"not a file: {relative}")
            try:
                old = file.read_text(encoding="utf-8").splitlines(keepends=True)
            except UnicodeDecodeError as exc:
                raise AgentError("only UTF-8 text files are supported") from exc
        new = arguments["content"].splitlines(keepends=True)
        diff = "".join(difflib.unified_diff(
            old,
            new,
            fromfile=f"a/{relative}",
            tofile=f"b/{relative}",
            lineterm="\n",
        ))
        return {"kind": "diff", "path": relative, "diff": diff or "(no textual changes)"}

    def write_file(self, path: str, content: str) -> str:
        file = self.resolve(path)
        prompt = f"write {file.relative_to(self.root)} ({len(content)} chars)"
        if not self.approve(prompt):
            self._audit("denied", action=prompt)
            result = "DENIED: user did not approve file write"
            self.last_full_result = result
            return result
        pending = self.undo_manager.capture(file, "write_file") if self.undo_manager else None
        file.parent.mkdir(parents=True, exist_ok=True)
        self._write_text_atomic(file, content)
        action = self.undo_manager.commit(pending) if self.undo_manager and pending else None
        self._audit("write_file", path=str(file.relative_to(self.root)), chars=len(content))
        result = f"wrote {file.relative_to(self.root)} ({len(content)} chars)"
        if action:
            result += f" [undo {action.id[:8]}]"
        self.last_full_result = result
        return result

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
            result = "DENIED: user did not approve patch"
            self.last_full_result = result
            return result
        pending = self.undo_manager.capture(file, "apply_patch") if self.undo_manager else None
        self._write_text_atomic(file, "".join(new))
        action = self.undo_manager.commit(pending) if self.undo_manager and pending else None
        self._audit("apply_patch", path=str(file.relative_to(self.root)))
        result = f"patched {file.relative_to(self.root)}"
        if action:
            result += f" [undo {action.id[:8]}]"
        self.last_full_result = result
        return result

    @staticmethod
    def _apply_unified_diff(old: list[str], patch: str) -> list[str]:
        """Apply a single-file diff while checking every context line.

        Models sometimes emit the Codex ``*** Begin Patch`` wrapper or an
        abbreviated ``@@`` header. Those forms are normalized below, but the
        actual context/removal lines are still matched against the current file.
        """
        if patch.lstrip().startswith("```"):
            fenced = patch.lstrip().splitlines(keepends=True)
            if len(fenced) < 3 or not fenced[-1].strip().startswith("```"):
                raise AgentError("invalid patch: unclosed code fence")
            patch = "".join(fenced[1:-1])
        lines = patch.splitlines(keepends=True)
        if any(line.strip() == "*** Begin Patch" for line in lines):
            update_lines: list[str] = []
            in_update = False
            update_count = 0
            for line in lines:
                stripped = line.strip()
                if stripped == "*** Begin Patch":
                    continue
                if stripped.startswith("*** Update File:"):
                    update_count += 1
                    if update_count > 1:
                        raise AgentError("apply_patch accepts one file per call")
                    in_update = True
                    continue
                if stripped.startswith("*** End Patch"):
                    break
                if stripped.startswith(("*** Add File:", "*** Delete File:", "*** Move to:")):
                    raise AgentError("apply_patch only supports updating an existing file")
                if in_update:
                    update_lines.append(line)
            if not update_lines:
                raise AgentError("invalid patch: missing *** Update File section")
            lines = update_lines
        hunk_indexes = [i for i, line in enumerate(lines) if line.startswith("@@")]
        if not hunk_indexes:
            raise AgentError("invalid patch: missing @@ hunk header")
        if any(not re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[i]) for i in hunk_indexes):
            return Workspace._apply_flexible_diff(old, lines, hunk_indexes)
        try:
            return Workspace._apply_numbered_diff(old, lines, hunk_indexes)
        except AgentError as exc:
            # A model may provide correct context with stale line numbers after
            # an earlier edit. Relocate only when the exact context still matches.
            if ("patch context does not match" in str(exc)
                    or "patch removal does not match" in str(exc)
                    or "patch context is outside" in str(exc)):
                return Workspace._apply_flexible_diff(old, lines, hunk_indexes)
            raise

    @staticmethod
    def _apply_numbered_diff(old: list[str], lines: list[str], hunk_indexes: list[int]) -> list[str]:
        """Apply a fully numbered unified diff at its declared line offsets."""
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
                    if cursor >= len(old) or not Workspace._same_line(old[cursor], content):
                        raise AgentError("patch context does not match file")
                    result.append(old[cursor])
                    cursor += 1
                elif marker == "-":
                    if cursor >= len(old) or not Workspace._same_line(old[cursor], content):
                        raise AgentError("patch removal does not match file")
                    cursor += 1
                elif marker == "+":
                    result.append(content)
                else:
                    raise AgentError(f"invalid patch line: {line.strip()}")
            old_cursor = cursor
        result.extend(old[old_cursor:])
        return result

    @staticmethod
    def _same_line(left: str, right: str) -> bool:
        """Compare patch text without treating LF/CRLF as different content."""
        return left.replace("\r\n", "\n") == right.replace("\r\n", "\n")

    @staticmethod
    def _apply_flexible_diff(old: list[str], lines: list[str], hunk_indexes: list[int]) -> list[str]:
        """Apply a diff with abbreviated hunk headers using context search."""
        result: list[str] = []
        old_cursor = 0
        for index, hunk_start in enumerate(hunk_indexes):
            header = lines[hunk_start].strip()
            if not re.match(r"^@@(?:\s.*)?$", header):
                raise AgentError(f"invalid patch hunk header: {header}")
            numbered_header = re.match(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", header)
            preferred_start = int(numbered_header.group(1)) - 1 if numbered_header else None
            body_end = hunk_indexes[index + 1] if index + 1 < len(hunk_indexes) else len(lines)
            operations: list[tuple[str, str]] = []
            for line in lines[hunk_start + 1:body_end]:
                if line.startswith("\\ No newline"):
                    continue
                if line.startswith(("*** ", "--- ", "+++ ")):
                    continue
                if line.startswith(("+", "-", " ")):
                    operations.append((line[0], line[1:]))
                else:
                    # Codex-style abbreviated hunks may omit the context marker.
                    operations.append((" ", line))
            expected = [content for marker, content in operations if marker in {" ", "-"}]
            source_start = old_cursor
            if expected:
                matches: list[int] = []
                for candidate in range(old_cursor, len(old) - len(expected) + 1):
                    if all(Workspace._same_line(old[candidate + offset], content)
                           for offset, content in enumerate(expected)):
                        matches.append(candidate)
                if not matches:
                    raise AgentError("patch context does not match file")
                source_start = min(
                    matches,
                    key=lambda candidate: abs(candidate - preferred_start)
                    if preferred_start is not None else candidate,
                )
            result.extend(old[old_cursor:source_start])
            cursor = source_start
            for marker, content in operations:
                if marker == " ":
                    if cursor >= len(old) or not Workspace._same_line(old[cursor], content):
                        raise AgentError("patch context does not match file")
                    result.append(old[cursor])
                    cursor += 1
                elif marker == "-":
                    if cursor >= len(old) or not Workspace._same_line(old[cursor], content):
                        raise AgentError("patch removal does not match file")
                    cursor += 1
                elif marker == "+":
                    result.append(content)
                else:
                    raise AgentError(f"invalid patch line: {content.strip()}")
            old_cursor = cursor
        result.extend(old[old_cursor:])
        return result

    def run_command(self, command: str, timeout: int | None = None) -> str:
        command = command.strip()
        if not command:
            raise AgentError("command is empty")
        if self.BLOCKED_COMMANDS.search(command):
            self._audit("blocked", action=command, reason="destructive command")
            result = "DENIED: destructive command is blocked"
            self.last_full_result = result
            return result
        # Keep shell use explicit and visible; approval is required for every invocation.
        if not self.approve(f"run command (not covered by file rollback): {command}"):
            self._audit("denied", action=command)
            result = "DENIED: user did not approve command"
            self.last_full_result = result
            return result
        try:
            result = subprocess.run(command, cwd=self.root, shell=True, capture_output=True, text=True, errors="replace",
                                    timeout=min(timeout or self.config.command_timeout, 120))
        except subprocess.TimeoutExpired as exc:
            result = f"TIMEOUT after {exc.timeout}s"
            self.last_full_result = result
            return result
        output = (result.stdout + ("\n" + result.stderr if result.stderr else "")).strip()
        full_result = f"exit_code={result.returncode}\n{output}" if output else f"exit_code={result.returncode}"
        self.last_full_result = full_result
        bounded_output = output[-self.config.max_output_chars:]
        self._audit("run_command", command=command, exit_code=result.returncode)
        return f"exit_code={result.returncode}\n{bounded_output}" if bounded_output else f"exit_code={result.returncode}"

    def verify_task(self, command: str, timeout: int | None = None) -> str:
        """Run an explicit validation command and label its result for the agent loop."""
        result = self.run_command(command, timeout)
        if result.startswith("exit_code=0"):
            return f"verification_passed\n{result}"
        return f"verification_failed\n{result}"

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        self.last_full_result = None
        self.last_images = []
        methods = {"list_files": self.list_files, "read_file": self.read_file, "read_pdf": self.read_pdf,
                   "read_image": self.read_image, "read_docx": self.read_docx, "search_files": self.search_files,
                   "write_file": self.write_file, "apply_patch": self.apply_patch, "run_command": self.run_command,
                   "verify_task": self.verify_task}
        if name not in methods:
            raise AgentError(f"unknown tool: {name}")
        return methods[name](**arguments)


SYSTEM_PROMPT = """You are a careful coding agent working in a local workspace.
Inspect before editing. Use tools for all file reads and changes; do not pretend a change happened.
For apply_patch, prefer a complete single-file unified diff with ---/+++
headers and @@ -start,count +start,count @@ hunks. The tool also accepts
Codex-style *** Begin Patch wrappers. If a patch is rejected, reread the file,
regenerate the patch from that exact content, or use write_file for a small file.
Use read_pdf for PDF text extraction and read_docx for Word DOCX paragraphs and tables. Set include_images=true when visual content matters; use read_image for standalone images. The local agent attaches images as multimodal input, so the configured model must support vision. Scanned PDFs can be rendered for visual inspection, while legacy .doc files must be converted first.
Explain a concise plan, make the smallest useful edits, and run relevant tests when approved.
After changing files, call verify_task with the narrowest relevant test, lint, or build command.
Only after verification succeeds call finish_task with a concise summary of changes and validation.
Do not claim completion in ordinary text after a file change; the local agent requires finish_task.
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
        self.context_messages: list[dict[str, Any]] = deepcopy(self.messages)

    def _checkpoint(self) -> None:
        if self.on_history_change:
            self.on_history_change(deepcopy(self.messages))

    def _append_message(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        self._checkpoint()

    def record_control_event(self, content: str) -> None:
        """Keep local undo operations visible to subsequent model turns."""
        self._append_message({"role": "system", "content": f"Local workspace event: {content}"})

    @staticmethod
    def _estimate_tokens(value: str) -> int:
        """Conservative standard-library token estimate for mixed text and code."""
        return max(1, (len(value) + 2) // 3)

    def _compact_tool_result(self, content: str, seen: dict[str, str]) -> str:
        digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
        if digest in seen:
            return f"[duplicate tool result omitted; same as {seen[digest]}]"
        seen[digest] = digest[:8]
        limit = max(200, self.config.tool_result_max_chars)
        if len(content) <= limit:
            return content
        head = max(80, limit // 3)
        tail = max(80, limit - head - 80)
        omitted = len(content) - head - tail
        return f"{content[:head]}\n...[{omitted} chars omitted]...\n{content[-tail:]}"

    def _compact_messages(self) -> list[dict[str, Any]]:
        """Copy history for the model, bounding large and repeated tool observations."""
        seen: dict[str, str] = {}
        compacted: list[dict[str, Any]] = []
        for message in self.messages:
            copy = deepcopy(message)
            if copy.get("role") == "tool" and isinstance(copy.get("content"), str):
                copy["content"] = self._compact_tool_result(copy["content"], seen)
            compacted.append(copy)
        return compacted

    def _summarize_turns(self, turns: list[list[dict[str, Any]]]) -> str:
        """Extract a small deterministic memory from turns removed from active context."""
        if not turns:
            return ""
        tasks: list[str] = []
        files: list[str] = []
        verifications: list[str] = []
        errors: list[str] = []
        completions: list[str] = []

        def add_unique(values: list[str], value: str, limit: int = 8) -> None:
            value = " ".join(value.split()).strip()
            if value and value not in values and len(values) < limit:
                values.append(value)

        for turn in turns:
            for message in turn:
                role = message.get("role")
                content = message.get("content")
                if role == "user" and isinstance(content, str):
                    add_unique(tasks, content[:240], limit=6)
                if role == "assistant":
                    for call in message.get("tool_calls") or []:
                        if not isinstance(call, dict):
                            continue
                        function = call.get("function") or {}
                        if not isinstance(function, dict):
                            continue
                        name = function.get("name", "")
                        raw_args = function.get("arguments", {})
                        args: dict[str, Any] = {}
                        if isinstance(raw_args, str):
                            try:
                                decoded = json.loads(raw_args)
                                if isinstance(decoded, dict):
                                    args = decoded
                            except json.JSONDecodeError:
                                pass
                        elif isinstance(raw_args, dict):
                            args = raw_args
                        path = args.get("path")
                        if name in {"write_file", "apply_patch"} and isinstance(path, str):
                            add_unique(files, path)
                if role == "tool" and isinstance(content, str):
                    if content.startswith("verification_passed"):
                        add_unique(verifications, "passed")
                    elif content.startswith("verification_failed"):
                        add_unique(verifications, "failed")
                    if content.startswith("TASK_COMPLETED:"):
                        add_unique(completions, content.removeprefix("TASK_COMPLETED:")[:240])
                    if content.startswith(("ERROR:", "verification_failed", "DENIED", "TIMEOUT")):
                        add_unique(errors, content.split("\n", 1)[0][:240])

        lines = ["<deterministic_task_summary>", "Earlier task context was compacted locally. Preserve these facts:"]
        if tasks:
            lines.append("Tasks: " + " | ".join(tasks))
        if files:
            lines.append("Files touched: " + ", ".join(files))
        if verifications:
            lines.append("Verification: " + ", ".join(verifications))
        if completions:
            lines.append("Completion summaries: " + " | ".join(completions))
        if errors:
            lines.append("Errors or warnings: " + " | ".join(errors))
        lines.append("</deterministic_task_summary>")
        summary = "\n".join(lines)
        limit = max(400, self.config.task_summary_max_chars)
        return summary if len(summary) <= limit else summary[:limit - 35] + "\n...[summary truncated]\n</deterministic_task_summary>"

    def _trim_context(self) -> list[dict[str, Any]]:
        """Build a bounded model context without deleting the persisted full history."""
        compacted = self._compact_messages()
        serialized = json.dumps(compacted, ensure_ascii=False)
        input_token_budget = max(1, self.config.max_context_tokens - self.config.max_output_tokens)
        if (
            len(serialized) <= self.config.max_context_chars
            and self._estimate_tokens(serialized) <= input_token_budget
        ):
            self.context_messages = compacted
            return deepcopy(compacted)
        # Keep complete user turns so trimming never separates a request from its response or
        # an assistant tool call from its results. The newest turn is kept even if it alone is
        # larger than the target, so the current user request is never silently discarded.
        system = compacted[0]
        turns: list[list[dict[str, Any]]] = []
        for message in compacted[1:]:
            if message.get("role") == "user" or not turns:
                turns.append([])
            turns[-1].append(message)
        kept_turns: list[list[dict[str, Any]]] = []
        size = len(json.dumps(system, ensure_ascii=False))
        token_size = self._estimate_tokens(json.dumps(system, ensure_ascii=False))
        for turn in reversed(turns):
            cost = len(json.dumps(turn, ensure_ascii=False))
            token_cost = self._estimate_tokens(json.dumps(turn, ensure_ascii=False))
            over_chars = size + cost > self.config.max_context_chars
            over_tokens = token_size + token_cost > input_token_budget
            if kept_turns and (over_chars or over_tokens):
                break
            kept_turns.insert(0, turn)
            size += cost
            token_size += token_cost
        dropped_turns = turns[:max(0, len(turns) - len(kept_turns))]
        summary = self._summarize_turns(dropped_turns)
        prefix = [system]
        if summary:
            prefix.append({"role": "system", "content": summary})
        # The summary is smaller than the dropped turns, but it still counts against
        # both budgets. Drop additional old turns if needed while retaining the newest.
        while len(kept_turns) > 1:
            candidate = prefix + [message for turn in kept_turns for message in turn]
            candidate_json = json.dumps(candidate, ensure_ascii=False)
            if len(candidate_json) <= self.config.max_context_chars and self._estimate_tokens(candidate_json) <= input_token_budget:
                break
            dropped_turns.append(kept_turns.pop(0))
            summary = self._summarize_turns(dropped_turns)
            prefix = [system] + ([{"role": "system", "content": summary}] if summary else [])
        self.context_messages = prefix + [message for turn in kept_turns for message in turn]
        return deepcopy(self.context_messages)

    @staticmethod
    def _tool_ok(name: str, result: str) -> bool:
        if result.startswith(("ERROR:", "DENIED", "TIMEOUT")):
            return False
        if name == "verify_task":
            return result.startswith("verification_passed\n")
        if name == "run_command":
            match = re.match(r"exit_code=(-?\d+)", result)
            if match and int(match.group(1)) != 0:
                return False
        return True

    def _run_end(self, emit: EventHandler, steps: int, started_at: float, result: str,
                 usage_before: dict[str, int]) -> None:
        usage = getattr(self.client, "total_usage", {})
        usage_delta = {
            key: int(value) - usage_before.get(key, 0)
            for key, value in usage.items()
            if isinstance(value, (int, float)) and int(value) - usage_before.get(key, 0) > 0
        } if isinstance(usage, dict) else {}
        emit(("run_end", {
            "steps": steps,
            "elapsed": time.perf_counter() - started_at,
            "result": result,
            "completion_status": getattr(self, "_last_task_status", "incomplete"),
            "completion_confirmed": getattr(self, "_last_task_status", "incomplete") in {"verified", "answered", "completed"},
            "usage": usage_delta,
        }))

    def run(self, task: str, on_event: EventHandler | None = None) -> str:
        undo_task_id = self.workspace.begin_undo_task(task)
        try:
            result = self._run_task(task, on_event)
        except BaseException:
            self.workspace.finish_undo_task(undo_task_id, "interrupted")
            raise
        self.workspace.finish_undo_task(undo_task_id, self._last_task_status)
        return result

    def _run_task(self, task: str, on_event: EventHandler | None = None) -> str:
        emit = on_event or (lambda _: None)
        started_at = time.perf_counter()
        usage_before = dict(getattr(self.client, "total_usage", {}) or {})
        changed_files = False
        verification_passed = False
        completion_summary: str | None = None
        self._last_task_status = "incomplete"
        self._append_message({"role": "user", "content": task})
        for step in range(1, self.config.max_steps + 1):
            context = self._trim_context()
            emit(("thinking", {"step": step, "max_steps": self.config.max_steps}))
            streamed = False
            stream_complete = getattr(self.client, "complete_stream", None)
            previous_debug = getattr(self.client, "debug_handler", None)
            if hasattr(self.client, "debug_handler"):
                self.client.debug_handler = lambda kind, payload: emit((kind, payload))
            try:
                if callable(stream_complete):
                    streamed = True
                    assistant = stream_complete(
                        context,
                        TOOL_SCHEMAS,
                        on_delta=lambda text: emit(("assistant_delta", text)),
                    )
                else:
                    assistant = self.client.complete(context, TOOL_SCHEMAS)
            finally:
                if hasattr(self.client, "debug_handler"):
                    self.client.debug_handler = previous_debug
            if not isinstance(assistant, dict):
                raise ModelError(f"model returned an invalid assistant message: {assistant!r}")
            tool_calls = assistant.get("tool_calls") or []
            content = assistant.get("content") or ""
            if not streamed and isinstance(content, str) and content:
                emit(("assistant_delta", content))
            self._append_message({"role": "assistant", "content": content, **({"tool_calls": tool_calls} if tool_calls else {})})
            if not tool_calls:
                if changed_files:
                    result = "尚未确认完成：本轮修改过文件，请先调用 verify_task 执行测试/检查，再调用 finish_task 提交完成。"
                    emit(("verification_required", {"reason": "file_changes_without_finish", "step": step}))
                    self._append_message({"role": "system", "content": result})
                    continue
                result = content.strip() or "完成。"
                if not content:
                    emit(("assistant_delta", result))
                self._last_task_status = "answered"
                self._run_end(emit, step, started_at, result, usage_before)
                return result
            pending_images: list[dict[str, Any]] = []
            pending_image_names: list[str] = []
            for call in tool_calls:
                function = call.get("function", {}) if isinstance(call, dict) else {}
                name = function.get("name", "") if isinstance(function, dict) else ""
                raw_args = function.get("arguments", {}) if isinstance(function, dict) else {}
                tool_id = call.get("id", name) if isinstance(call, dict) else name
                emit(("tool_call", {"name": name, "arguments": raw_args, "tool_call_id": tool_id, "raw": call}))
                try:
                    arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must be an object")
                except Exception as exc:
                    emit(("tool_start", {"name": name, "arguments": raw_args, "tool_call_id": tool_id}))
                    error = str(exc)
                    emit(("tool_end", {
                        "name": name,
                        "arguments": raw_args,
                        "tool_call_id": tool_id,
                        "ok": False,
                        "error": error,
                        "result": f"ERROR: {error}",
                        "elapsed": 0.0,
                    }))
                    result = f"ERROR: {error}"
                    self.workspace._audit("tool_error", tool=name, error=error)
                    self._append_message({"role": "tool", "tool_call_id": tool_id, "content": result})
                    continue

                preview: dict[str, Any] | None = None
                preview_error: str | None = None
                try:
                    preview = self.workspace.preview(name, arguments)
                except Exception as exc:
                    preview_error = str(exc)
                emit(("tool_start", {
                    "name": name,
                    "arguments": arguments,
                    "tool_call_id": tool_id,
                    "preview": preview,
                    "preview_error": preview_error,
                }))
                tool_started_at = time.perf_counter()
                try:
                    if name == "finish_task":
                        result = "finish_task is evaluated by the local completion gate"
                    else:
                        result = self.workspace.execute(name, arguments)
                    ok = self._tool_ok(name, result)
                    if ok:
                        error = None
                    elif name == "run_command" and result.startswith("exit_code="):
                        error = result.split("\n", 1)[0].replace("exit_code=", "exit ")
                    else:
                        error = result.split("\n", 1)[0].removeprefix("ERROR:").strip() or result.split("\n", 1)[0]
                except Exception as exc:  # tool failures are recoverable model observations
                    result = f"ERROR: {exc}"
                    ok = False
                    error = str(exc)
                    self.workspace._audit("tool_error", tool=name, error=error)
                if ok and name in {"write_file", "apply_patch"}:
                    changed_files = True
                    verification_passed = False
                if ok and name == "verify_task":
                    verification_passed = True
                if name == "finish_task":
                    summary = arguments.get("summary", "")
                    if not isinstance(summary, str) or not summary.strip():
                        result = "ERROR: finish_task requires a non-empty summary"
                        ok = False
                        error = "finish_task requires a non-empty summary"
                    elif changed_files and not verification_passed:
                        result = "ERROR: cannot finish a task with file changes before verify_task succeeds"
                        ok = False
                        error = "verification is required after file changes"
                    else:
                        completion_summary = summary.strip()
                        result = f"TASK_COMPLETED: {completion_summary}"
                        ok = True
                        error = None
                    if not ok:
                        emit(("completion_rejected", {
                            "reason": error,
                            "verification_passed": verification_passed,
                            "changed_files": changed_files,
                        }))
                emit(("tool_end", {
                    "name": name,
                    "arguments": arguments,
                    "tool_call_id": tool_id,
                    "ok": ok,
                    "error": error,
                    "result": result,
                    "full_result": self.workspace.last_full_result,
                    "elapsed": time.perf_counter() - tool_started_at,
                }))
                self._append_message({"role": "tool", "tool_call_id": tool_id, "content": result})
                if ok and name in {"read_image", "read_pdf", "read_docx"} and self.workspace.last_images:
                    pending_images.extend(self.workspace.last_images)
                    pending_image_names.extend(
                        str(image.get("name", "workspace image"))
                        for image in self.workspace.last_images
                    )
                if completion_summary is not None:
                    self._last_task_status = "verified" if changed_files else "completed"
                    self._run_end(emit, step, started_at, completion_summary, usage_before)
                    return completion_summary
            if pending_images:
                self._append_message({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Visual attachments from the previous tool call. Inspect them directly and use their visible content in the task. Files: " + ", ".join(pending_image_names),
                        },
                        *pending_images,
                    ],
                })
        result = "达到最大步骤数，任务尚未确认完成。请检查工作区后继续。"
        emit(("assistant_delta", result))
        self._run_end(emit, self.config.max_steps, started_at, result, usage_before)
        return result


def make_agent(root: str | Path, config: AgentConfig | None = None, approve: Callable[[str], bool] | None = None,
               audit_path: str | Path | None = None, api_key: str | None = None,
               messages: list[dict[str, Any]] | None = None,
               on_history_change: Callable[[list[dict[str, Any]]], None] | None = None,
               undo_manager: Any | None = None) -> CodingAgent:
    cfg = config or AgentConfig()
    client = OpenAICompatibleClient(cfg, api_key=api_key)
    workspace = Workspace(root, approve=approve, config=cfg, audit_path=audit_path, undo_manager=undo_manager)
    return CodingAgent(client, workspace, cfg, messages=messages, on_history_change=on_history_change)
