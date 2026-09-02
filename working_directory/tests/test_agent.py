import io
import json
import urllib.error
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agent import AgentConfig, AgentError, CodingAgent, ModelError, OpenAICompatibleClient, TOOL_SCHEMAS, Workspace
from undo import UndoManager


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.config = AgentConfig(max_steps=4)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append(deepcopy(messages))
        return next(self.replies)



class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class _FakeStreamResponse:
    def __init__(self, text):
        self.lines = [line.encode("utf-8") for line in text.splitlines(keepends=True)]
        self.closed = False

    def __iter__(self):
        return iter(self.lines)

    def close(self):
        self.closed = True


class AgentTests(unittest.TestCase):
    def test_workspace_cannot_escape(self):
        workspace = Workspace(Path.cwd())
        with self.assertRaises(AgentError):
            workspace.read_file("../secret.txt")

    def test_tool_schema_marks_non_default_arguments_required(self):
        schemas = {item["function"]["name"]: item["function"]["parameters"] for item in TOOL_SCHEMAS}
        self.assertEqual(schemas["write_file"]["required"], ["path", "content"])
        self.assertEqual(schemas["list_files"]["required"], [])

    def test_write_requires_approval(self):
        with TemporaryDirectory() as directory:
            workspace = Workspace(directory, approve=lambda _: False)
            self.assertTrue(workspace.write_file("a.txt", "hello").startswith("DENIED"))
            self.assertFalse((Path(directory) / "a.txt").exists())

    def test_write_and_read(self):
        with TemporaryDirectory() as directory:
            workspace = Workspace(directory, approve=lambda _: True)
            self.assertIn("wrote", workspace.write_file("a.txt", "one\ntwo\n"))
            self.assertIn("2 | two", workspace.read_file("a.txt"))

    def test_command_guardrails(self):
        with TemporaryDirectory() as directory:
            workspace = Workspace(directory, approve=lambda _: True)
            self.assertTrue(workspace.run_command("rm -rf .").startswith("DENIED"))

    def test_audit_log_records_mutations(self):
        with TemporaryDirectory() as directory:
            audit = Path(directory) / "audit.jsonl"
            workspace = Workspace(directory, approve=lambda _: True, audit_path="audit.jsonl")
            workspace.write_file("a.txt", "ok")
            records = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[0]["event"], "write_file")
            self.assertEqual(records[0]["path"], "a.txt")

    def test_apply_unified_patch(self):
        with TemporaryDirectory() as directory:
            workspace = Workspace(directory, approve=lambda _: True)
            workspace.write_file("a.txt", "one\ntwo\n")
            patch = "@@ -1,2 +1,2 @@\n one\n-two\n+three\n"
            self.assertIn("patched", workspace.apply_patch("a.txt", patch))
            self.assertEqual((Path(directory) / "a.txt").read_text(), "one\nthree\n")

    def test_agent_executes_tool_then_finishes(self):
        replies = [
            {"content": "", "tool_calls": [{"id": "1", "function": {"name": "write_file", "arguments": json.dumps({"path": "x.txt", "content": "ok"})}}]},
            {"content": "", "tool_calls": [{"id": "2", "function": {"name": "verify_task", "arguments": json.dumps({"command": "echo verification"})}}]},
            {"content": "", "tool_calls": [{"id": "3", "function": {"name": "finish_task", "arguments": json.dumps({"summary": "done"})}}]},
        ]
        with TemporaryDirectory() as directory:
            workspace = Workspace(directory, approve=lambda _: True)
            agent = CodingAgent(FakeClient(replies), workspace)
            self.assertEqual(agent.run("create x"), "done")
            self.assertEqual((Path(directory) / "x.txt").read_text(), "ok")

    def test_agent_file_tools_create_persistent_undo_checkpoints(self):
        replies = [
            {"content": "", "tool_calls": [{"id": "1", "function": {"name": "write_file", "arguments": json.dumps({"path": "x.txt", "content": "after"})}}]},
            {"content": "", "tool_calls": [{"id": "2", "function": {"name": "verify_task", "arguments": json.dumps({"command": "echo verification"})}}]},
            {"content": "", "tool_calls": [{"id": "3", "function": {"name": "finish_task", "arguments": json.dumps({"summary": "done"})}}]},
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            file = root / "x.txt"
            file.write_text("before", encoding="utf-8")
            manager = UndoManager(Path(directory) / "snapshots", "session-1", root)
            workspace = Workspace(root, approve=lambda _: True, undo_manager=manager)
            agent = CodingAgent(FakeClient(replies), workspace)

            self.assertEqual(agent.run("edit x"), "done")
            self.assertEqual(file.read_text(encoding="utf-8"), "after")
            self.assertEqual(len(workspace.undo_checkpoints()), 1)

            action = workspace.undo_last()
            self.assertEqual(action.tool, "write_file")
            self.assertEqual(file.read_text(encoding="utf-8"), "before")

    def test_agent_undo_and_task_rollback_end_to_end(self):
        replies = [
            {"content": "", "tool_calls": [{"id": "1", "function": {"name": "write_file", "arguments": json.dumps({"path": "app.txt", "content": "first"})}}]},
            {"content": "", "tool_calls": [{"id": "v1", "function": {"name": "verify_task", "arguments": json.dumps({"command": "echo verification"})}}]},
            {"content": "", "tool_calls": [{"id": "f1", "function": {"name": "finish_task", "arguments": json.dumps({"summary": "first task done"})}}]},
            {"content": "", "tool_calls": [{"id": "2", "function": {"name": "write_file", "arguments": json.dumps({"path": "app.txt", "content": "second"})}}]},
            {"content": "", "tool_calls": [{"id": "3", "function": {"name": "write_file", "arguments": json.dumps({"path": "new.txt", "content": "created"})}}]},
            {"content": "", "tool_calls": [{"id": "v2", "function": {"name": "verify_task", "arguments": json.dumps({"command": "echo verification"})}}]},
            {"content": "", "tool_calls": [{"id": "f2", "function": {"name": "finish_task", "arguments": json.dumps({"summary": "second task done"})}}]},
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            existing = root / "app.txt"
            created = root / "new.txt"
            existing.write_text("original", encoding="utf-8")
            manager = UndoManager(Path(directory) / "snapshots", "session-1", root)
            workspace = Workspace(root, approve=lambda _: True, undo_manager=manager)
            agent = CodingAgent(FakeClient(replies), workspace)

            self.assertEqual(agent.run("first edit"), "first task done")
            self.assertEqual(len(workspace.undo_checkpoints()), 1)
            workspace.undo_last()
            self.assertEqual(existing.read_text(encoding="utf-8"), "original")

            self.assertEqual(agent.run("edit and create"), "second task done")
            self.assertEqual(existing.read_text(encoding="utf-8"), "second")
            self.assertEqual(created.read_text(encoding="utf-8"), "created")
            task, actions = workspace.rollback_latest_task()
            self.assertEqual(len(actions), 2)
            self.assertEqual(task.prompt, "edit and create")
            self.assertEqual(existing.read_text(encoding="utf-8"), "original")
            self.assertFalse(created.exists())
            self.assertEqual(workspace.undo_checkpoints(), [])

    def test_completion_requires_verification_after_file_changes(self):
        replies = [
            {"content": "", "tool_calls": [{"id": "1", "function": {"name": "write_file", "arguments": json.dumps({"path": "x.txt", "content": "after"})}}]},
            {"content": "", "tool_calls": [{"id": "2", "function": {"name": "finish_task", "arguments": json.dumps({"summary": "premature"})}}]},
            {"content": "", "tool_calls": [{"id": "3", "function": {"name": "verify_task", "arguments": json.dumps({"command": "echo verification"})}}]},
            {"content": "", "tool_calls": [{"id": "4", "function": {"name": "finish_task", "arguments": json.dumps({"summary": "verified"})}}]},
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            manager = UndoManager(Path(directory) / "snapshots", "session-1", root)
            workspace = Workspace(root, approve=lambda _: True, undo_manager=manager)
            events = []

            result = CodingAgent(FakeClient(replies), workspace).run("edit x", on_event=events.append)

            self.assertEqual(result, "verified")
            self.assertIn("completion_rejected", [kind for kind, _ in events])
            self.assertEqual((root / "x.txt").read_text(encoding="utf-8"), "after")

    def test_failed_verification_does_not_close_the_task(self):
        replies = [
            {"content": "", "tool_calls": [{"id": "1", "function": {"name": "write_file", "arguments": json.dumps({"path": "x.txt", "content": "after"})}}]},
            {"content": "", "tool_calls": [{"id": "2", "function": {"name": "verify_task", "arguments": json.dumps({"command": "python -c \"import sys; sys.exit(1)\""})}}]},
            {"content": "", "tool_calls": [{"id": "3", "function": {"name": "finish_task", "arguments": json.dumps({"summary": "still broken"})}}]},
            {"content": "", "tool_calls": [{"id": "4", "function": {"name": "verify_task", "arguments": json.dumps({"command": "echo fixed"})}}]},
            {"content": "", "tool_calls": [{"id": "5", "function": {"name": "finish_task", "arguments": json.dumps({"summary": "verified after retry"})}}]},
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            manager = UndoManager(Path(directory) / "snapshots", "session-1", root)
            workspace = Workspace(root, approve=lambda _: True, undo_manager=manager)
            events = []

            result = CodingAgent(
                FakeClient(replies), workspace, config=AgentConfig(max_steps=6)
            ).run("edit x", on_event=events.append)

            self.assertEqual(result, "verified after retry")
            failed = [payload for kind, payload in events if kind == "tool_end" and payload["name"] == "verify_task" and not payload["ok"]]
            self.assertEqual(len(failed), 1)

    def test_denied_write_does_not_create_an_undo_checkpoint(self):
        replies = [
            {"content": "", "tool_calls": [{"id": "1", "function": {"name": "write_file", "arguments": json.dumps({"path": "x.txt", "content": "after"})}}]},
            {"content": "denied", "tool_calls": []},
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            manager = UndoManager(Path(directory) / "snapshots", "session-1", root)
            workspace = Workspace(root, approve=lambda _: False, undo_manager=manager)
            CodingAgent(FakeClient(replies), workspace).run("edit x")
            self.assertEqual(workspace.undo_checkpoints(), [])
            self.assertFalse((root / "x.txt").exists())

    def test_failed_patch_does_not_create_an_undo_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            file = root / "x.txt"
            file.write_text("before\n", encoding="utf-8")
            manager = UndoManager(Path(directory) / "snapshots", "session-1", root)
            manager.begin_task("bad patch")
            workspace = Workspace(root, approve=lambda _: True, undo_manager=manager)

            with self.assertRaises(AgentError):
                workspace.apply_patch("x.txt", "not a unified diff")

            self.assertEqual(file.read_text(encoding="utf-8"), "before\n")
            self.assertEqual(workspace.undo_checkpoints(), [])

    def test_agent_emits_structured_events(self):
        replies = [
            {"content": "", "tool_calls": [{"id": "1", "function": {"name": "read_file", "arguments": json.dumps({"path": "x.txt", "start_line": 1})}}]},
            {"content": "done", "tool_calls": []},
        ]
        events = []
        with TemporaryDirectory() as directory:
            (Path(directory) / "x.txt").write_text("ok\n", encoding="utf-8")
            agent = CodingAgent(FakeClient(replies), Workspace(directory))
            self.assertEqual(agent.run("read x", on_event=events.append), "done")
        kinds = [event[0] for event in events]
        self.assertIn("thinking", kinds)
        self.assertIn("tool_start", kinds)
        self.assertIn("tool_end", kinds)
        self.assertIn("assistant_delta", kinds)
        self.assertIn("run_end", kinds)
        self.assertTrue(all(isinstance(event, tuple) for event in events))
        self.assertFalse(any(isinstance(event, str) and "step" in event for event in events))

    def test_chat_stream_forwards_deltas_and_reassembles_tool_calls(self):
        stream = """data: {json1}

data: {json2}

data: {json3}

data: [DONE]

""".format(
            json1=json.dumps({"choices": [{"delta": {"role": "assistant", "content": "hel"}}]}),
            json2=json.dumps({"choices": [{"delta": {"content": "lo", "tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "read_file", "arguments": "{\"path\":\"x"}}]}}]}),
            json3=json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ".txt\"}"}}]}}]}),
        )
        response = _FakeStreamResponse(stream)
        config = AgentConfig(base_url="https://relay.test/codex/v1", api_mode="chat")
        client = OpenAICompatibleClient(config, api_key="relay-secret")
        deltas = []
        with patch("urllib.request.urlopen", return_value=response) as mocked:
            message = client.complete_stream([{"role": "user", "content": "read"}], TOOL_SCHEMAS, deltas.append)
        body = json.loads(mocked.call_args.args[0].data.decode("utf-8"))
        self.assertTrue(body["stream"])
        self.assertEqual("".join(deltas), "hello")
        self.assertEqual(message["content"], "hello")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"]), {"path": "x.txt"})
        self.assertTrue(response.closed)

    def test_stream_emits_model_debug_events_only_when_requested(self):
        stream = "data: {payload}\n\ndata: [DONE]\n\n".format(
            payload=json.dumps({"choices": [{"delta": {"content": "ok"}}]})
        )
        response = _FakeStreamResponse(stream)
        config = AgentConfig(base_url="https://relay.test/codex/v1", api_mode="chat")
        client = OpenAICompatibleClient(config, api_key="relay-secret")
        events = []
        with patch("urllib.request.urlopen", return_value=response):
            agent = CodingAgent(client, Workspace(Path.cwd()))
            agent.run("hello", on_event=events.append)
        self.assertEqual([event[0] for event in events if event[0].startswith("model_")], [
            "model_request", "model_chunk",
        ])

    def test_responses_stream_forwards_text_and_reassembles_function_arguments(self):
        events = [
            {"type": "response.output_text.delta", "delta": "ok"},
            {"type": "response.output_item.added", "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read_file", "arguments": ""}},
            {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": "{\"path\":\"x.txt\"}"},
            {"type": "response.completed", "response": {"output": [{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read_file", "arguments": "{\"path\":\"x.txt\"}"}]}} ,
        ]
        stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
        response = _FakeStreamResponse(stream)
        config = AgentConfig(base_url="https://relay.test/codex/v1", api_mode="responses")
        client = OpenAICompatibleClient(config, api_key="relay-secret")
        deltas = []
        with patch("urllib.request.urlopen", return_value=response):
            message = client.complete_stream([{"role": "user", "content": "read"}], TOOL_SCHEMAS, deltas.append)
        self.assertEqual("".join(deltas), "ok")
        self.assertEqual(message["content"], "ok")
        self.assertEqual(len(message["tool_calls"]), 1)
        self.assertEqual(message["tool_calls"][0]["id"], "call_1")
        self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"]), {"path": "x.txt"})

    def test_stream_accepts_a_non_sse_responses_fallback_payload(self):
        response = _FakeStreamResponse(json.dumps({"output_text": "ordinary response"}) + "\n")
        config = AgentConfig(base_url="https://relay.test/codex/v1", api_mode="responses")
        client = OpenAICompatibleClient(config, api_key="relay-secret")
        deltas = []
        with patch("urllib.request.urlopen", return_value=response):
            message = client.complete_stream([{"role": "user", "content": "hello"}], TOOL_SCHEMAS, deltas.append)
        self.assertEqual(message["content"], "ordinary response")
        self.assertEqual(deltas, ["ordinary response"])

    def test_tool_failure_is_reported_as_a_failed_tool_event(self):
        client = FakeClient([
            {"content": "", "tool_calls": [{"id": "1", "function": {"name": "read_file", "arguments": json.dumps({"path": "missing.py"})}}]},
            {"content": "recovered", "tool_calls": []},
        ])
        events = []
        with TemporaryDirectory() as directory:
            agent = CodingAgent(client, Workspace(directory))
            agent.run("read missing", on_event=events.append)
        failed = [payload for kind, payload in events if kind == "tool_end" and not payload["ok"]]
        self.assertEqual(len(failed), 1)
        self.assertIn("not a file", failed[0]["error"])

    def test_renderer_hides_read_results_and_shows_diff_and_command_tail(self):
        from cli import Renderer

        output = io.StringIO()
        renderer = Renderer(stream=output, color=False)
        renderer.on_event(("tool_start", {"name": "apply_patch", "arguments": {"path": "x.py"}, "preview": {"kind": "diff", "diff": "@@ -1 +1 @@\n-old\n+new\n"}}))
        renderer.on_event(("tool_end", {"name": "apply_patch", "arguments": {"path": "x.py"}, "result": "patched x.py", "ok": True, "elapsed": 0.1}))
        renderer.on_event(("tool_start", {"name": "read_file", "arguments": {"path": "x.py"}}))
        renderer.on_event(("tool_end", {"name": "read_file", "arguments": {"path": "x.py"}, "result": "secret body", "ok": True, "elapsed": 0.1}))
        renderer.on_event(("tool_start", {"name": "run_command", "arguments": {"command": "pytest"}}))
        renderer.on_event(("tool_end", {"name": "run_command", "arguments": {"command": "pytest"}, "result": "exit_code=0\nline 1\nline 2", "ok": True, "elapsed": 0.1}))
        rendered = output.getvalue()
        self.assertIn("@@ -1 +1 @@", rendered)
        self.assertIn("+new", rendered)
        self.assertIn("line 2", rendered)
        self.assertNotIn("secret body", rendered)

    def test_write_file_preview_is_a_readable_unified_diff(self):
        with TemporaryDirectory() as directory:
            preview = Workspace(directory).preview("write_file", {"path": "new.py", "content": "print(1)\n"})
        self.assertEqual(preview["diff"].splitlines()[:3], ["--- a/new.py", "+++ b/new.py", "@@ -0,0 +1 @@"])
        self.assertIn("+print(1)", preview["diff"])

    def test_renderer_quiet_keeps_thinking_and_answer_but_hides_actions(self):
        from cli import Renderer

        output = io.StringIO()
        renderer = Renderer(stream=output, color=False, quiet=True)
        renderer.on_event(("thinking", {"step": 1, "max_steps": 100}))
        renderer.on_event(("tool_start", {"name": "read_file", "arguments": {"path": "x.py"}}))
        renderer.on_event(("tool_end", {"name": "read_file", "arguments": {"path": "x.py"}, "result": "hidden", "ok": True, "elapsed": 0.1}))
        renderer.on_event(("assistant_delta", "answer"))
        renderer.on_event(("run_end", {"steps": 1, "elapsed": 0.1, "usage": {}}))
        rendered = output.getvalue()
        self.assertIn("Thinking", rendered)
        self.assertIn("answer", rendered)
        self.assertNotIn("read_file", rendered)
        self.assertNotIn("hidden", rendered)

    def test_agent_reuses_context_across_tasks(self):
        client = FakeClient([
            {"content": "The value is 7", "tool_calls": []},
            {"content": "You said 7", "tool_calls": []},
        ])
        with TemporaryDirectory() as directory:
            agent = CodingAgent(client, Workspace(directory))
            self.assertEqual(agent.run("Remember the value 7"), "The value is 7")
            self.assertEqual(agent.run("What value did I give you?"), "You said 7")
        second_call = client.calls[1]
        self.assertEqual(
            [(message["role"], message["content"]) for message in second_call[1:]],
            [
                ("user", "Remember the value 7"),
                ("assistant", "The value is 7"),
                ("user", "What value did I give you?"),
            ],
        )

    def test_context_trimming_keeps_the_newest_complete_turn(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old response"},
            {"role": "user", "content": "current request"},
        ]
        with TemporaryDirectory() as directory:
            agent = CodingAgent(
                FakeClient([]),
                Workspace(directory),
                config=AgentConfig(max_context_chars=1),
                messages=messages,
            )
            agent._trim_context()
        self.assertEqual(agent.messages, [messages[0], messages[-1]])

    def test_chat_client_uses_relay_endpoint_and_auth(self):
        response = _FakeResponse({"choices": [{"message": {"role": "assistant", "content": "done"}}]})
        config = AgentConfig(base_url="https://relay.test/codex/v1", api_mode="chat")
        client = OpenAICompatibleClient(config, api_key="relay-secret")
        with patch("urllib.request.urlopen", return_value=response) as mocked:
            message = client.complete([{"role": "user", "content": "hello"}], TOOL_SCHEMAS)
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "https://relay.test/codex/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer relay-secret")
        self.assertEqual(message["content"], "done")

    def test_responses_client_translates_function_calls(self):
        response = _FakeResponse({
            "output": [{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "list_files", "arguments": json.dumps({"path": "."})}],
        })
        config = AgentConfig(base_url="https://relay.test/codex/v1", api_mode="responses")
        client = OpenAICompatibleClient(config, api_key="relay-secret")
        with patch("urllib.request.urlopen", return_value=response) as mocked:
            message = client.complete([{"role": "user", "content": "list"}], TOOL_SCHEMAS)
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "https://relay.test/codex/v1/responses")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["input"][0]["content"], "list")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "list_files")
        self.assertEqual(message["tool_calls"][0]["id"], "call_1")

    def test_auto_mode_falls_back_to_responses_on_not_found(self):
        failure = urllib.error.HTTPError("https://relay.test/codex/v1/chat/completions", 404, "not found", {}, io.BytesIO(b'{"error":{"message":"unsupported"}}'))
        config = AgentConfig(base_url="https://relay.test/codex/v1", api_mode="auto")
        client = OpenAICompatibleClient(config, api_key="relay-secret")
        with patch("urllib.request.urlopen", side_effect=[failure, _FakeResponse({"output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]})]) as mocked:
            message = client.complete([{"role": "user", "content": "hello"}], TOOL_SCHEMAS)
        self.assertEqual(message["content"], "ok")
        self.assertEqual([call.args[0].full_url for call in mocked.call_args_list], [
            "https://relay.test/codex/v1/chat/completions",
            "https://relay.test/codex/v1/responses",
        ])

    def test_responses_message_accepts_string_content(self):
        message = OpenAICompatibleClient._responses_message({
            "output": [{"type": "message", "content": "plain text"}],
        })
        self.assertEqual(message["content"], "plain text")
    def test_client_rejects_unknown_api_mode(self):
        with self.assertRaises(ModelError):
            OpenAICompatibleClient(AgentConfig(api_mode="legacy"), api_key="key")

if __name__ == "__main__":
    unittest.main()
