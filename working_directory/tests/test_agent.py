import io
import json
import urllib.error
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agent import AgentConfig, AgentError, CodingAgent, ModelError, OpenAICompatibleClient, TOOL_SCHEMAS, Workspace


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
            {"content": "done", "tool_calls": []},
        ]
        with TemporaryDirectory() as directory:
            workspace = Workspace(directory, approve=lambda _: True)
            agent = CodingAgent(FakeClient(replies), workspace)
            self.assertEqual(agent.run("create x"), "done")
            self.assertEqual((Path(directory) / "x.txt").read_text(), "ok")

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
