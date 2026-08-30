import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent import AgentConfig, AgentError, CodingAgent, TOOL_SCHEMAS, Workspace


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.config = AgentConfig(max_steps=4)

    def complete(self, messages, tools):
        return next(self.replies)


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


if __name__ == "__main__":
    unittest.main()
