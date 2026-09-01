import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from agent import AgentConfig, AgentError, CodingAgent, Workspace
from session import SessionStore


class RecordingClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []
        self.config = AgentConfig(max_steps=4)

    def complete(self, messages, tools):
        self.calls.append(deepcopy(messages))
        return next(self.replies)


class SessionTests(unittest.TestCase):
    def test_store_saves_multiple_isolated_sessions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            store = SessionStore(Path(directory) / "sessions")
            first = store.create(root)
            first.messages = [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "first task"},
            ]
            store.save(first)
            second = store.create(root)
            second.messages = [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "second task"},
            ]
            store.save(second)

            self.assertNotEqual(first.id, second.id)
            self.assertEqual(store.load(first.id).messages[-1]["content"], "first task")
            self.assertEqual(store.load(second.id).messages[-1]["content"], "second task")
            self.assertEqual({item.title for item in store.list_sessions(root)}, {"first task", "second task"})
            self.assertEqual(list((Path(directory) / "sessions").glob("*.tmp")), [])

    def test_saved_history_can_resume_in_a_new_agent(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            store = SessionStore(Path(directory) / "sessions")
            session = store.create(root)

            def checkpoint(messages):
                session.messages = messages
                store.save(session)

            first_agent = CodingAgent(
                RecordingClient([{"content": "remembered", "tool_calls": []}]),
                Workspace(root),
                on_history_change=checkpoint,
            )
            first_agent.run("Remember alpha")

            resumed = store.load(session.id)
            second_client = RecordingClient([{"content": "alpha", "tool_calls": []}])
            second_agent = CodingAgent(second_client, Workspace(root), messages=resumed.messages)
            self.assertEqual(second_agent.run("What did I ask you to remember?"), "alpha")
            user_messages = [
                message["content"] for message in second_client.calls[0] if message["role"] == "user"
            ]
            self.assertEqual(user_messages, ["Remember alpha", "What did I ask you to remember?"])

    def test_session_ids_cannot_escape_the_store(self):
        with TemporaryDirectory() as directory:
            store = SessionStore(directory)
            with self.assertRaises(AgentError):
                store.load("../outside")


if __name__ == "__main__":
    unittest.main()
