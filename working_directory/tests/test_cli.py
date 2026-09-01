import io
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from contextlib import redirect_stdout

import cli
from agent import SYSTEM_PROMPT
from session import SessionStore


class FakeAgent:
    def __init__(self, messages, checkpoint, tasks):
        self.messages = messages or [{"role": "system", "content": SYSTEM_PROMPT}]
        self.checkpoint = checkpoint
        self.tasks = tasks

    def run(self, task, on_event=None):
        self.tasks.append(task)
        self.messages.extend([
            {"role": "user", "content": task},
            {"role": "assistant", "content": f"done: {task}"},
        ])
        self.checkpoint(self.messages)
        return f"done: {task}"


class CLITests(unittest.TestCase):
    def test_cli_keeps_prompting_after_the_initial_task(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            sessions = Path(directory) / "sessions"
            root.mkdir()
            tasks = []

            def fake_make_agent(*args, **kwargs):
                return FakeAgent(kwargs.get("messages"), kwargs["on_history_change"], tasks)

            argv = [
                "cli.py", "--root", str(root), "--session-dir", str(sessions), "first task",
            ]
            with patch.object(sys, "argv", argv), \
                    patch("cli.make_agent", side_effect=fake_make_agent), \
                    patch("builtins.input", side_effect=["second task", "/exit"]), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(), 0)

            self.assertEqual(tasks, ["first task", "second task"])
            saved = SessionStore(sessions).list_sessions(root)
            self.assertEqual(len(saved), 1)
            self.assertEqual(
                [message["content"] for message in saved[0].messages if message["role"] == "user"],
                ["first task", "second task"],
            )

    def test_once_exits_without_another_prompt(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            tasks = []

            def fake_make_agent(*args, **kwargs):
                return FakeAgent(kwargs.get("messages"), kwargs["on_history_change"], tasks)

            argv = [
                "cli.py", "--root", str(root), "--session-dir", str(Path(directory) / "sessions"),
                "--once", "only task",
            ]
            with patch.object(sys, "argv", argv), \
                    patch("cli.make_agent", side_effect=fake_make_agent), \
                    patch("builtins.input") as mocked_input, \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(), 0)
            mocked_input.assert_not_called()
            self.assertEqual(tasks, ["only task"])

    def test_new_command_creates_an_isolated_session(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            sessions = Path(directory) / "sessions"
            root.mkdir()
            tasks = []

            def fake_make_agent(*args, **kwargs):
                return FakeAgent(kwargs.get("messages"), kwargs["on_history_change"], tasks)

            argv = [
                "cli.py", "--root", str(root), "--session-dir", str(sessions), "first task",
            ]
            with patch.object(sys, "argv", argv), \
                    patch("cli.make_agent", side_effect=fake_make_agent), \
                    patch("builtins.input", side_effect=["/new", "second task", "/exit"]), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(), 0)

            saved = SessionStore(sessions).list_sessions(root)
            self.assertEqual(len(saved), 2)
            histories = {
                tuple(message["content"] for message in item.messages if message["role"] == "user")
                for item in saved
            }
            self.assertEqual(histories, {("first task",), ("second task",)})

    def test_session_option_restores_saved_context(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            sessions = Path(directory) / "sessions"
            root.mkdir()
            store = SessionStore(sessions)
            saved = store.create(root)
            saved.messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "old task"},
                {"role": "assistant", "content": "done: old task"},
            ]
            store.save(saved)
            tasks = []

            def fake_make_agent(*args, **kwargs):
                return FakeAgent(kwargs.get("messages"), kwargs["on_history_change"], tasks)

            argv = [
                "cli.py", "--root", str(root), "--session-dir", str(sessions),
                "--session", saved.id, "--once", "follow-up task",
            ]
            with patch.object(sys, "argv", argv), \
                    patch("cli.make_agent", side_effect=fake_make_agent), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(), 0)

            restored = store.load(saved.id)
            user_messages = [
                message["content"] for message in restored.messages if message["role"] == "user"
            ]
            self.assertEqual(user_messages, ["old task", "follow-up task"])


if __name__ == "__main__":
    unittest.main()
