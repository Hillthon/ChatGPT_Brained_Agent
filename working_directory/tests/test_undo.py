import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from undo import UndoConflict, UndoError, UndoManager


class UndoTests(unittest.TestCase):
    def make_manager(self, directory, root, session="session-1"):
        return UndoManager(Path(directory) / "snapshots", session, root)

    def test_undo_restores_an_existing_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            file = root / "app.py"
            file.write_bytes(b"before\n")
            manager = self.make_manager(directory, root)
            manager.begin_task("edit app")
            pending = manager.capture(file, "write_file")
            file.write_bytes(b"after\n")
            manager.commit(pending)

            action = manager.undo_last()

            self.assertEqual(action.path, "app.py")
            self.assertEqual(file.read_bytes(), b"before\n")
            self.assertEqual(manager.active_actions(), [])

    def test_undo_removes_a_file_created_by_the_agent(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            file = root / "new.py"
            manager = self.make_manager(directory, root)
            manager.begin_task("create file")
            pending = manager.capture(file, "write_file")
            file.write_text("created", encoding="utf-8")
            manager.commit(pending)

            manager.undo_last()

            self.assertFalse(file.exists())

    def test_undo_refuses_to_overwrite_a_later_user_edit(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            file = root / "app.py"
            file.write_text("before", encoding="utf-8")
            manager = self.make_manager(directory, root)
            manager.begin_task("edit")
            pending = manager.capture(file, "apply_patch")
            file.write_text("agent", encoding="utf-8")
            manager.commit(pending)
            file.write_text("user", encoding="utf-8")

            with self.assertRaises(UndoConflict):
                manager.undo_last()

            self.assertEqual(file.read_text(encoding="utf-8"), "user")

    def test_rollback_latest_task_restores_multiple_edits_in_reverse(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            file = root / "app.py"
            file.write_text("zero", encoding="utf-8")
            manager = self.make_manager(directory, root)
            task_id = manager.begin_task("two edits")
            first = manager.capture(file, "write_file")
            file.write_text("one", encoding="utf-8")
            manager.commit(first)
            second = manager.capture(file, "apply_patch")
            file.write_text("two", encoding="utf-8")
            manager.commit(second)
            manager.finish_task(task_id, "completed")

            task, actions = manager.rollback_latest_task()

            self.assertEqual(task.id, task_id)
            self.assertEqual(len(actions), 2)
            self.assertEqual(file.read_text(encoding="utf-8"), "zero")

    def test_two_edits_to_the_same_file_can_be_undone_one_at_a_time(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            file = root / "app.py"
            file.write_text("zero", encoding="utf-8")
            manager = self.make_manager(directory, root)
            manager.begin_task("two edits")
            first = manager.capture(file, "write_file")
            file.write_text("one", encoding="utf-8")
            manager.commit(first)
            second = manager.capture(file, "apply_patch")
            file.write_text("two", encoding="utf-8")
            manager.commit(second)

            manager.undo_last()
            self.assertEqual(file.read_text(encoding="utf-8"), "one")
            manager.undo_last()
            self.assertEqual(file.read_text(encoding="utf-8"), "zero")

    def test_rollback_latest_task_restores_multiple_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            existing = root / "app.py"
            created = root / "new.py"
            existing.write_text("before", encoding="utf-8")
            manager = self.make_manager(directory, root)
            task_id = manager.begin_task("edit two files")
            old_file = manager.capture(existing, "write_file")
            existing.write_text("after", encoding="utf-8")
            manager.commit(old_file)
            new_file = manager.capture(created, "write_file")
            created.write_text("created", encoding="utf-8")
            manager.commit(new_file)
            manager.finish_task(task_id, "completed")

            _, actions = manager.rollback_latest_task()

            self.assertEqual(len(actions), 2)
            self.assertEqual(existing.read_text(encoding="utf-8"), "before")
            self.assertFalse(created.exists())

    def test_snapshots_survive_process_restart(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            file = root / "app.py"
            file.write_text("before", encoding="utf-8")
            manager = self.make_manager(directory, root)
            manager.begin_task("edit")
            pending = manager.capture(file, "write_file")
            file.write_text("after", encoding="utf-8")
            manager.commit(pending)

            resumed = self.make_manager(directory, root)
            resumed.undo_last()

            self.assertEqual(file.read_text(encoding="utf-8"), "before")

    def test_session_and_workspace_are_validated(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            other = Path(directory) / "other"
            root.mkdir()
            other.mkdir()
            self.make_manager(directory, root)
            with self.assertRaises(UndoError):
                self.make_manager(directory, other)
            with self.assertRaises(UndoError):
                UndoManager(Path(directory) / "snapshots", "../bad", root)

    def test_sessions_have_isolated_checkpoint_histories(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            file = root / "app.py"
            file.write_text("before", encoding="utf-8")
            first = self.make_manager(directory, root, session="session-a")
            first.begin_task("edit")
            pending = first.capture(file, "write_file")
            file.write_text("after", encoding="utf-8")
            first.commit(pending)

            second = self.make_manager(directory, root, session="session-b")
            with self.assertRaises(UndoError):
                second.undo_last()
            self.assertEqual(file.read_text(encoding="utf-8"), "after")
            first.undo_last()
            self.assertEqual(file.read_text(encoding="utf-8"), "before")

    def test_checkpoint_save_failure_restores_the_original_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            file = root / "app.py"
            file.write_text("before", encoding="utf-8")
            manager = self.make_manager(directory, root)
            manager.begin_task("edit")
            pending = manager.capture(file, "write_file")
            file.write_text("after", encoding="utf-8")

            with patch.object(manager, "_save", side_effect=UndoError("disk full")):
                with self.assertRaisesRegex(UndoError, "edit was restored"):
                    manager.commit(pending)

            self.assertEqual(file.read_text(encoding="utf-8"), "before")
            self.assertEqual(manager.active_actions(), [])

    def test_large_snapshot_is_rejected_before_edit(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            file = root / "large.bin"
            file.write_bytes(b"1234")
            manager = UndoManager(Path(directory) / "snapshots", "session-1", root, max_snapshot_bytes=3)
            manager.begin_task("edit")
            with self.assertRaises(UndoError):
                manager.capture(file, "write_file")
            self.assertEqual(file.read_bytes(), b"1234")


if __name__ == "__main__":
    unittest.main()
