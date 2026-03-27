from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from interleave_codebench.bench.memory import FilesystemPerTaskMemory, SharedTranscriptMemory
from interleave_codebench.bench.types import MixedEpisode, ScheduleEvent, TaskSpec, TaskState


class MemoryBackendTests(unittest.TestCase):
    def _episode(self) -> tuple[MixedEpisode, dict[str, TaskState]]:
        task_a = TaskSpec(
            task_id="task-a",
            source_benchmark="fixture-a",
            repo_id="repo-a",
            repo_checkout="base",
            task_prompt="Fix task A",
            language="python",
            eval_harness={"type": "shell", "command": "python verify.py task-a"},
            setup_cmds=[],
            success_type="explicit_finish_and_eval",
            metadata={"target_file": "a.py", "search_text": "x", "replace_text": "y"},
        )
        task_b = TaskSpec(
            task_id="task-b",
            source_benchmark="fixture-b",
            repo_id="repo-b",
            repo_checkout="base",
            task_prompt="Fix task B",
            language="python",
            eval_harness={"type": "shell", "command": "python verify.py task-b"},
            setup_cmds=[],
            success_type="explicit_finish_and_eval",
            metadata={"target_file": "b.py", "search_text": "x", "replace_text": "y"},
        )
        episode = MixedEpisode(
            episode_id="dev-0001",
            tasks=[task_a, task_b],
            schedule=[ScheduleEvent(step_idx=0, active_task_id="task-a")],
            seed=7,
            max_total_actions=8,
            max_actions_per_task=4,
            split="dev",
        )
        with tempfile.TemporaryDirectory() as tmp:
            pass
        states = {
            "task-a": TaskState(
                task_id="task-a",
                status="pending",
                workspace_path="/tmp/task-a",
                memory_path="/tmp/.task-memory/task-a",
            ),
            "task-b": TaskState(
                task_id="task-b",
                status="pending",
                workspace_path="/tmp/task-b",
                memory_path="/tmp/.task-memory/task-b",
            ),
        }
        return episode, states

    def test_shared_transcript_includes_inactive_history(self) -> None:
        episode, states = self._episode()
        with tempfile.TemporaryDirectory() as tmp:
            backend = SharedTranscriptMemory(Path(tmp))
            backend.initialize_episode(episode, states)
            backend.append_event("task-b", "assistant", "task-b event")
            prompt = backend.build_prompt("task-a", states)
            self.assertTrue(prompt.contains_inactive_task_history)
            self.assertIn("task-b event", prompt.text)

    def test_filesystem_memory_writes_per_task_files(self) -> None:
        episode, states = self._episode()
        with tempfile.TemporaryDirectory() as tmp:
            backend = FilesystemPerTaskMemory(Path(tmp))
            backend.initialize_episode(episode, states)
            backend.append_event("task-a", "assistant", "task-a event")
            prompt = backend.build_prompt("task-a", states)
            self.assertFalse(prompt.contains_inactive_task_history)
            self.assertIn("task-a event", prompt.text)
            self.assertNotIn("task-b event", prompt.text)
            task_dir = Path(tmp) / episode.episode_id / "task-a"
            self.assertTrue((task_dir / "interaction.md").exists())
            self.assertTrue((task_dir / "summary.md").exists())

