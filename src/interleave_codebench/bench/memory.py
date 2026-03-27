from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Any

from ..utils import ensure_directory, estimate_tokens, write_json
from .types import MixedEpisode, PromptBundle, TaskSpec, TaskState


GLOBAL_MULTITASK_INSTRUCTION = """You are a repo coding agent operating under forced task switching.
Treat the prompt as working memory only.
Stay focused on the active task.
One slice equals one action.
Return only the next action decision for the active task."""


class MemoryBackend(abc.ABC):
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.episode_dir: Path | None = None
        self.task_specs: dict[str, TaskSpec] = {}

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def initialize_episode(self, episode: MixedEpisode, task_states: dict[str, TaskState]) -> None:
        self.episode_dir = ensure_directory(self.root_dir / episode.episode_id)
        self.task_specs = {task.task_id: task for task in episode.tasks}
        self._initialize_storage(episode, task_states)

    @abc.abstractmethod
    def _initialize_storage(self, episode: MixedEpisode, task_states: dict[str, TaskState]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def append_event(self, task_id: str, role: str, content: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def build_prompt(self, active_task_id: str, task_states: dict[str, TaskState]) -> PromptBundle:
        raise NotImplementedError

    @abc.abstractmethod
    def sync_task_state(self, task_state: TaskState, task_states: dict[str, TaskState]) -> None:
        raise NotImplementedError

    def _registry_snapshot(self, task_states: dict[str, TaskState]) -> list[dict[str, Any]]:
        snapshot = []
        for task_id, state in sorted(task_states.items()):
            spec = self.task_specs[task_id]
            snapshot.append(
                {
                    "task_id": task_id,
                    "source_benchmark": spec.source_benchmark,
                    "status": state.status,
                    "action_count": state.action_count,
                    "done": state.done,
                    "score": state.score,
                }
            )
        return snapshot

    def _render_registry(self, task_states: dict[str, TaskState]) -> str:
        lines = ["[TASK REGISTRY]"]
        for item in self._registry_snapshot(task_states):
            lines.append(
                "- {task_id} | source={source_benchmark} | status={status} | actions={action_count} | done={done}".format(
                    **item
                )
            )
        return "\n".join(lines)

    def _render_task_spec(self, task_spec: TaskSpec) -> str:
        metadata = task_spec.metadata
        lines = [
            f"[ACTIVE TASK] {task_spec.task_id}",
            f"SOURCE_BENCHMARK={task_spec.source_benchmark}",
            f"REPO_ID={task_spec.repo_id}",
            f"REPO_CHECKOUT={task_spec.repo_checkout}",
            f"TASK_PROMPT={task_spec.task_prompt}",
            f"EVAL_COMMAND={task_spec.eval_harness.get('command', '')}",
        ]
        for key in ("target_file", "search_text", "replace_text", "target_symbol", "desired_value"):
            if key in metadata:
                lines.append(f"{key.upper()}={metadata[key]}")
        return "\n".join(lines)


class SharedTranscriptMemory(MemoryBackend):
    def __init__(self, root_dir: str | Path):
        super().__init__(root_dir)
        self.transcript: list[dict[str, str]] = []

    @property
    def name(self) -> str:
        return "shared_transcript"

    def _initialize_storage(self, episode: MixedEpisode, task_states: dict[str, TaskState]) -> None:
        self.transcript = []
        for task in episode.tasks:
            self.transcript.append(
                {
                    "task_id": task.task_id,
                    "role": "task_context",
                    "content": self._render_task_spec(task),
                }
            )
        write_json(self.episode_dir / "shared_transcript.json", self.transcript)

    def append_event(self, task_id: str, role: str, content: str) -> None:
        self.transcript.append({"task_id": task_id, "role": role, "content": content})
        write_json(self.episode_dir / "shared_transcript.json", self.transcript)

    def build_prompt(self, active_task_id: str, task_states: dict[str, TaskState]) -> PromptBundle:
        active_spec = self.task_specs[active_task_id]
        contains_inactive = any(event["task_id"] != active_task_id for event in self.transcript)
        transcript_text = "\n\n".join(
            f"[{event['task_id']}::{event['role']}]\n{event['content']}" for event in self.transcript
        )
        text = "\n\n".join(
            [
                GLOBAL_MULTITASK_INSTRUCTION,
                f"INACTIVE_TASK_HISTORY_PRESENT={'true' if contains_inactive else 'false'}",
                self._render_registry(task_states),
                self._render_task_spec(active_spec),
                "[GLOBAL TRANSCRIPT]",
                transcript_text,
            ]
        )
        return PromptBundle(
            text=text,
            estimated_tokens=estimate_tokens(text),
            contains_inactive_task_history=contains_inactive,
            active_task_id=active_task_id,
            registry_snapshot=self._registry_snapshot(task_states),
        )

    def sync_task_state(self, task_state: TaskState, task_states: dict[str, TaskState]) -> None:
        write_json(self.episode_dir / "task_states.json", [state.to_dict() for state in task_states.values()])


class FilesystemPerTaskMemory(MemoryBackend):
    def __init__(self, root_dir: str | Path, *, tail_events: int = 6):
        super().__init__(root_dir)
        self.tail_events = tail_events
        self.task_events: dict[str, list[dict[str, str]]] = {}

    @property
    def name(self) -> str:
        return "filesystem_per_task"

    def _initialize_storage(self, episode: MixedEpisode, task_states: dict[str, TaskState]) -> None:
        self.task_events = {}
        registry = {"episode_id": episode.episode_id, "tasks": self._registry_snapshot(task_states)}
        for task in episode.tasks:
            task_dir = ensure_directory(self.episode_dir / task.task_id)
            self.task_events[task.task_id] = []
            (task_dir / "spec.md").write_text(self._render_task_spec(task) + "\n")
            (task_dir / "interaction.md").write_text("")
            (task_dir / "summary.md").write_text("No interaction yet.\n")
            write_json(task_dir / "state.json", task_states[task.task_id].to_dict())
        write_json(self.episode_dir / "registry.json", registry)

    def append_event(self, task_id: str, role: str, content: str) -> None:
        task_dir = self.episode_dir / task_id
        self.task_events[task_id].append({"role": role, "content": content})
        with (task_dir / "interaction.md").open("a") as handle:
            handle.write(f"[{role}]\n{content}\n\n")
        summary = self._build_summary(task_id)
        (task_dir / "summary.md").write_text(summary + "\n")

    def build_prompt(self, active_task_id: str, task_states: dict[str, TaskState]) -> PromptBundle:
        active_spec = self.task_specs[active_task_id]
        task_dir = self.episode_dir / active_task_id
        summary = (task_dir / "summary.md").read_text().strip()
        interaction_tail = self._render_tail(active_task_id)
        text = "\n\n".join(
            [
                GLOBAL_MULTITASK_INSTRUCTION,
                "INACTIVE_TASK_HISTORY_PRESENT=false",
                self._render_registry(task_states),
                self._render_task_spec(active_spec),
                "[ACTIVE TASK SUMMARY]",
                summary,
                "[ACTIVE TASK INTERACTION TAIL]",
                interaction_tail,
            ]
        )
        return PromptBundle(
            text=text,
            estimated_tokens=estimate_tokens(text),
            contains_inactive_task_history=False,
            active_task_id=active_task_id,
            registry_snapshot=self._registry_snapshot(task_states),
        )

    def sync_task_state(self, task_state: TaskState, task_states: dict[str, TaskState]) -> None:
        task_dir = self.episode_dir / task_state.task_id
        write_json(task_dir / "state.json", task_state.to_dict())
        write_json(
            self.episode_dir / "registry.json",
            {"tasks": self._registry_snapshot(task_states)},
        )

    def _render_tail(self, task_id: str) -> str:
        events = self.task_events.get(task_id, [])[-self.tail_events :]
        if not events:
            return "No task-local interaction yet."
        return "\n\n".join(f"[{event['role']}]\n{event['content']}" for event in events)

    def _build_summary(self, task_id: str) -> str:
        events = self.task_events.get(task_id, [])
        if not events:
            return "No interaction yet."
        tail = events[-3:]
        fragments = [f"{event['role']}: {event['content'].splitlines()[0]}" for event in tail]
        return " | ".join(fragments)
