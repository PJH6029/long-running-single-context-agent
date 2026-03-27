from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TaskSpec:
    task_id: str
    source_benchmark: str
    repo_id: str
    repo_checkout: str
    task_prompt: str
    language: str
    eval_harness: dict[str, Any]
    setup_cmds: list[str]
    success_type: str
    split: str = "eval"
    repo_source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskSpec":
        return cls(**payload)


@dataclass(slots=True)
class TaskState:
    task_id: str
    status: str
    workspace_path: str
    memory_path: str
    action_count: int = 0
    last_result: str = ""
    done: bool = False
    score: float = 0.0
    eval_passed: bool = False
    resume_count: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def workspace(self) -> Path:
        return Path(self.workspace_path)

    @property
    def memory_dir(self) -> Path:
        return Path(self.memory_path)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ScheduleEvent:
    step_idx: int
    active_task_id: str
    slice_budget: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MixedEpisode:
    episode_id: str
    tasks: list[TaskSpec]
    schedule: list[ScheduleEvent]
    seed: int
    max_total_actions: int
    max_actions_per_task: int
    split: str = "eval"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tasks"] = [task.to_dict() for task in self.tasks]
        payload["schedule"] = [event.to_dict() for event in self.schedule]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MixedEpisode":
        tasks = [TaskSpec.from_dict(task) for task in payload["tasks"]]
        schedule = [ScheduleEvent(**event) for event in payload["schedule"]]
        return cls(
            episode_id=payload["episode_id"],
            tasks=tasks,
            schedule=schedule,
            seed=payload["seed"],
            max_total_actions=payload["max_total_actions"],
            max_actions_per_task=payload["max_actions_per_task"],
            split=payload.get("split", "eval"),
        )


@dataclass(slots=True)
class PromptBundle:
    text: str
    estimated_tokens: int
    contains_inactive_task_history: bool
    active_task_id: str
    registry_snapshot: list[dict[str, Any]]


@dataclass(slots=True)
class TaskEvaluation:
    task_id: str
    eval_passed: bool
    finished: bool
    success: bool
    score: float
    details: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EpisodeMetrics:
    episode_id: str
    memory_mode: str
    per_task: list[TaskEvaluation]
    total_slices: int
    per_task_slices: dict[str, int]
    prompt_tokens_per_slice: list[int]
    cumulative_prompt_tokens: int
    duplicate_work_rate: float
    stale_memory_errors: int
    unfinished_task_abandonment_rate: float
    both_tasks_solved: bool
    one_task_solved: bool
    zero_tasks_solved: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["per_task"] = [item.to_dict() for item in self.per_task]
        return payload

