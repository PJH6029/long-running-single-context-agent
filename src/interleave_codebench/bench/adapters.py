from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

from .types import TaskSpec


class BenchmarkAdapter(ABC):
    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)

    @abstractmethod
    def load_tasks(self, *, split: str | None = None, limit: int | None = None) -> list[TaskSpec]:
        raise NotImplementedError

    def _apply_split_and_limit(
        self, tasks: Iterable[TaskSpec], *, split: str | None, limit: int | None
    ) -> list[TaskSpec]:
        filtered = [task for task in tasks if split is None or task.split == split]
        filtered.sort(key=lambda item: item.task_id)
        if limit is not None:
            filtered = filtered[:limit]
        return filtered

    def _resolve_path(self, raw_path: str | None) -> str | None:
        if not raw_path:
            return None
        path = Path(raw_path)
        if path.is_absolute():
            return str(path)
        return str((self.dataset_path.parent / path).resolve())


class SWEBenchVerifiedAdapter(BenchmarkAdapter):
    """Adapter for official or fixture-style SWE-bench Verified task dumps."""

    def load_tasks(self, *, split: str | None = None, limit: int | None = None) -> list[TaskSpec]:
        records = _load_json_records(self.dataset_path)
        tasks: list[TaskSpec] = []
        for record in records:
            if not _is_python_record(record):
                continue
            if not record.get("pilot_ready", True):
                continue
            eval_harness = _normalize_harness(record)
            task = TaskSpec(
                task_id=record.get("task_id", record["instance_id"]),
                source_benchmark="swe_bench_verified",
                repo_id=record["repo"],
                repo_checkout=record.get("repo_checkout", record.get("base_commit", "")),
                task_prompt=record.get("task_prompt", record.get("problem_statement", "")),
                language=record.get("language", "python"),
                eval_harness=eval_harness,
                setup_cmds=_ensure_list(record.get("setup_cmds")),
                success_type=record.get("success_type", "explicit_finish_and_eval"),
                split=record.get("split", "eval"),
                repo_source_path=self._resolve_path(record.get("repo_source_path")),
                metadata=_filtered_metadata(
                    record,
                    drop_keys={
                        "task_id",
                        "instance_id",
                        "repo",
                        "repo_checkout",
                        "base_commit",
                        "problem_statement",
                        "task_prompt",
                        "language",
                        "eval_harness",
                        "eval_command",
                        "setup_cmds",
                        "success_type",
                        "split",
                        "repo_source_path",
                    },
                ),
            )
            tasks.append(task)
        return self._apply_split_and_limit(tasks, split=split, limit=limit)


class SWECIAdapter(BenchmarkAdapter):
    """Adapter for SWE-CI metadata CSV augmented with prompt/harness columns."""

    def load_tasks(self, *, split: str | None = None, limit: int | None = None) -> list[TaskSpec]:
        with self.dataset_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            tasks: list[TaskSpec] = []
            for record in reader:
                if not _is_python_record(record):
                    continue
                if record.get("pilot_ready", "true").lower() not in {"1", "true", "yes"}:
                    continue
                prompt = record.get("task_prompt") or (
                    "Use the CI gap to evolve the repository from "
                    f"{record.get('current_sha', 'current')} toward {record.get('target_sha', 'target')}."
                )
                eval_harness = _normalize_harness(record)
                task = TaskSpec(
                    task_id=record["task_id"],
                    source_benchmark="swe_ci",
                    repo_id=record["repo_name"],
                    repo_checkout=record.get("current_sha", ""),
                    task_prompt=prompt,
                    language=record.get("language", "python"),
                    eval_harness=eval_harness,
                    setup_cmds=_ensure_list(record.get("setup_cmds")),
                    success_type=record.get("success_type", "explicit_finish_and_eval"),
                    split=record.get("split", "eval"),
                    repo_source_path=self._resolve_path(record.get("repo_source_path")),
                    metadata=_filtered_metadata(
                        record,
                        drop_keys={
                            "task_id",
                            "repo_name",
                            "current_sha",
                            "target_sha",
                            "task_prompt",
                            "language",
                            "eval_harness",
                            "eval_command",
                            "setup_cmds",
                            "success_type",
                            "split",
                            "repo_source_path",
                        },
                    ),
                )
                tasks.append(task)
        return self._apply_split_and_limit(tasks, split=split, limit=limit)


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        if "tasks" in payload:
            return list(payload["tasks"])
        return [payload]
    return list(payload)


def _normalize_harness(record: dict[str, Any]) -> dict[str, Any]:
    if "eval_harness" in record and isinstance(record["eval_harness"], dict):
        return record["eval_harness"]
    command = record.get("eval_command")
    if command is None and "eval_harness" in record and isinstance(record["eval_harness"], str):
        command = record["eval_harness"]
    if command is None:
        raise ValueError(f"Record is missing eval harness information: {record}")
    return {"type": "shell", "command": command}


def _ensure_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            return [str(item) for item in json.loads(stripped)]
        return [item.strip() for item in stripped.split("&&") if item.strip()]
    return [str(value)]


def _is_python_record(record: dict[str, Any]) -> bool:
    language = str(record.get("language", "python")).lower()
    return language == "python"


def _filtered_metadata(record: dict[str, Any], *, drop_keys: set[str]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in drop_keys}

