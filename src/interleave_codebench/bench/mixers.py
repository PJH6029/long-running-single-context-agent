from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

from ..utils import ensure_directory, write_json
from .scheduler import build_round_robin_schedule
from .types import MixedEpisode, TaskSpec


class PilotEpisodeBuilder:
    def __init__(
        self,
        *,
        dev_episodes: int = 4,
        eval_episodes: int = 4,
        seed: int = 7,
        max_total_actions: int = 8,
        max_actions_per_task: int = 4,
        slice_budget: int = 1,
    ):
        self.dev_episodes = dev_episodes
        self.eval_episodes = eval_episodes
        self.seed = seed
        self.max_total_actions = max_total_actions
        self.max_actions_per_task = max_actions_per_task
        self.slice_budget = slice_budget

    def build(
        self, swe_bench_tasks: Iterable[TaskSpec], swe_ci_tasks: Iterable[TaskSpec]
    ) -> dict[str, list[MixedEpisode]]:
        rng = random.Random(self.seed)
        payload = {"dev": [], "eval": []}
        for split, count in (("dev", self.dev_episodes), ("eval", self.eval_episodes)):
            swe_bench = [task for task in swe_bench_tasks if task.split == split]
            swe_ci = [task for task in swe_ci_tasks if task.split == split]
            swe_bench.sort(key=lambda item: item.task_id)
            swe_ci.sort(key=lambda item: item.task_id)
            rng.shuffle(swe_bench)
            rng.shuffle(swe_ci)
            if len(swe_bench) < count or len(swe_ci) < count:
                raise ValueError(
                    f"Not enough {split} tasks to build the requested pilot split. "
                    f"Need {count} from each source but received {len(swe_bench)} and {len(swe_ci)}."
                )
            for index in range(count):
                swe_task = swe_bench[index]
                ci_task = swe_ci[index]
                episode_id = f"{split}-{index + 1:04d}"
                schedule = build_round_robin_schedule(
                    [swe_task.task_id, ci_task.task_id],
                    max_total_actions=self.max_total_actions,
                    max_actions_per_task=self.max_actions_per_task,
                    slice_budget=self.slice_budget,
                )
                payload[split].append(
                    MixedEpisode(
                        episode_id=episode_id,
                        tasks=[swe_task, ci_task],
                        schedule=schedule,
                        seed=self.seed,
                        max_total_actions=self.max_total_actions,
                        max_actions_per_task=self.max_actions_per_task,
                        split=split,
                    )
                )
        return payload

    def write(self, output_dir: str | Path, episodes: dict[str, list[MixedEpisode]]) -> Path:
        output_dir = ensure_directory(Path(output_dir))
        write_json(
            output_dir / "mixed_episodes.json",
            {split: [episode.to_dict() for episode in items] for split, items in episodes.items()},
        )
        return output_dir / "mixed_episodes.json"
