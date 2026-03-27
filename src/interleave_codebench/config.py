from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(slots=True)
class PilotConfig:
    swe_bench_dataset: str
    swe_ci_dataset: str
    output_dir: str
    dev_episodes: int = 4
    eval_episodes: int = 4
    seed: int = 7
    max_total_actions: int = 8
    max_actions_per_task: int = 4
    slice_budget: int = 1
    memory_tail_events: int = 6


def load_config(path: str | Path) -> PilotConfig:
    payload = tomllib.loads(Path(path).read_text())
    benchmarks = payload["benchmarks"]
    pilot = payload["pilot"]
    runner = payload["runner"]
    return PilotConfig(
        swe_bench_dataset=benchmarks["swe_bench_verified"]["dataset_path"],
        swe_ci_dataset=benchmarks["swe_ci"]["dataset_path"],
        output_dir=runner["output_dir"],
        dev_episodes=pilot.get("dev_episodes", 4),
        eval_episodes=pilot.get("eval_episodes", 4),
        seed=pilot.get("seed", 7),
        max_total_actions=pilot.get("max_total_actions", 8),
        max_actions_per_task=pilot.get("max_actions_per_task", 4),
        slice_budget=pilot.get("slice_budget", 1),
        memory_tail_events=runner.get("memory_tail_events", 6),
    )

