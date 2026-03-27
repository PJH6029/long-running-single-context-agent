from __future__ import annotations

import unittest
from pathlib import Path

from interleave_codebench.bench.adapters import SWEBenchVerifiedAdapter, SWECIAdapter
from interleave_codebench.bench.mixers import PilotEpisodeBuilder
from interleave_codebench.bench.scheduler import build_round_robin_schedule


ROOT = Path(__file__).resolve().parent / "fixtures"


class MixerSchedulerTests(unittest.TestCase):
    def test_round_robin_alternates_and_honors_budgets(self) -> None:
        schedule = build_round_robin_schedule(
            ["task-a", "task-b"], max_total_actions=8, max_actions_per_task=4, slice_budget=1
        )
        self.assertEqual([item.active_task_id for item in schedule], ["task-a", "task-b"] * 4)

    def test_builder_is_deterministic(self) -> None:
        swe = SWEBenchVerifiedAdapter(ROOT / "swe_bench_verified" / "tasks.json").load_tasks()
        ci = SWECIAdapter(ROOT / "swe_ci" / "tasks.csv").load_tasks()
        builder = PilotEpisodeBuilder(seed=7, dev_episodes=4, eval_episodes=4)
        first = builder.build(swe, ci)
        second = builder.build(swe, ci)
        self.assertEqual(
            [episode.episode_id for episode in first["dev"]],
            [episode.episode_id for episode in second["dev"]],
        )
        self.assertEqual(
            [episode.tasks[0].task_id for episode in first["eval"]],
            [episode.tasks[0].task_id for episode in second["eval"]],
        )

