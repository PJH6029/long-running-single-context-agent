from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from interleave_codebench.agents.policies import ToyPromptLeakPolicy
from interleave_codebench.agents.runner import RepoAgentRunner, build_memory_backend
from interleave_codebench.bench.adapters import SWEBenchVerifiedAdapter, SWECIAdapter
from interleave_codebench.bench.mixers import PilotEpisodeBuilder


ROOT = Path(__file__).resolve().parent / "fixtures"


class RunnerTests(unittest.TestCase):
    def test_filesystem_mode_solves_more_than_shared_mode(self) -> None:
        swe = SWEBenchVerifiedAdapter(ROOT / "swe_bench_verified" / "tasks.json").load_tasks(split="dev")
        ci = SWECIAdapter(ROOT / "swe_ci" / "tasks.csv").load_tasks(split="dev")
        episode = PilotEpisodeBuilder(dev_episodes=1, eval_episodes=0, seed=7).build(swe, ci)["dev"][0]
        runner = RepoAgentRunner(ToyPromptLeakPolicy())
        with tempfile.TemporaryDirectory() as tmp:
            shared = runner.run_mixed_episode(
                episode,
                memory_backend=build_memory_backend("shared", Path(tmp) / "memory"),
                output_root=Path(tmp) / "runs",
            )
            filesystem = runner.run_mixed_episode(
                episode,
                memory_backend=build_memory_backend("filesystem", Path(tmp) / "memory"),
                output_root=Path(tmp) / "runs",
            )
        shared_success = sum(1 for item in shared.per_task if item.success)
        filesystem_success = sum(1 for item in filesystem.per_task if item.success)
        self.assertLess(shared_success, filesystem_success)
        self.assertEqual(filesystem_success, 2)
        self.assertGreater(shared.stale_memory_errors, 0)
        self.assertEqual(filesystem.stale_memory_errors, 0)

