from __future__ import annotations

import unittest
from pathlib import Path

from interleave_codebench.bench.adapters import SWEBenchVerifiedAdapter, SWECIAdapter


ROOT = Path(__file__).resolve().parent / "fixtures"


class AdapterTests(unittest.TestCase):
    def test_swe_bench_verified_adapter_normalizes_tasks(self) -> None:
        tasks = SWEBenchVerifiedAdapter(ROOT / "swe_bench_verified" / "tasks.json").load_tasks()
        self.assertEqual(len(tasks), 8)
        task = tasks[0]
        self.assertEqual(task.source_benchmark, "swe_bench_verified")
        self.assertEqual(task.language, "python")
        self.assertEqual(task.eval_harness["type"], "shell")
        self.assertTrue(task.repo_source_path.endswith("python_bug_repo"))

    def test_swe_ci_adapter_normalizes_tasks(self) -> None:
        tasks = SWECIAdapter(ROOT / "swe_ci" / "tasks.csv").load_tasks()
        self.assertEqual(len(tasks), 8)
        task = tasks[0]
        self.assertEqual(task.source_benchmark, "swe_ci")
        self.assertEqual(task.language, "python")
        self.assertEqual(task.eval_harness["type"], "shell")
        self.assertTrue(task.repo_source_path.endswith("python_ci_repo"))

