from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from interleave_codebench.agents.policies import ExternalCLIJsonPolicy, ToyPromptLeakPolicy
from interleave_codebench.agents.runner import RepoAgentRunner, build_memory_backend
from interleave_codebench.bench.adapters import SWEBenchVerifiedAdapter, SWECIAdapter
from interleave_codebench.bench.mixers import PilotEpisodeBuilder


ROOT = Path(__file__).resolve().parent / "fixtures"
TOOLS = ROOT / "tools"


class RunnerTests(unittest.TestCase):
    def _dev_episode(self):
        swe = SWEBenchVerifiedAdapter(ROOT / "swe_bench_verified" / "tasks.json").load_tasks(split="dev")
        ci = SWECIAdapter(ROOT / "swe_ci" / "tasks.csv").load_tasks(split="dev")
        return PilotEpisodeBuilder(dev_episodes=1, eval_episodes=0, seed=7).build(swe, ci)["dev"][0]

    def _single_dev_task(self):
        return SWEBenchVerifiedAdapter(ROOT / "swe_bench_verified" / "tasks.json").load_tasks(split="dev")[0]

    def _stub_command(self, mode: str) -> list[str]:
        return [
            sys.executable,
            str(TOOLS / "external_cli_stub.py"),
            "--mode",
            mode,
            "--output-schema",
            "{schema_path}",
            "-o",
            "{response_path}",
            "-C",
            "{workspace_path}",
            "-",
        ]

    def test_filesystem_mode_solves_more_than_shared_mode(self) -> None:
        episode = self._dev_episode()
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

    def test_python_prefix_is_rewritten_to_active_interpreter(self) -> None:
        runner = RepoAgentRunner(ToyPromptLeakPolicy())
        with tempfile.TemporaryDirectory() as tmp:
            details = runner._run_command(Path(tmp), 'python -c "import sys; print(sys.executable)"')
        self.assertIn("returncode=0", details)
        self.assertIn(sys.executable, details)

    def test_external_cli_policy_matches_toy_signal(self) -> None:
        episode = self._dev_episode()
        runner = RepoAgentRunner(
            ExternalCLIJsonPolicy(
                name="stub-cli",
                command_template=self._stub_command("toy"),
                timeout_seconds=30,
            )
        )
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
        self.assertEqual(shared.policy_error_count, 0)
        self.assertEqual(filesystem.policy_error_count, 0)

    def test_external_cli_failures_are_counted_and_stop_the_task(self) -> None:
        task = self._single_dev_task()
        for mode, timeout in (
            ("invalid-json", 30),
            ("missing-field", 30),
            ("exit-1", 30),
            ("sleep", 1),
        ):
            with self.subTest(mode=mode):
                runner = RepoAgentRunner(
                    ExternalCLIJsonPolicy(
                        name=f"stub-{mode}",
                        command_template=self._stub_command(mode),
                        timeout_seconds=timeout,
                    )
                )
                with tempfile.TemporaryDirectory() as tmp:
                    metrics = runner.run_single_task(
                        task,
                        memory_backend=build_memory_backend("filesystem", Path(tmp) / "memory"),
                        output_root=Path(tmp) / "runs",
                        max_actions=4,
                    )
                self.assertEqual(metrics.policy_error_count, 1)
                self.assertEqual(metrics.total_slices, 1)
                self.assertFalse(metrics.per_task[0].success)
                self.assertFalse(metrics.per_task[0].finished)
