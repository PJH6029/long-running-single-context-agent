from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from interleave_codebench.bench.adapters import SWEBenchVerifiedAdapter, SWECIAdapter
from interleave_codebench.bench.v0_1 import prepare_v0_1_dataset
from interleave_codebench.utils import write_json


class V01BenchmarkPrepTests(unittest.TestCase):
    def test_prepare_v0_1_materializes_snapshots_and_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_path, base_commit, target_commit = self._create_repo_fixture(tmp_path)
            manifest_path = tmp_path / "manifest.json"
            write_json(
                manifest_path,
                {
                    "swe_bench_verified": [
                        {
                            "task_id": "sample-swe-bench",
                            "upstream_task_id": "sample-swe-bench",
                            "repo_id": "local/sample-repo",
                            "repo_url": str(repo_path),
                            "repo_checkout": base_commit,
                            "task_prompt": "Fix the return value in pkg.py.",
                            "split": "dev",
                            "target_file": "pkg.py",
                            "target_symbol": "value",
                            "setup_cmds": [],
                            "verify_spec": {
                                "path": "pkg.py",
                                "must_contain": ["return 2"],
                                "must_not_contain": ["return 1"],
                            },
                        }
                    ],
                    "swe_ci": [
                        {
                            "task_id": "sample-swe-ci",
                            "upstream_task_id": "sample-swe-ci",
                            "repo_id": "local/sample-repo",
                            "repo_url": str(repo_path),
                            "repo_checkout": base_commit,
                            "target_sha": target_commit,
                            "task_prompt": "Update pkg.py so value returns 2.",
                            "split": "eval",
                            "target_file": "pkg.py",
                            "target_symbol": "value",
                            "setup_cmds": [],
                            "verify_spec": {
                                "path": "pkg.py",
                                "must_contain": ["return 2"],
                                "must_not_contain": ["return 1"],
                            },
                        }
                    ],
                },
            )
            output_root = prepare_v0_1_dataset(
                manifest_path=manifest_path,
                output_root=tmp_path / "prepared",
            )

            swe_bench_path = output_root / "swe_bench_verified" / "tasks.json"
            swe_ci_path = output_root / "swe_ci" / "tasks.csv"
            self.assertTrue(swe_bench_path.exists())
            self.assertTrue(swe_ci_path.exists())

            swe_bench_tasks = SWEBenchVerifiedAdapter(swe_bench_path).load_tasks()
            swe_ci_tasks = SWECIAdapter(swe_ci_path).load_tasks()
            self.assertEqual(len(swe_bench_tasks), 1)
            self.assertEqual(len(swe_ci_tasks), 1)
            self.assertEqual(swe_bench_tasks[0].metadata["target_file"], "pkg.py")
            self.assertEqual(swe_ci_tasks[0].metadata["target_symbol"], "value")

            snapshot = Path(swe_bench_tasks[0].repo_source_path or "")
            self.assertTrue((snapshot / "verify.py").exists())
            self.assertTrue((snapshot / ".interleave-task" / "verify_spec.json").exists())

            failed = subprocess.run(
                [sys.executable, "verify.py"],
                cwd=snapshot,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(failed.returncode, 0)

            (snapshot / "pkg.py").write_text("def value():\n    return 2\n")
            passed = subprocess.run(
                [sys.executable, "verify.py"],
                cwd=snapshot,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(passed.returncode, 0)

    def _create_repo_fixture(self, root: Path) -> tuple[Path, str, str]:
        repo = root / "repo"
        repo.mkdir()
        self._git(["init"], cwd=repo)
        self._git(["config", "user.name", "Codex Tests"], cwd=repo)
        self._git(["config", "user.email", "codex@example.com"], cwd=repo)
        (repo / "pkg.py").write_text("def value():\n    return 1\n")
        self._git(["add", "pkg.py"], cwd=repo)
        self._git(["commit", "-m", "base"], cwd=repo)
        base_commit = self._git(["rev-parse", "HEAD"], cwd=repo)
        (repo / "pkg.py").write_text("def value():\n    return 2\n")
        self._git(["commit", "-am", "target"], cwd=repo)
        target_commit = self._git(["rev-parse", "HEAD"], cwd=repo)
        return repo, base_commit, target_commit

    def _git(self, args: list[str], *, cwd: Path) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
