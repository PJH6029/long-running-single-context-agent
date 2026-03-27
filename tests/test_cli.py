from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from interleave_codebench.cli import build_episodes, run_compare


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tests" / "fixtures" / "tools"


class CLITests(unittest.TestCase):
    def test_build_and_compare_outputs_exist(self) -> None:
        config_template = (ROOT / "configs" / "pilot_fixture.toml").read_text()
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "pilot.toml"
            config_path.write_text(
                config_template.replace('.runs/pilot_fixture', str((Path(tmp) / "runs").resolve()))
            )
            build_path = build_episodes(str(config_path))
            compare_path = run_compare(str(config_path), "dev")
            self.assertTrue(build_path.exists())
            self.assertTrue(compare_path.exists())
            self.assertEqual(compare_path.parent.name, "toy-prompt-leak")
            self.assertTrue((compare_path.parent / "summary-dev.json").exists())

    def test_external_agent_outputs_are_scoped_by_agent_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_text = "\n".join(
                [
                    '[benchmarks.swe_bench_verified]',
                    'dataset_path = "tests/fixtures/swe_bench_verified/tasks.json"',
                    "",
                    '[benchmarks.swe_ci]',
                    'dataset_path = "tests/fixtures/swe_ci/tasks.csv"',
                    "",
                    "[pilot]",
                    "dev_episodes = 1",
                    "eval_episodes = 0",
                    "seed = 7",
                    "max_total_actions = 8",
                    "max_actions_per_task = 4",
                    "slice_budget = 1",
                    "",
                    "[runner]",
                    f'output_dir = "{(Path(tmp) / "runs").resolve()}"',
                    "memory_tail_events = 6",
                    "",
                    "[agent]",
                    'kind = "external_cli_json"',
                    'name = "stub cli"',
                    "command = [",
                    f'  "{sys.executable}",',
                    f'  "{TOOLS / "external_cli_stub.py"}",',
                    '  "--mode",',
                    '  "toy",',
                    '  "--output-schema",',
                    '  "{schema_path}",',
                    '  "-o",',
                    '  "{response_path}",',
                    '  "-C",',
                    '  "{workspace_path}",',
                    '  "-",',
                    "]",
                    "timeout_seconds = 30",
                ]
            )
            config_path = Path(tmp) / "pilot_external.toml"
            config_path.write_text(config_text)
            build_path = build_episodes(str(config_path))
            compare_path = run_compare(str(config_path), "dev")
            self.assertTrue(build_path.exists())
            self.assertTrue(compare_path.exists())
            self.assertEqual(compare_path.parent.name, "stub-cli")
            self.assertTrue((compare_path.parent / "summary-dev.json").exists())
