from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from interleave_codebench.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_default_agent_config_is_toy(self) -> None:
        config = load_config(ROOT / "configs" / "pilot_fixture.toml")
        self.assertEqual(config.agent.kind, "toy")
        self.assertEqual(config.agent.name, "toy-prompt-leak")
        self.assertEqual(config.agent.command, [])

    def test_external_cli_agent_config_loads_command_and_timeout(self) -> None:
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
                'output_dir = ".runs/test-config"',
                "memory_tail_events = 6",
                "",
                "[agent]",
                'kind = "external_cli_json"',
                'name = "stub cli"',
                "command = [",
                f'  "{sys.executable}",',
                '  "stub.py",',
                "]",
                "timeout_seconds = 12",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(config_text)
            config = load_config(config_path)
        self.assertEqual(config.agent.kind, "external_cli_json")
        self.assertEqual(config.agent.name, "stub cli")
        self.assertEqual(config.agent.slug, "stub-cli")
        self.assertEqual(config.agent.command, [sys.executable, "stub.py"])
        self.assertEqual(config.agent.timeout_seconds, 12)
