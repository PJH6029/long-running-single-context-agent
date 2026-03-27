from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from interleave_codebench.cli import build_episodes, run_compare


ROOT = Path(__file__).resolve().parents[1]


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

