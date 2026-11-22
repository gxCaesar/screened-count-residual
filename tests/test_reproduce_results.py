"""End-to-end verification of archived confirmatory tables and CPU tile calibration example."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_NO_PYC_ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")


class TestConfirmatoryTableReproduction(unittest.TestCase):
    def test_reproduce_script_exits_zero(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts" / "reproduce_journal_results.py")],
            cwd=str(ROOT),
            env=_NO_PYC_ENV,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
        self.assertIn("Verified 20 frozen generator instances", proc.stdout)

    def test_example_tile_calibration_exits_zero(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts" / "example_tile_calibration.py")],
            cwd=str(ROOT),
            env=_NO_PYC_ENV,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
        self.assertIn("Example tile calibration passed", proc.stdout)

    def test_campaign_runners_import_and_help(self) -> None:
        for mod in (
            "s2.run_hemit_multibackbone_cell",
            "s2.run_orion_multibackbone_survival",
            "s2.run_her2match_dab_campaign",
        ):
            proc = subprocess.run(
                [sys.executable, "-B", "-m", mod, "--help"],
                cwd=str(ROOT),
                env=_NO_PYC_ENV,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=f"{mod}: {proc.stderr}{proc.stdout}")


if __name__ == "__main__":
    unittest.main()
