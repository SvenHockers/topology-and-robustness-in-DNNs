from __future__ import annotations

import subprocess
import sys
import unittest


class TestCliEntrypointsSmoke(unittest.TestCase):
    def _run_help(self, module: str) -> str:
        proc = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"{module} --help failed:\n{proc.stdout}")
        return proc.stdout

    def test_optimizers_cli_help(self) -> None:
        out = self._run_help("optimizers.cli")
        self.assertIn("Gaussian-process Bayesian optimiser", out)

    def test_optimisers_cli_batch_help(self) -> None:
        out = self._run_help("optimisers.cli_batch")
        self.assertIn("Batch GP optimiser", out)


if __name__ == "__main__":
    unittest.main()

