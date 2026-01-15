from __future__ import annotations

import subprocess
import sys
import unittest


class TestCliEntrypointsSmoke(unittest.TestCase):
    def _run_help(self, module: str, *extra_args: str) -> str:
        proc = subprocess.run(
            [sys.executable, "-m", module, *extra_args, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        msg = f"{module} {' '.join(extra_args)} --help failed:\n{proc.stdout}"
        self.assertEqual(proc.returncode, 0, msg=msg)
        return proc.stdout

    def test_optimisers_root_help(self) -> None:
        out = self._run_help("optimisers")
        self.assertIn("Single command entrypoint", out)

    def test_optimisers_batch_help(self) -> None:
        out = self._run_help("optimisers", "batch")
        self.assertIn("Batch GP optimiser", out)

    def test_optimisers_plot_history_help(self) -> None:
        out = self._run_help("optimisers", "plot-history")
        self.assertIn("Plot optimiser parameter-space visualisations", out)

    def test_optimisers_plot_history_module_help(self) -> None:
        out = self._run_help("optimisers.plot_history")
        self.assertIn("Plot optimiser parameter-space visualisations", out)


if __name__ == "__main__":
    unittest.main()

