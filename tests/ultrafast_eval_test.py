"""Verify evaluation command routing without requiring CUDA or metric packages."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("eval_lod", ROOT / "python/eval_lod.py")
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


class UltrafastEvaluationTest(unittest.TestCase):
    def verify_commands(self, mode):
        ultrafast = mode is None or mode == 1
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "result"
            checkpoint = result / "12_shutdown" / "ply"
            checkpoint.mkdir(parents=True)
            (result / "999_shutdown").mkdir()  # Must not accidentally evaluate this Full model.
            selector = checkpoint / "selector.pt"
            selector.touch()
            config = checkpoint / "selector_config.yaml"
            config.write_text(
                "%YAML:1.0\n" + ("Training.ultra_light_mode: {}\n".format(mode) if mode is not None else "") +
                "Selector.target_ratios: [0.01, 0.05, 0.10, 0.15]\n",
                encoding="utf-8",
            )
            output = Path(temporary) / "evaluation"
            commands = []

            def run(command, check):
                self.assertTrue(check)
                commands.append(command)
                folder = Path(command[command.index("--output_path") + 1])
                (folder / "eval.txt").write_text("PSNR: 1\nSSIM: 2\nLPIPS: 3\n", encoding="utf-8")
                (folder / "selector_stats.json").write_text(
                    json.dumps({"active_gaussians": 4, "actual_vs_target": 1}), encoding="utf-8")

            argv = ["eval_lod.py", str(result), str(Path(temporary) / "gt"),
                    "--selector", str(selector), "--output", str(output), "--correct-scale"]
            with patch("sys.argv", argv), patch.object(EVAL.subprocess, "run", side_effect=run):
                EVAL.main()
            self.assertEqual(len(commands), 5 if ultrafast else 4)
            for command, ratio in zip(commands[:4], [0.01, 0.05, 0.10, 0.15]):
                self.assertEqual(float(command[command.index("--selector-ratio") + 1]), ratio)
                self.assertEqual(command[command.index("--selector") + 1], str(selector))
                self.assertIn("--correct_scale", command)
            if ultrafast:
                full = commands[-1]
                self.assertNotIn("--selector", full)
                self.assertNotIn("--selector-ratio", full)
                self.assertEqual(full[full.index("--shutdown-model") + 1], "12_shutdown")
                summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary[-1]["phase"], "FULL")
                self.assertEqual(summary[-1]["active_gaussians"], 4)
                self.assertEqual(summary[-1]["PSNR"], 1)

    def test_ultrafast_evaluates_four_ratios_and_matching_full(self):
        self.verify_commands(1)

    def test_other_modes_keep_four_ratios(self):
        self.verify_commands(0)
        self.verify_commands(2)

    def test_missing_mode_defaults_to_mode1(self):
        self.verify_commands(None)


if __name__ == "__main__":
    unittest.main()
