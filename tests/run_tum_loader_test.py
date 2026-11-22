import ast
import glob
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


# run.py initializes CUDA metrics on import; test only the loader and association.
run_path = Path(__file__).resolve().parents[1] / "python" / "run.py"
module = ast.parse(run_path.read_text(encoding="utf-8"), filename=str(run_path))
functions = ast.Module(
    body=[
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("loadTUM", "associate_frames")
    ],
    type_ignores=[],
)
namespace = {"os": os, "glob": glob, "np": np}
exec(compile(functions, str(run_path), "exec"), namespace)
loadTUM = namespace["loadTUM"]
associate_frames = namespace["associate_frames"]


class TUMRGBLoaderTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def image(self, relative_path):
        image_path = self.root / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.touch()
        return str(image_path)

    def index(self, text):
        (self.root / "rgb.txt").write_text(text, encoding="utf-8-sig")

    def test_jpg_frame_numbers_use_index_timestamps_for_association(self):
        expected = [self.image("rgb/{:06d}.jpg".format(i)) for i in range(3)]
        self.image("rgb3/9.0.png")
        self.index(
            "# timestamp filename\n\n"
            "0.000000 rgb/000000.jpg\n"
            "0.033333\trgb/000001.jpg # second frame\n"
            "0.066667 rgb/000002.jpg\n"
        )

        paths, timestamps = loadTUM(str(self.root))

        self.assertEqual(list(map(Path, paths)), list(map(Path, expected)))
        self.assertEqual(timestamps.dtype, np.dtype(np.float64))
        np.testing.assert_allclose(timestamps, [0.0, 0.033333, 0.066667])
        self.assertEqual(
            associate_frames([0.001, 0.034, 0.067], timestamps, max_dt=0.005),
            [(0, 0), (1, 1), (2, 2)],
        )

    def test_timestamp_filenames_support_png_jpg_and_jpeg(self):
        expected = [
            self.image("rgb/1.000000.png"),
            self.image("rgb/2.000000.jpg"),
            self.image("rgb/3.000000.JPEG"),
        ]
        self.image("rgb/notes.txt")

        paths, timestamps = loadTUM(str(self.root))

        self.assertEqual(paths, expected)
        self.assertEqual(timestamps, [1.0, 2.0, 3.0])

    def test_rgb3_keeps_priority_without_index(self):
        self.image("rgb/1.0.png")
        expected = self.image("rgb3/2.0.png")

        paths, timestamps = loadTUM(str(self.root))

        self.assertEqual(paths, [expected])
        self.assertEqual(timestamps, [2.0])

    def test_missing_index_image_reports_path(self):
        self.index("0.0 rgb/missing.jpg\n")
        with self.assertRaisesRegex(FileNotFoundError, "missing.jpg"):
            loadTUM(str(self.root))

    def test_empty_index_does_not_fall_back_to_filename_timestamps(self):
        self.image("rgb/000000.jpg")
        self.index("# no entries\n\n")
        with self.assertRaisesRegex(RuntimeError, "rgb.txt"):
            loadTUM(str(self.root))

    def test_malformed_index_reports_line_number(self):
        self.index("# header\n0.0\n")
        with self.assertRaisesRegex(ValueError, "rgb.txt:2"):
            loadTUM(str(self.root))

    def test_empty_rgb_directory_reports_error_before_association(self):
        (self.root / "rgb").mkdir()
        with self.assertRaisesRegex(RuntimeError, "No RGB images found"):
            loadTUM(str(self.root))


if __name__ == "__main__":
    unittest.main()
