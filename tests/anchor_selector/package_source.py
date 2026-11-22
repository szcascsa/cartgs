"""打包隔离构建所需的一方源码；不读取模型、缓存或第三方构建目录。"""
import argparse
from pathlib import Path
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[2]
args.output.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(args.output, "w:gz") as archive:
    for directory in ("include", "src", "improvements", "python", "cfg", "cuda_rasterizer", "viewer", "examples"):
        for path in sorted((repo / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.add(path, arcname=path.relative_to(repo))
    archive.add(repo / "CMakeLists.txt", arcname="CMakeLists.txt")
    for path in sorted((repo / "tests").glob("*")):
        if path.is_file():
            archive.add(path, arcname=path.relative_to(repo))
    archive.add(repo / "tests/anchor_selector", arcname="tests/anchor_selector")
    for directory in ("cuda_tf_implementation_20261013", "cuda_selector_revised_20261015"):
        path = repo / "tests/diagnostics" / directory / "online_driver.cmake"
        archive.add(path, arcname=path.relative_to(repo))
print(args.output)
