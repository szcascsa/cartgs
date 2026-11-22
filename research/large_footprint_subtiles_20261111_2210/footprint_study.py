"""只读复用前轮冻结网络、原图轨迹和数值oracle，名称避开上游study。"""
from pathlib import Path
import sys
import argparse
import collections
import statistics

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'render_count_scaling_20261111_2022'
sys.path.insert(0,str(PREVIOUS))
import render_study as previous
from render_study import *
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from background_runtime import Renderer as BaselineRenderer
from subtile_runtime import Renderer,MODES,extension
from balanced_order import order


def verify_baseline():
    manifest=json.loads((HERE/'reports/BASELINE_SHA256.json').read_text())
    assert all(sha(path)==h for path,h in manifest.items())
    frozen=json.loads((PREVIOUS/'background_stage/reports/FROZEN.json').read_text())
    for path,h in frozen['sources'].items():
        assert sha(PREVIOUS/'background_stage'/path)==h,path
    assert sha(frozen['binary'])==frozen['binary_sha256']


def compare(run,cam,packet,bg,reference):
    image,radii=run.render(cam,packet,bg)
    return dict(rgb_bitwise=bits_equal(image,reference['render']),
                radii_bitwise=bits_equal(radii,reference['radii']),
                max_rgb_error=float((image-reference['render']).abs().max()))
