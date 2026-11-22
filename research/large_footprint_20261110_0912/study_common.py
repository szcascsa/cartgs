"""只读复用既有冻结模型，模块名避开上游common，保持原输入准备。"""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'sparse_rasterizer_sixhour_20261110_0053'
sys.path.insert(0,str(PREVIOUS))
from harness import *
from final_benchmark import native_modules,rotate,CsvSink,digest
from final_edges import packet as synthetic_packet
from graph_runtime import GraphRenderer as PreviousGraphRenderer
from runtime import SparseRenderer as PreviousRegularRenderer
from camera_async_preparation import AsyncGraphCameraPreparer

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'sparse_rasterizer_sixhour_20261110_0053'
sys.path.insert(0,str(HERE))
BASELINE_CONFIG=json.loads((PREVIOUS/'reports/FROZEN_CANDIDATE.json').read_text())


def previous_renderer():
    native_modules(BASELINE_CONFIG)
    return PreviousGraphRenderer(**BASELINE_CONFIG['graph'])


def save_json(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2))
