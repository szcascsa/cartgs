"""固定评价seed；沿用全轨迹评价和指标导出，不保存图片。"""
from pathlib import Path
import runpy
import sys
import torch

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
sys.argv[0] = str(Path(__file__).resolve().parent / 'lod_eval_entry.py')
runpy.run_path(sys.argv[0], run_name='__main__')
