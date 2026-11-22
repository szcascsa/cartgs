from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = Path('/home/prmi/zwj/experiments/view_conditioned_li_20261106/runs/full_both_li/23560_shutdown/ply')
DATASET = Path('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz')
VANILLA = Path('/home/prmi/zwj/realLOD/saved_experiments/vanilla_all_datasets_20261031_2011/jobs/tum/rgbd/fr1_xyz/repeat1/seed0/attempt_2/model/10400_shutdown/ply')
V21_BINARY = ROOT / 'build/mycartgs_deployment_fp32_v21/mycartgs_deployment_fp32_v21.so'
FROZEN_BINARY = ROOT / 'build/mycartgs_frozen_inference_v3/mycartgs_frozen_inference_v3.so'
OLD_EVAL = ROOT / 'source/evaluation'
os.environ['TORCH_EXTENSIONS_DIR'] = str(ROOT / 'build')
os.environ['CUDA_HOME'] = '/usr/local/cuda-11.8'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
os.environ['MAX_JOBS'] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PATH'] = '/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:' + os.environ.get('PATH', '')
sys.path[:0] = [str(ROOT / 'source/python'), str(OLD_EVAL), str(OLD_EVAL / 'diff-gaussian-rasterization')]

import torch
import numpy as np
from budget_viewer import BudgetScene, load_views
from renderer import render_prepared, prepare_render_data
from utils import MiniCam, focal2fov
from gaussian_model import GaussianModel

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('Cannot write an empty CSV')
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def camera_from_view(calibration, view):
    return MiniCam(
        calibration['width'], calibration['height'],
        focal2fov(calibration['fx'], calibration['width']),
        focal2fov(calibration['fy'], calibration['height']),
        np.linalg.inv(np.asarray(view['pose'])),
    )


def gpu_state(label):
    return {
        'label': label,
        'unix': time.time(),
        'gpu': subprocess.check_output([
            'nvidia-smi', '--query-gpu=name,uuid,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw',
            '--format=csv,noheader',
        ], text=True),
        'processes': subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader',
        ], text=True),
    }


def identity():
    pid = os.getpid()
    return dict(
        pid=pid,
        start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    )
