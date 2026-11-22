"""复用冻结输入和网络，独立研究Rasterizer，不改任何选择或补偿输出。"""
from pathlib import Path
import csv
import gc
import hashlib
import importlib.util
import json
import math
import sys
import time

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'sparse_rasterizer_sixhour_20261110_0053'
TRANSFER=HERE.parent/'tensor_core_vector_tum_validation_20261111'
FOOTPRINT=HERE.parent/'large_footprint_20261110_0912'
spec=importlib.util.spec_from_file_location('frozen_transfer_inputs',TRANSFER/'validate.py')
catalog=importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)
p,harness=catalog.frozen_modules(catalog.frozen_paths())
torch,np=p.torch,p.np
sys.path.insert(0,str(HERE))
from scaling_runtime import Renderer,MODES,extension
spec=importlib.util.spec_from_file_location('readonly_footprint_runtime',FOOTPRINT/'footprint_runtime.py')
old=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
from final_edges import packet as synthetic_packet

TUNE=['Replica__office2__rgbd','Replica__room0__rgbd','TUM_RGBD__freiburg1_xyz__rgbd','VECtor__corner-slow__stereo']
SCENES=[s for s in p.SCENES if s['dataset_kind']=='Replica']+catalog.specs()
INDEX={Path(s['checkpoint']).resolve():s for s in SCENES}
old_views=harness.trajectory_views


def load_views(checkpoint,dataset):
    s=INDEX[Path(checkpoint).resolve()]
    if s['dataset_kind']=='Replica':
        return old_views(checkpoint,dataset)
    return catalog.historical_views(p,s,checkpoint)


harness.budget_viewer.load_views=load_views


def load_source(slug):
    s=next(s for s in SCENES if s['slug']==slug)
    return s,p.TeacherSamples(Path(s['checkpoint']),Path(s['dataset']))


def baseline():
    return p.create_renderer


def bits_equal(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def append(path,rows):
    if not rows:return
    exists=path.exists()
    with path.open('a',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        if not exists:w.writeheader()
        w.writerows(rows)


def request(source,camera,condition):
    return p.request(source.runtime,camera,condition)


def motions(source,quick=False):
    n=len(source.scene.views)
    if quick:
        return [('continuous',int(.43*n)),('continuous',int(.43*n)+1),('jump',int(.17*n)),('jump',int(.87*n))]
    return [('continuous',int(.43*n)+i) for i in range(6)]+[('jump',int(f*n)) for f in (.07,.31,.71,.97)]
