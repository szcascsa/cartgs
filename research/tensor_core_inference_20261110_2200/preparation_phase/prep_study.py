"""新增准备专项；只读A/B/C及原模型，D使用独立源码与构建。"""
from pathlib import Path
import sys
import importlib.util
from functools import lru_cache

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import study as prior
from study import *
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def runtime(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dp = runtime('tensor_preparation_runtime', HERE / 'source/deployment_runtime.py')


@lru_cache(maxsize=1)
def initial_runtime():
    from torch.utils.cpp_extension import _import_module_from_library
    mod = runtime('initial_tensor_runtime', ROOT / 'development_v1/source/deployment_runtime.py')
    ext = _import_module_from_library('mycartgs_tensor_deployment_v1',
                                     str(ROOT / 'development_v1'), True)
    mod.deployment_extension = lambda: ext
    return mod


def make_engine(source, student, variant='D', *, frustum=0, projection=0, device=0,
                graph=True, selector=2, compensator=2):
    cls = dict(A=OriginalRenderer, C=prior.tc.DeploymentRenderer,
               D=dp.DeploymentRenderer).get(variant)
    kw = dict(student=student, prepare_graph=graph)
    if variant == 'B':
        cls = initial_runtime().DeploymentRenderer
        kw.update(tensor_selector=1, tensor_compensator=1, tensor_warps=4)
    elif variant != 'A':
        kw.update(tensor_selector=selector, tensor_compensator=compensator,
                  tensor_warps=4, tensor_compensator_warps=2)
    run = cls(source.scene.model, source.scene.gaussians, source.scene.config, **kw)
    if variant == 'D':
        run._pipeline._preparation_options(frustum, projection, device)
        run.preparation_config = dict(frustum=frustum, projection=projection, device=device)
    return run


if __name__ == '__main__':
    ext = dp.deployment_extension()
    print('PREPARATION_BUILD_PASS', ext.__file__, sha(ext.__file__), flush=True)
