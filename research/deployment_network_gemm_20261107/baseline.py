from common import *


def previous_extension():
    sys.path.insert(0,'/home/prmi/zwj/experiments/deployment_radix_tiling_20261107/build/mycartgs_deployment_fp32_v11')
    import mycartgs_deployment_fp32_v11
    return mycartgs_deployment_fp32_v11


def previous_runtime(s):
    spec=importlib.util.spec_from_file_location('deployment_v11_reference',ROOT/'reference/python/deployment_runtime.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.deployment_extension=previous_extension
    return module.DeploymentRenderer(s.model,s.gaussians,s.config)
