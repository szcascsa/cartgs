from common import *


def previous_runtime(s):
    spec=importlib.util.spec_from_file_location('previous_deployment_runtime',ROOT/'reference/python/deployment_runtime.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.path.insert(0,'/home/prmi/zwj/experiments/deployment_native_select_20261106/build/mycartgs_deployment_fp32_v4')
    import mycartgs_deployment_fp32_v4
    module.deployment_extension=lambda:mycartgs_deployment_fp32_v4
    return module.DeploymentRenderer(s.model,s.gaussians,s.config)


def first_stage_runtime(s):
    spec=importlib.util.spec_from_file_location('first_stage_deployment_runtime',ROOT/'reference/python/deployment_runtime.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.path.insert(0,str(ROOT/'build/mycartgs_deployment_fp32_v5'))
    import mycartgs_deployment_fp32_v5
    module.deployment_extension=lambda:mycartgs_deployment_fp32_v5
    return module.DeploymentRenderer(s.model,s.gaussians,s.config)
