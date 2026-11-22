from common import *


def previous_runtime(s):
    spec=importlib.util.spec_from_file_location('deployment_v7_reference',ROOT/'reference/python/deployment_runtime.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.path.insert(0,'/home/prmi/zwj/experiments/deployment_device_counts_20261107/build/mycartgs_deployment_fp32_v7')
    import mycartgs_deployment_fp32_v7
    module.deployment_extension=lambda:mycartgs_deployment_fp32_v7
    return module.DeploymentRenderer(s.model,s.gaussians,s.config)


def previous_extension():
    sys.path.insert(0,'/home/prmi/zwj/experiments/deployment_device_counts_20261107/build/mycartgs_deployment_fp32_v7')
    import mycartgs_deployment_fp32_v7
    return mycartgs_deployment_fp32_v7
