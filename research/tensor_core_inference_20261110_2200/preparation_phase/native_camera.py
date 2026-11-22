"""研究入口适配；正式可复用API在source/preparation_camera.py。"""
from prep_study import dp, HERE, runtime

_api=runtime('native_camera_api',HERE/'source/preparation_camera.py')

class NativeCameraPreparer(_api.NativeCameraPreparer):
    def __init__(self,camera_class,*,parts=True):
        super().__init__(camera_class,dp.deployment_extension(),parts=parts)
