"""只导出一个真实 camera 和 RGB ground truth，不保存渲染图。"""
import json,struct,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
dest=Path(__file__).resolve().parent
repo=dest.parents[2]
sys.path.insert(0,str(repo/'python'))
from utils import MiniCam,focal2fov
checkpoint=Path(json.loads((dest/'inventory.json').read_text())['checkpoint'])
tf_state=json.loads((checkpoint/'transform_state.json').read_text())
tf_spec=[tf_state[k] for k in ['spatial_resolution','ratio_resolution','feature_dim','hidden_dim']]
tf_spec += [len(tf_state['multires']),*tf_state['multires'],*tf_state['aabb_min'],*tf_state['aabb_max']]
(dest/'tf_fixture_spec.txt').write_text(' '.join(map(str,tf_spec)))
c=json.loads((checkpoint/'cameras.json').read_text())[0]
c2w=np.eye(4);c2w[:3,:3]=c['rotation'];c2w[:3,3]=c['position']
camera=MiniCam(320,240,focal2fov(c['fx'],c['width']),focal2fov(c['fy'],c['height']),np.linalg.inv(c2w))
values=np.concatenate([camera.world_view_transform.cpu().numpy().reshape(-1),camera.full_proj_transform.cpu().numpy().reshape(-1),camera.camera_center.cpu().numpy(),[camera.FoVx,camera.FoVy]]).astype('<f4')
(dest/'camera.bin').write_bytes(values.tobytes())
path=Path('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household')/c['img_name']
image=np.asarray(Image.open(path).convert('RGB').resize((320,240),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1)/255
(dest/'gt.bin').write_bytes(struct.pack('<qq',3,240*320)+image.astype('<f4').tobytes())
(dest/'render_fixture.json').write_text(json.dumps(dict(camera=c,image=str(path),width=320,height=240,resize='PIL bilinear; native Mapping image-cache resize parity not asserted',loss='native 0.8 L1 + 0.2 (1-fast_ssim)'),indent=2))
print(path)
