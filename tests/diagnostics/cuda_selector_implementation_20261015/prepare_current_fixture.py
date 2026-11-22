"""冻结当前 continuous maturity checkpoint，保持原行号和 saved uniform。"""
import json
from pathlib import Path
import struct
import shutil
import sys

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData
import torch

dest = Path(__file__).resolve().parent
root = dest.parents[2]
sys.path.insert(0, str(root/'python'))
from selector import load_selector_metadata
from utils import MiniCam, focal2fov

checkpoint = Path('/home/prmi/zwj/mycartgs/results/cudatfnewprotection/tum_rgbd/roomfixednice/6900_shutdown/ply')
cloud = next((checkpoint/'point_cloud').glob('iteration_*/point_cloud.ply'))
vertex = PlyData.read(cloud)['vertex']
sh = sorted((n for n in vertex.data.dtype.names if n.startswith('f_rest_')), key=lambda n:int(n[7:]))
names = ['x','y','z']+[f'rot_{i}' for i in range(4)]+[f'scale_{i}' for i in range(3)]+['opacity']+[f'f_dc_{i}' for i in range(3)]+sh


def binary(name, data):
    data = np.asarray(data, dtype='<f4')
    (dest/name).write_bytes(struct.pack('<qq', *data.shape)+data.tobytes())


binary('current_input.bin', np.stack([vertex[n] for n in names], axis=1))
metadata = load_selector_metadata(cloud.parent/'selector_metadata.ply', 'cpu')
binary('current_metadata.bin', torch.stack([metadata[k] for k in ['maturity','drift_ema','cached_uniform']], 1).numpy())
storage = cv2.FileStorage(str(checkpoint/'selector_config.yaml'), cv2.FILE_STORAGE_READ)
entry_node = storage.getNode('Elastic.entry_maturity')
entry = entry_node.real() if not entry_node.empty() else 0.
storage.release()
(dest/'current_spec.txt').write_text(f'{entry}\n')
frames = []
for i, c in enumerate(json.loads((checkpoint/'cameras.json').read_text())[:3]):
    c2w = np.eye(4); c2w[:3,:3] = c['rotation']; c2w[:3,3] = c['position']
    camera = MiniCam(320,240,focal2fov(c['fx'],c['width']),focal2fov(c['fy'],c['height']),np.linalg.inv(c2w))
    values = np.concatenate([camera.world_view_transform.cpu().numpy().reshape(-1),camera.full_proj_transform.cpu().numpy().reshape(-1),
                             camera.camera_center.cpu().numpy(), [camera.FoVx,camera.FoVy]]).astype('<f4')
    (dest/f'current_camera_{i}.bin').write_bytes(values.tobytes())
    path = Path('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_room')/c['img_name']
    image = np.asarray(Image.open(path).convert('RGB').resize((320,240),Image.Resampling.BILINEAR), dtype=np.float32).transpose(2,0,1)/255
    binary(f'current_gt_{i}.bin', image.reshape(3,-1))
    frames.append(str(path))
(dest/'current_checkpoint.txt').write_text(str(checkpoint)+'\n')
frozen = dest/'current_checkpoint'
frozen.mkdir(exist_ok=True)
for name in ('selector.pt', 'transform_field.pt', 'transform_state.json', 'selector_config.yaml'):
    shutil.copy2(checkpoint/name, frozen/name)
(dest/'current_fixture.json').write_text(json.dumps(dict(checkpoint=str(checkpoint), N=len(vertex),
    entry=entry, eligible=int((metadata['maturity']>=entry).sum()), frames=frames,
    limit='Fixed real map and saved maturity/uniform; no topology updates in deterministic replay'), indent=2)+'\n')
print((dest/'current_fixture.json').read_text())
