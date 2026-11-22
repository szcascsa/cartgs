"""从只读 MCGS 导出与 0.4.1 相同的嵌套 K；不修改训练或原模型。"""
from pathlib import Path
import hashlib
import json
import struct
import numpy as np

root = Path(__file__).resolve().parent.parent
source = Path('Q:/assets/Replica__office2__rgbd.mcgs')
raw = source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '1d735b0871ff94cc6acbf039ad014e0d1490ca4e209caef92648a0b9e53a6685'
version, size = struct.unpack_from('<II', raw, 8)
assert version == 1
header = json.loads(raw[16:16+size])
payload = (16+size+63) & ~63
values = {}
for name, s in header['sections'].items():
    data = raw[payload+s['offset']:payload+s['offset']+s['bytes']]
    assert hashlib.sha256(data).hexdigest() == s['sha256']
    values[name] = np.frombuffer(data, dtype=s['dtype']).reshape(s['shape'])

uids = values['uids'].astype(np.uint64)
x = uids + np.uint64(0x9e3779b97f4a7c15)
x = (x ^ (x >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
x = (x ^ (x >> 27)) * np.uint64(0x94d049bb133111eb)
rank = np.lexsort((uids, x ^ (x >> 31)))
n = header['N']
fields = ['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)] + [f'f_rest_{i}' for i in range(45)]
fields += ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
data = np.concatenate((values['xyz'], np.zeros((n, 3), dtype=np.float32), values['dc'].reshape(n, 3),
                       values['sh_rest'].transpose(0, 2, 1).reshape(n, 45), values['opacity_logits'],
                       values['log_scale'], values['rotation']), axis=1).astype('<f4')
assert data.shape == (n, 62)
assets = root/'source/SplatSample/app/src/main/assets/fixtures'
assets.mkdir(parents=True, exist_ok=True)
entries = []
for k in [1000, 2000, 3000, 4000, 5000, 10000, 15000, n]:
    rows = np.sort(rank[:k])
    text = 'ply\nformat binary_little_endian 1.0\ncomment mycartgs original Master; UID-hash nested subset\n'
    text += f'element vertex {k}\n' + ''.join(f'property float {f}\n' for f in fields) + 'end_header\n'
    body = data[rows].tobytes()
    path = assets/f'office2_k{k}.ply'
    path.write_bytes(text.encode('ascii')+body)
    # Decode what will be loaded and prove field order/FP32 identity, including SH channel order.
    readback = np.frombuffer(path.read_bytes()[len(text):], dtype='<f4').reshape(k, 62)
    assert readback.tobytes() == data[rows].tobytes()
    assert readback[:, 9:54].reshape(k, 3, 15).transpose(0, 2, 1).tobytes() == values['sh_rest'][rows].tobytes()
    entries.append(dict(k=k, asset='fixtures/'+path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        uid_sha256=hashlib.sha256(values['uids'][rows].tobytes()).hexdigest(), bytes=path.stat().st_size))

catalog = json.loads(Path('Q:/app/src/main/assets/scenes/catalog.json').read_text())['scenes'][0]
cv_rotation = np.array(catalog['view_rotation']).reshape(3, 3).T
r = np.diag([1., -1., -1.]) @ cv_rotation
c = np.array(catalog['view_position'])
t = -(r @ c)
# Stable matrix-to-quaternion conversion; proper rotation only, no Gaussian attribute rewrite.
evals, evecs = np.linalg.eigh(np.array([
    [r[0,0]-r[1,1]-r[2,2], r[0,1]+r[1,0], r[0,2]+r[2,0], r[2,1]-r[1,2]],
    [r[0,1]+r[1,0], r[1,1]-r[0,0]-r[2,2], r[1,2]+r[2,1], r[0,2]-r[2,0]],
    [r[0,2]+r[2,0], r[1,2]+r[2,1], r[2,2]-r[0,0]-r[1,1], r[1,0]-r[0,1]],
    [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], np.trace(r)],
])/3.)
q = evecs[:, np.argmax(evals)]
if q[3] < 0: q = -q
manifest = dict(scene=header['scene_id'], n=n, source_sha256=hashlib.sha256(raw).hexdigest(),
                source_fingerprint=header['teacher_fingerprint'], sh_degree=3, export='binary FP32 PLY; no SPZ quantization',
                fields=fields, subsets=entries, room_from_map_position=t.tolist(),
                room_from_map_quaternion_wxyz=q[[3, 0, 1, 2]].tolist(), camera_center=c.tolist(), camera_cv_to_world=catalog['view_rotation'])
(assets/'manifest.json').write_text(json.dumps(manifest, indent=2))
(root/'evidence/FIXTURE_EXPORT.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(dict(status='FP32_FIELDS_AND_SH_ORDER_PASS',n=n,subsets=len(entries),bytes=sum(v['bytes'] for v in entries))))
