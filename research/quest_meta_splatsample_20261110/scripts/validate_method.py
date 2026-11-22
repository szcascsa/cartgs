"""校验设备捕获、原版逐位一致性、独立 FP32 网络参考与官方 PLY 传输。"""
from pathlib import Path
import hashlib
import json
import struct
import zipfile
import numpy as np

root = Path(__file__).resolve().parents[1]
folder = root / 'evidence/method_probe'
raw = (root / 'method-assets/scenes/Replica__office2__rgbd.mcgs').read_bytes()
size = struct.unpack_from('<I', raw, 12)[0]
meta = json.loads(raw[16:16+size]); base = (16+size+63)&~63
def tensor(name):
    s = meta['sections'][name]
    return np.frombuffer(raw, dtype=s['dtype'], count=int(np.prod(s['shape'])), offset=base+s['offset']).reshape(s['shape'])
def read(prefix, suffix, width, dtype='<f4'):
    return np.fromfile(folder / f'{prefix}_{suffix}.bin', dtype=dtype).reshape(-1, width)
def network(x, kind):
    for j in range(3):
        name = f'{kind}_w{j}' + ('_fused' if kind == 'student' and j == 0 else '')
        x = x @ tensor(name).T + tensor(f'{kind}_b{j}')
        if j < 2: x = np.maximum(x, np.float32(0))
    return x

checks = []
for case in json.loads((folder/'probe.json').read_text()):
    if 'case' not in case: continue
    prefix = case['case'].rsplit('/', 1)[-1]
    candidates = read(prefix, 'candidates', 1, '<u4').ravel()
    rows = read(prefix, 'selected', 1, '<u4').ravel()
    assert len(rows) == case['k'] and len(candidates) == case['candidates']
    assert np.all(np.diff(rows.astype(np.int64)) > 0)
    if case['global']: assert np.array_equal(candidates, np.arange(meta['N']))
    uid = tensor('uids')[rows]
    assert hashlib.sha256(uid.tobytes()).hexdigest() == case['selected_uid_sha256']
    info = dict(case=prefix, n=len(candidates), k=len(rows), global_scope=case['global'])
    if (folder/f'{prefix}_selector_input.bin').exists():
        x = read(prefix, 'selector_input', 19); noise = read(prefix, 'noise', 2)
        logits = network(x, 'selector') + noise
        exp = np.exp(logits-logits.max(1, keepdims=True)); probability = exp/exp.sum(1, keepdims=True)
        actual = read(prefix, 'selector_output', 2)
        info['selector_numpy_max_abs'] = float(np.abs(actual-probability).max(initial=0))
        assert info['selector_numpy_max_abs'] < 5e-5
        assert np.array_equal(x[:, :10], tensor('selector_static10')[candidates])
        assert np.array_equal(noise, tensor('noise2')[candidates])
        assert np.all(x[:, 18] == (0 if case['global'] else 1))
        if case['global']: assert np.all(x[:, 11:18] == 0)
        eligible = np.flatnonzero(tensor('eligible')[candidates])
        order = np.lexsort((tensor('uids')[candidates][eligible], -probability[eligible, 1]))
        reference_rows = np.sort(candidates[eligible[order[:len(rows)]]])
        info['numpy_selected_ids_equal'] = bool(np.array_equal(reference_rows, rows))
    if (folder/f'{prefix}_student_input.bin').exists():
        x = read(prefix, 'student_input', 62); actual = read(prefix, 'student_output', 14)
        reference = network(x, 'student')
        info['student_numpy_max_abs'] = float(np.abs(reference-actual).max(initial=0))
        assert info['student_numpy_max_abs'] < 5e-5
        assert np.array_equal(x[:, :3], tensor('offset3')[rows])
        assert np.all(x[:, 61] == (0 if case['global'] else 1))
        if case['global']: assert np.all(x[:, 54:61] == 0)
    data = (folder/f'{prefix}.ply').read_bytes(); start = data.index(b'end_header\n')+len(b'end_header\n')
    ply = np.frombuffer(data, dtype='<f4', offset=start).reshape(-1, 62)
    assert len(ply) == len(rows) and np.isfinite(ply).all()
    decoded = read(prefix, 'decoded', 14)
    assert np.array_equal(ply[:, :3], decoded[:, :3])
    assert np.array_equal(ply[:, 6:9], decoded[:, 6:9])
    assert np.array_equal(ply[:, 58:62], decoded[:, 9:13])
    assert np.array_equal(ply[:, 54:55], read(prefix, 'opacity_logits', 1))
    assert np.array_equal(ply[:, 9:54].reshape(-1, 3, 15).transpose(0,2,1), tensor('sh_rest')[rows])
    scale = np.exp(ply[:, 55:58]); info['ply_scale_roundtrip_max_rel'] = float((np.abs(scale-decoded[:,3:6])/decoded[:,3:6]).max(initial=0))
    assert info['ply_scale_roundtrip_max_rel'] < 2e-6
    checks.append(info)

legacy = []
with zipfile.ZipFile(Path('Q:/evidence/xnnpack_frames_inputs.zip')) as z, zipfile.ZipFile(Path('Q:/evidence/xnnpack_frame_oracle.zip')) as oracle:
    for prefix, old in [('local100', 'ratio10.rgba'), ('local_k1000', 'fixed1000.rgba')]:
        equality = {}
        for suffix in ['candidates','selector_input','noise','selected','student_input','selector_output','student_output']:
            equality[suffix] = (folder/f'{prefix}_{suffix}.bin').read_bytes() == z.read(f'{old}_{suffix}.bin')
        assert all(equality.values())
        differences = {}
        for kind, width in [('selector',2),('student',14)]:
            actual = read(prefix,f'{kind}_output',width)
            reference = np.frombuffer(oracle.read(f'{old}_{kind}_reference.bin'), dtype='<f4').reshape(-1,width)
            differences[kind+'_pytorch_oracle_max_abs'] = float(np.abs(actual-reference).max(initial=0))
            assert differences[kind+'_pytorch_oracle_max_abs'] < 5e-5
        legacy.append(dict(case=prefix, original_arm_bitwise=equality, **differences))
result = dict(status='PASS', note='NumPy is an independent FP32 arithmetic reference; PyTorch references are prior captured ordinary PyTorch CPU outputs for identical inputs.',
              cases=checks, legacy_parity=legacy, student_schema=meta['student_schema'],
              selector_parameters=meta['selector_parameters'], student_parameters=meta['student_parameters'])
(root/'evidence/METHOD_CORRECTNESS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(dict(status='PASS',cases=len(checks), legacy_bitwise_cases=len(legacy),
                     numpy_selection_mismatch=[c['case'] for c in checks if c.get('numpy_selected_ids_equal') is False],
                     selector_max_abs=max(c.get('selector_numpy_max_abs',0) for c in checks),
                     student_max_abs=max(c.get('student_numpy_max_abs',0) for c in checks)),indent=2))
