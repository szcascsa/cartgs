"""将正式测速绑定到同一份已通过GPU验收的实现，禁止复用旧计费口径。"""
import hashlib
import json
from pathlib import Path

PROTOCOL_ID='sm_area_feedback_v1'


def implementation_identity(root):
    paths=[root/'source/research/gpu_load_distilled_ramp_20261108'/name for name in
           ('experiment.py','phase_load.py','ramp_load.cu','sm_area.h','run_guard.py',
            'reference.py','previous_experiment.py','phase_gate.py','duty_budget.h')]
    paths.extend(root/name for name in (
        'build/libramp_load.so',
        'build/mycartgs_deployment_fp32_v31/mycartgs_deployment_fp32_v31.so',
        'build/mycartgs_frozen_inference_v3/mycartgs_frozen_inference_v3.so',
        'reports/PROTECTED_BEFORE.json',
        'source/original_renderer.py',
    ))
    for name in ('source/python','source/evaluation','source/original_raster'):
        paths.extend(sorted(path for path in (root/name).rglob('*') if path.is_file()
            and path.suffix in ('.py','.so','.yaml','.h','.hpp','.cpp','.cu')))
    hashes={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return dict(protocol=PROTOCOL_ID,files=hashes)


def require_gpu_acceptance(root,identity):
    proof=root/'reports/GPU_ACCEPTANCE.json'
    if not proof.is_file():
        raise RuntimeError('当前实现尚未通过GPU验收，禁止正式测速')
    payload=json.loads(proof.read_text())
    if payload.get('passed') is not True or payload.get('identity')!=identity:
        raise RuntimeError('GPU验收未通过或实现哈希已变化，必须重新验收')
    if payload.get('cases')!=3 or payload.get('rows')!=6000 or payload.get('graph_cases')!=40:
        raise RuntimeError('GPU验收记录不完整，禁止正式测速')
    return payload


def write_gpu_acceptance(root,folder,identity):
    if implementation_identity(root)!=identity:
        raise RuntimeError('GPU验收期间源码或二进制发生变化')
    checks=[]
    for method in ('cartgs_full','ours_full','student_fixed1000'):
        case=json.loads((folder/method/'ACCEPTANCE.json').read_text())
        assert case['summary']['requests']==2000 and case['bin_match']
        assert case['hardware']['billing_sm_area'] and case['maximum_on_ns']<=2000000
        assert len(case['bins'])==8 and all(item['samples']==250 for item in case['bins'])
        checks.append(method)
    graph=json.loads((folder/'GRAPH_NATIVE_PARITY.json').read_text())
    assert graph['passed'] and len(graph['cases'])==40
    proof=dict(passed=True,identity=identity,source_run=str(folder),cases=len(checks),
               rows=6000,graph_cases=40)
    (root/'reports/GPU_ACCEPTANCE.json').write_text(json.dumps(proof,indent=2,ensure_ascii=False))
