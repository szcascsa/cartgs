"""本地独立复核源文件 SHA、训练 trace、成员数直方图与交付证据。"""
import csv
import hashlib
import json
from pathlib import Path
import re
import tarfile

root = Path(__file__).resolve().parent
repo = root.parents[2]
evidence = root/'evidence'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
expected_archive = 'c68da2cdcb24cc0a2210b5e0cbc1160acbba5c836b09ee45ea99c76af07ed2ae'
assert sha(root/'evidence.tar.gz') == expected_archive
with tarfile.open(root/'evidence.tar.gz') as archive:
    archive.extractall(evidence, filter='data')
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
files = read(evidence/'evidence_manifest.json')
assert all(sha(evidence/p) == digest for p, digest in files.items())
sources = read(evidence/'source_manifest.json')
assert all(sha(repo/p) == digest for p, digest in sources.items())
gates = read(evidence/'validation/correctness.json')
assert len(gates) == 11 and all(v['passed'] and v['code'] == 0 for v in gates.values())
summary = read(evidence/'validation/audit_summary.json')
hist = summary['occupancy_histogram_0_to_10']
assert hist[0] == 0 and sum(hist) == summary['anchors']
assert sum(j*n for j, n in enumerate(hist)) == summary['gaussians']
assert sum(summary['anchors_by_level']) == summary['anchors']
run = evidence/'variable_seed2026'
cloud = run/'10000_shutdown/ply/point_cloud/iteration_10000'
with (cloud/'point_cloud.ply').open('rb') as stream:
    header = b''
    while not header.endswith(b'end_header\n'):
        line = stream.readline()
        assert line
        header += line
assert int(re.search(rb'element vertex (\d+)', header).group(1)) == summary['gaussians']
trace = list(csv.DictReader((run/'serial_full_lod_trace.csv').open()))
assert len(trace) == 10000
for i, row in enumerate(trace):
    assert int(row['mapping']) == i+1 and int(row['ratio_index']) == i%4
    assert all(int(row[k]) == 1 for k in ['full', 'lod', 'full_gaussian_update', 'lod_gaussian_update', 'selector_update'])
events = re.findall(r'Alternating Topology\] round=(\d+) phase=(Gaussian|Anchor) mapping=(\d+)', (run/'training.log').read_text())
assert len(events) == summary['topology_events'] == 19
for j, (index, phase, _) in enumerate(events):
    assert int(index) == j and phase == ('Gaussian' if j%2 == 0 else 'Anchor')
baseline = repo/'tests/anchor_selector/serial_joint_lod_validation/evidence/joint_seed2026/input_selector.yaml'
assert (run/'input_selector.yaml').read_text() == baseline.read_text()+'\nSelector.variable_anchor_members: 1\n'
result = dict(passed=True, evidence_files_sha_verified=len(files), source_files_sha_verified=len(sources),
              archive_sha256=expected_archive, correctness_gates=len(gates), trace_rows=len(trace),
              topology_events=len(events), anchors=summary['anchors'], gaussians=summary['gaussians'],
              same_config_except_variable_members=True, quality_improvement_established=False)
(root/'local_delivery_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
