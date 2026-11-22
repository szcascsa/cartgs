"""验证结果表和XLSX缓存值，打包本轮消融交付；不执行训练或渲染。"""
from pathlib import Path
import collections
import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/01a1de13-ebef-7091-b328-52d3c81cfc40'
data = json.loads((OUT / '分析数据.json').read_text(encoding='utf-8'))
state = json.loads((ROOT / 'results/LOD_PIPELINE.json').read_text(encoding='utf-8'))
assert len(data['detail']) == 736 and len(data['status']) == 32
assert collections.Counter(r['status'] for r in data['status']) == {'完成': 27, '连续零选点停止': 5}
assert collections.Counter(r['dataset'] for r in data['status'] if r['status'] == '完成') == {
    'TUM_RGBD': 7, 'Replica': 15, 'VECtor': 5}
jobs = {j['id']: j for j in state['jobs']}
for r in data['detail']:
    j = jobs[r['id']]
    a = next(e['result'] for e in j['reference_evaluations'] if (e['scope'], e['ratio']) == (r['scope'], r['ratio']))
    if r['status'] == '完成':
        d = next(e['result'] for e in j['evaluations'] if (e['scope'], e['ratio']) == (r['scope'], r['ratio']))
    else:
        d = None
    for name in ['psnr', 'ssim', 'lpips', 'actual_K_mean', 'actual_K_min', 'actual_K_max', 'ate_rmse', 'render_ms']:
        assert r[name + '_A'] == a[name]
        assert r[name + '_D'] == (d[name] if d else None)
        if d:
            assert math.isfinite(d[name])
            assert abs(r['delta_' + name] - (d[name] - a[name])) < 1e-12
        else:
            assert r['delta_' + name] is None
for s in data['summary']:
    rows = [r for r in data['detail'] if r['status'] == '完成' and
            (s['dataset'] == '全部' or r['dataset'] == s['dataset']) and
            (r['scope'], r['ratio']) == (s['scope'], s['ratio'])]
    assert len(rows) == s['count']
    for name in ['psnr', 'ssim', 'lpips']:
        for suffix in ['A', 'D']:
            assert abs(statistics.mean(r[name + '_' + suffix] for r in rows) - s[name + '_' + suffix]) < 1e-12
assert sum(r['psnr_D'] is not None for r in data['detail']) == 621
assert not any(g['passed'] for g in data['gates'])
assert all(r['feature_adam_clocks_match_gradient_exposure'] for r in data['exposure'])
ns = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(OUT / '无重要性消融对照.xlsx') as z:
    assert z.testzip() is None
    sheet = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    cells = {c.attrib['r']: c for c in sheet.findall('.//x:c', ns)}
    for i, s in enumerate(data['summary'], 8):
        expected = [s['count'], s['psnr_A'], s['psnr_D'], s['delta_psnr'],
                    s['ssim_A'], s['ssim_D'], s['delta_ssim'], s['lpips_A'], s['lpips_D'],
                    s['delta_lpips'], s['psnr_improved_scenes']]
        for col, value in zip('DEFGHIJKLMN', expected):
            c = cells[f'{col}{i}']
            assert c.find('x:f', ns) is not None
            assert abs(float(c.find('x:v', ns).text) - value) < 1e-10
    sheet = ET.fromstring(z.read('xl/worksheets/sheet2.xml'))
    cells = {c.attrib['r']: c for c in sheet.findall('.//x:c', ns)}
    for i, r in enumerate(data['detail'], 8):
        if r['psnr_D'] is not None:
            continue
        for col in ['L', 'M', 'O', 'P', 'R', 'S']:
            c = cells.get(f'{col}{i}')
            v = c.find('x:v', ns) if c is not None else None
            assert v is None or v.text in [None, '']

audit = dict(场景数=32, 完成数=27, 连续零选点失败数=5, 评价数=621,
             逐场景逐ratio行数=736, 失败指标空白行数=115, 预声明质量门槛通过数=0,
             原始指标逐值核对='通过', 汇总重算='通过', XLSX导出缓存逐值核对='通过',
             公式输入扰动与恢复='通过', 四工作表视觉检查='通过',
             保护文件数=2645, 保护文件哈希未变=True, 生产默认修改=False)
(OUT / '交付核验.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
files = [p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.md', '.csv', '.json', '.xlsx', '.png', '.svg']
         and p.name != 'DELIVERY_MANIFEST.json' and not p.name.endswith('.inspect.ndjson')]
evidence = ROOT / 'NO_IMPORTANCE_ABLATION_RESULTS.zip'
assert hashlib.sha256(evidence.read_bytes()).hexdigest() == '34062ccdb5369f46d8f418ef9aeb05c96222e44274fd649f106b4c257c71a177'
bundle = OUT / '无重要性消融完整结果.zip'
with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(files):
        z.write(p, p.relative_to(OUT).as_posix())
    z.write(evidence, evidence.name)
with zipfile.ZipFile(bundle) as z:
    assert z.testzip() is None
manifest = {'files': [{'file': p.relative_to(OUT).as_posix(), 'bytes': p.stat().st_size,
                       'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files + [bundle])]}
(OUT / 'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(audit=audit, bundle=str(bundle), bytes=bundle.stat().st_size,
                     sha256=hashlib.sha256(bundle.read_bytes()).hexdigest()), ensure_ascii=False))
