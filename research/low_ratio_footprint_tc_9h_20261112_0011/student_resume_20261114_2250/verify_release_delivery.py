"""交付前离线核验覆盖、质量、逐位证据、保护、图清单及实际数据hash。"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
R = HERE / 'reports'


def main():
    result = {}
    for filename in ['FINAL_AUDIT.json', 'FINAL_DELIVERY_AUDIT.json', 'FINAL_PROTECTED_INTEGRITY.json']:
        data = json.loads((R / filename).read_text(encoding='utf-8'))
        assert data['passed'], filename
        result[filename] = True
    parity = json.loads((R / 'final_conservative_parity/DONE.json').read_text(encoding='utf-8'))
    assert parity['checks'] == 3528 and parity['bitwise']
    frame = pd.read_csv(R / 'final_conservative_parity/checks.csv')
    assert frame.rgb_bitwise.all() and frame.radii_bitwise.all() and frame.ids_bitwise.all()
    pure = pd.read_csv(R / 'RELEASE_THIN_SCENE_RATIO.csv')
    assert pure.groupby('workload').scene.nunique().to_dict() == {'nested': 32, 'student_lod': 18}
    assert pure.condition.nunique() == 14 and set(pure['round']) == {1, 2}
    full = pure[pure.condition.eq('full') & pure.variant.eq('new_thin')]
    assert len(full) == 100 and full.F2_speedup.ge(1).all() and full.thin182_speedup.ge(1).all()
    unseen = json.loads((R / 'UNSEEN_SUMMARY.json').read_text(encoding='utf-8'))
    assert unseen['quality_pass'] and unseen['nested_quality_pass']
    decision = json.loads((R / 'FINAL_ACCEPTANCE_DECISION.json').read_text(encoding='utf-8'))
    assert decision['universal_large_footprint_quality_pass'] is False
    assert decision['production_default_changed'] is False
    figures = HERE / 'figures/release_20261115'
    png = sorted(figures.glob('*.png'))
    assert len(png) == 54 and len(list(figures.glob('*.svg'))) == 54
    for path in png:
        assert path.with_suffix('.svg').exists()
        with Image.open(path) as image:
            assert image.width > 1000 and image.height > 500 and image.info['dpi'][0] >= 449
    data_hashes = json.loads((figures / 'DATA_SHA256.json').read_text(encoding='utf-8'))
    for name, value in data_hashes.items():
        assert hashlib.sha256((R / name).read_bytes()).hexdigest() == value, name
    for filename in ['release_joint_memcheck_continued_a1.log', 'release_joint_initcheck_continued_a1.log',
                     'release_joint_synccheck_continued_a1.log', 'release_tail_memcheck_a1.log',
                     'release_tail_initcheck_a1.log', 'release_tail_synccheck_a1.log']:
        assert 'ERROR SUMMARY: 0 errors' in (HERE / 'logs' / filename).read_text(encoding='utf-8')
    report = HERE / 'LOW_RATIO_FOOTPRINT_AND_TENSOR_CORE_9H_REPORT.md'
    text = report.read_text(encoding='utf-8')
    for value in ['20.84', '3528', '0.6325', '1.0645', '未普遍随K单调下降']:
        assert value in text, value
    result.update(passed=True, verified_at=datetime.now().astimezone().isoformat(),
        full_pairs=100, plot_pairs=54, sanitizer_logs=6, conservative_parity_checks=3528,
        universal_target_claimed=False, report_bytes=report.stat().st_size,
        report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
    (R / 'FINAL_LOCAL_DELIVERY_VERIFICATION.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
