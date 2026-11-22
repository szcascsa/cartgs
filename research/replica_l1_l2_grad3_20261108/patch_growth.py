"""只在本实验源码副本中降低L1→L2增生阈值，原工程保持不变。"""
from pathlib import Path
import difflib
import hashlib
import json

HEADER = Path('improvements/selector/explicit_anchor_blocks.h')
BEFORE = '    double low=threshold*std::pow(alpha,level), high=low*alpha;'
AFTER = BEFORE+'\n    if(level==1)high=threshold*3.;  // Isolated L1-to-L2 threshold experiment.'


def patch(text):
    if text.count(BEFORE) != 1 or 'Isolated L1-to-L2 threshold experiment' in text:
        raise ValueError('增生实现与核查版本不一致，或补丁已应用；拒绝重复修改')
    return text.replace(BEFORE, AFTER)


def apply(source, reports):
    source, reports = Path(source), Path(reports)
    path = source/HEADER
    before = path.read_text(encoding='utf-8')
    after = patch(before)
    path.write_text(after, encoding='utf-8')
    reports.mkdir(parents=True, exist_ok=True)
    (reports/'GROWTH_THRESHOLD.diff').write_text(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile='baseline/'+str(HEADER),
        tofile='experiment/'+str(HEADER))), encoding='utf-8')
    receipt = dict(changed_file=HEADER.as_posix(), before_sha256=hashlib.sha256(before.encode()).hexdigest(),
                   after_sha256=hashlib.sha256(after.encode()).hexdigest(),
                   original_l1_to_l2_tau0_multiplier=4, new_l1_to_l2_tau0_multiplier=3,
                   base_threshold_calibration_unchanged=True, level0_thresholds_unchanged=True,
                   level1_same_level_lower_threshold_unchanged=True, level2_rules_unchanged=True,
                   gradient_statistic='per-Gaussian temporal mean; no new across-members mean')
    (reports/'GROWTH_THRESHOLD_CHANGE.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return receipt
