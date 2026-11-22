"""仅在非计时阶段导出代表性Nsight；保留原rep及分析CSV，SQLite可重建。"""
from pathlib import Path
import argparse
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = HERE / 'reports' / args.output
    out.mkdir(exist_ok=False)
    names = [p.stem for p in (HERE / 'reports').glob('release_profile_g*.nsys-rep')]
    names += [p.stem for p in (HERE / 'reports').glob('release_raster_profile*.nsys-rep')]
    assert len(names) == 3, names
    for name in sorted(names):
        sqlite = HERE / 'reports' / (name + '.sqlite')
        assert not sqlite.exists()
        subprocess.run(['/usr/local/cuda-11.8/bin/nsys', 'export', '--type=sqlite',
            '--output=' + str(sqlite), str(sqlite.with_suffix('.nsys-rep'))], check=True)
        subprocess.run([sys.executable, '-B', 'analyze_student_profile.py', name,
            '--output', name + '_RELEASE_ANALYSIS'], cwd=HERE, check=True)
    (out / 'DONE.json').write_text(json.dumps(dict(passed=True, names=names,
        scope='Profile Kernel累计与Host API，仅用于归因，不替代非Profile墙钟'), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
