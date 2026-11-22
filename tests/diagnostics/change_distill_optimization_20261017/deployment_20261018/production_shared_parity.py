"""使用正式CMake共享库验证跨ratio梯度；不采集性能时间。"""
from pathlib import Path
import json
import os
import subprocess

ROOT = Path('/home/prmi/zwj/mycartgs')
D = Path(__file__).resolve().parent
AUDIT = ROOT / 'results/change_distill_optimization/20261017'
source = '#include "include/transform_field_cuda_optimizer.h"\n#define main savedBenchmarkMain\n'
source += '#include "' + str(AUDIT / 'baseline/benchmark.cpp') + '"\n#undef main\n'
source += (AUDIT / 'selector_shared_verify.cpp').read_text()
source += '''
int main(int argc, char** argv) {
  torch::set_num_threads(4);
  Snapshot snapshot(argv[1]);
  writeJson(argv[2], sharedParity(snapshot));
  return 0;
}
'''
(D / 'production_shared_parity.cpp').write_text(source)
command = json.loads((D / 'benchmark_build_command.json').read_text())
command[command.index(str(AUDIT / 'baseline/benchmark.cpp'))] = str(D / 'production_shared_parity.cpp')
command[command.index('-o') + 1] = str(D / 'production_shared_parity')
subprocess.run(command, check=True)
env = os.environ.copy()
env['LD_LIBRARY_PATH'] = ':'.join(str(p) for p in [ROOT / 'lib', ROOT / 'third_party/libtorch-cu118/libtorch/lib',
    Path('/usr/local/cuda-11.8/lib64'), ROOT / 'third_party/ORB-SLAM3/lib'])
subprocess.run([str(D / 'production_shared_parity'), str(ROOT / 'results/change_distill_ablation/20261017-2038/events_supplement/local_loop/event_6207_1'),
                str(D / 'production_shared_parity.json')], env=env, check=True)
rows = json.loads((D / 'production_shared_parity.json').read_text())
assert len(rows) == 762 and all(r['allclose'] for r in rows.values())
print('Formal CMake library cross-ratio parity PASS:', len(rows), 'comparisons;',
      'bitwise', all(r['bitwise_equal'] for r in rows.values()), flush=True)
