"""从逐阶段快照构建独立正确性 runner，不影响 native 计时。"""
from pathlib import Path
import json
import shutil
import subprocess
import sys

D = Path(__file__).resolve().parent
stage = D / sys.argv[1]
shutil.copytree(stage / 'source', stage / 'verify', dirs_exist_ok=True)
path = stage / 'verify/src/change_distill.cpp'
s = path.read_text()
s = '#include "verify_scope.h"\n' + s
s = s.replace('return torch::cat({target.xyz, target.log_scaling, target.rotation}, 1);',
              'auto packed_target = torch::cat({target.xyz, target.log_scaling, target.rotation}, 1);\n'
              '        cdverify::target(packed_target); return packed_target;')
# Capture loss, first-step gradients and updates, all E5 scalar losses.
for branch in ['selector', 'tf']:
    token = ('          total.backward();' if branch == 'selector' else '            total.backward();')
    # The Selector token also matches the TF suffix, so replace exactly once in order.
    s = s.replace(token, token + f'\n          cdverify::step(total, context.{branch}->parameters(), '
                  + ('true' if branch == 'tf' else 'false') + ');', 1)
    token = f'++report.{branch}_steps;'
    s = s.replace(token, token + f'\n          cdverify::updated(context.{branch}->parameters(), '
                  + ('true' if branch == 'tf' else 'false') + ');')
path.write_text(s)
bench = (stage / 'benchmark.cpp').read_text()
bench = bench.replace('int main(int argc,char** argv)', 'int benchmarkMain(int argc,char** argv)')
if sys.argv[1].startswith('stage_a'):
    original = (D / 'baseline/source/src/change_distill.cpp').read_text()
    original = original[original.index('GaussianAttrState transportTarget('):original.index('std::array<torch::Tensor, 3> tfErrors')]
    bench += '\n' + original.replace('transportTarget(', 'referenceTransport(')
    bench += (D / 'transport_verify.cpp').read_text()
    bench += '''
int main(int argc, char** argv) {
  torch::set_num_threads(4);
  Snapshot snapshot(argv[1]);
  fs::create_directories(argv[2]);
  writeJson(fs::path(argv[2]) / "transport_parity.json", transportParity(snapshot));
  const int result = benchmarkMain(argc, argv);
  if (result == 0) cdverify::archive.save_to((fs::path(argv[2]) / "verification.pt").string());
  return result;
}
'''
elif sys.argv[1] == 'stage_c':
    bench += (D / 'selector_shared_verify.cpp').read_text()
    bench += '''
int main(int argc, char** argv) {
  torch::set_num_threads(4);
  Snapshot snapshot(argv[1]); fs::create_directories(argv[2]);
  writeJson(fs::path(argv[2]) / "selector_shared_parity.json", sharedParity(snapshot));
  const int result = benchmarkMain(argc, argv);
  if (result == 0) cdverify::archive.save_to((fs::path(argv[2]) / "verification.pt").string());
  return result;
}
'''
else:
    bench += '''
int main(int argc, char** argv) {
  const int result = benchmarkMain(argc, argv);
  if (result == 0) cdverify::archive.save_to((fs::path(argv[2]) / "verification.pt").string());
  return result;
}
'''
(stage / 'benchmark_verify.cpp').write_text(bench)
build = (stage / 'build_native.py').read_text()
build = build.replace("versions=[('native', 'source')]", "versions=[('verify', 'verify')]")
build = build.replace("'benchmark.cpp'", "'benchmark_verify.cpp'")
build = build.replace("source = '#include \"perf_scope.h\"\\n'", "source = '#include \"perf_scope.h\"\\n#include \"verify_scope.h\"\\n'")
build = build.replace("'raw/build_commands.json'", "'raw/verify_build_commands.json'")
(stage / 'build_verify_stage.py').write_text(build)
shutil.copy2(D / 'verify_scope.h', stage / 'verify_scope.h')
subprocess.run([sys.executable, stage / 'build_verify_stage.py'], check=True)
