"""复用原审计插桩，生成独立 profile binary。"""
from pathlib import Path
import subprocess
import sys

D = Path(__file__).resolve().parent
stage = D / sys.argv[1]
s = (D / 'baseline/generate_instrumentation.py').read_text()
if sys.argv[1] == 'stage_c':
    # These original expressions moved into the shared-probability helper.
    s = s.replace("    ('torch::full({indices.numel(), 1}, ratio, after.xyz.options())', 'ratio_tensor'),\n", '')
    s = s.replace("    ('student->forwardProbability(\\n      after.xyz, after.rotation, after.log_scaling.exp(), ratios, temperature)', 'student_forward'),",
                  "    ('student->forwardProbabilities(\\n      after.xyz, after.rotation, after.log_scaling.exp(), ratios, temperature)', 'student_forward'),")
    start = s.index('# Shape probes are metadata reads')
    end = s.index("s = s.replace('        auto total", start)
    s = s[:start] + s[end:]
    # The original scope signature now occurs in the shared-feature helper.
    s = s.replace("('  auto after = current.select(indices);', 'selector_loss_call')",
                  "('  auto after = current.select(indices);', 'selector_probabilities')")
    s = s.replace("write('src/change_distill.cpp', s)",
                  "write('src/change_distill.cpp', s)")
if sys.argv[1] != 'baseline':
    s = s.replace("  torch::NoGradGuard no_grad;\\n  auto ids = event.correction_group_id",
                  "  return transport.apply(std::move(target), indices);")
    # Explicit separate EVENT-ONCE metadata scope.
    s = s.replace("write('src/change_distill.cpp', s)",
                  "s = s.replace('const TargetTransport transport(event);', 'const auto transport = CD_OP(\"transport_metadata\", TargetTransport(event));')\nwrite('src/change_distill.cpp', s)")
(stage / 'generate_profile_stage.py').write_text(s)
subprocess.run([sys.executable, stage / 'generate_profile_stage.py'], check=True)
build = (stage / 'build_native.py').read_text().replace("versions=[('native', 'source')]", "versions=[('instrumented', 'instrumented')]")
build = build.replace("'raw/build_commands.json'", "'raw/profile_build_commands.json'")
(stage / 'build_profile_stage.py').write_text(build)
subprocess.run([sys.executable, stage / 'build_profile_stage.py'], check=True)
