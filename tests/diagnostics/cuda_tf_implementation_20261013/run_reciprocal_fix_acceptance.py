"""正则数值修正后的完整复验；独立路径保存，保留修正前全部结果。"""
import json
import os
import subprocess
import time
from pathlib import Path

d = Path(__file__).resolve().parent
root = d.parents[2]
python = "/home/prmi/anaconda3/envs/zwj/bin/python"
cmake = "/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake"
folder = d / "regularization_reciprocal_fix"
environment = os.environ.copy()
environment["PATH"] = "/usr/local/cuda-11.8/bin:" + environment["PATH"]
environment["LD_LIBRARY_PATH"] = ":".join([
    str(root / "third_party/libtorch-cu118/libtorch/lib"), "/usr/local/cuda-11.8/lib64",
    str(root / "lib"), str(root / "third_party/ORB-SLAM3/lib"), environment.get("LD_LIBRARY_PATH", "")])

def run(command, log_name, cwd=root):
    print("BEGIN", log_name, flush=True)
    with (folder / log_name).open("w") as log:
        result = subprocess.run(command, cwd=cwd, env=environment, stdout=log,
                                stderr=subprocess.STDOUT, timeout=10800)
    assert result.returncode == 0, (log_name, result.returncode)
    print("END", log_name, flush=True)

try:
    while not (d / "reciprocal_fix_main_build.exit").exists():
        time.sleep(2)
    assert (d / "reciprocal_fix_main_build.exit").read_text().strip() == "0"
    assert (folder / "near_zero_adam/phase_d.exit").read_text().strip() == "0"
    # The generated copies retain the same driver/config/measurement code.
    # Only output/manifest paths change; exact sources are saved for review.
    for name in ("run_online", "sample_online_memory", "run_online_evaluation", "collect_online_results"):
        text = (d / (name+".py")).read_text()
        assert "guard_off_" in text, name
        (d / (name+"_reciprocal_fix.py")).write_text(text.replace("guard_off_", "guard_off_reciprocal_fix_"))
    profile = (d / "run_online_profile.py").read_text()
    profile = profile.replace("guard_off_", "guard_off_reciprocal_fix_")
    profile = profile.replace("online_tf_profile", "online_tf_profile_reciprocal_fix")
    profile = profile.replace("cuda_tf_stage_profile_20261014", "cuda_tf_stage_profile_reciprocal_fix_20261014")
    (d / "run_online_profile_reciprocal_fix.py").write_text(profile)
    run([cmake, "--build", str(d / "build"), "--target", "training_benchmark", "online_profiler_parity", "-j2"], "benchmark_build.log")
    checkpoint = root / "results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply"
    run([str(d / "build/online_profiler_parity"), str(checkpoint), str(folder / "online_profiler_parity.csv")], "online_profiler_parity.log")
    run([str(d / "build/training_benchmark"), str(checkpoint),
         str(root / "tests/diagnostics/original_tf_cuda_audit_20261013/input.bin"),
         str(folder / "phase_5"), "cuda_optimizer"], "phase_5.log")
    (folder / "phase_5.exit").write_text("0\n")
    for protocol in ("original", "desk_start1000"):
        with (folder / (protocol+"_memory_sampler.log")).open("w") as log:
            sampler = subprocess.Popen([python, str(d / "sample_online_memory_reciprocal_fix.py"),
                                        "--protocol", protocol], env=environment, stdout=log,
                                       stderr=subprocess.STDOUT)
            run([python, str(d / "run_online_reciprocal_fix.py"), "--protocol", protocol], protocol+"_mapping.log")
            assert sampler.wait(timeout=10) == 0
    for protocol in ("original", "desk_start1000"):
        run([python, str(d / "run_online_evaluation_reciprocal_fix.py"), "--protocol", protocol], protocol+"_evaluation.log")
        run([python, str(d / "collect_online_results_reciprocal_fix.py"), "--protocol", protocol], protocol+"_summary.log")
    (d / "guard_off_reciprocal_fix_remaining_acceptance.exit").write_text("0\n")
    run([python, str(d / "run_online_profile_reciprocal_fix.py")], "online_tf_profile.log")
    run([str(d / "build/training_benchmark"), str(checkpoint),
         str(root / "tests/diagnostics/original_tf_cuda_audit_20261013/input.bin"),
         str(folder / "phase_3")], "phase_3.log")
    (folder / "phase_3.exit").write_text("0\n")
    nsight = (d / "run_nsight_final.sh").read_text()
    assert "phase_5" in nsight
    # Prefix every output while keeping the complete legacy/candidate matrix.
    (d / "run_nsight_reciprocal_fix.sh").write_text(nsight.replace("phase_5", "reciprocal_fix_phase_5"))
    run(["bash", str(d / "run_nsight_reciprocal_fix.sh")], "nsight.log")
    (folder / "acceptance.exit").write_text("0\n")
except BaseException:
    (folder / "acceptance.exit").write_text("1\n")
    raise
