"""串行完成补充Mapping、细分benchmark与完整评估，避免GPU任务互相干扰。"""
import subprocess
import time
from pathlib import Path

diagnostics = Path(__file__).resolve().parent
python = "/home/prmi/anaconda3/envs/zwj/bin/python"

def run(command, filename):
    with (diagnostics / filename).open("w") as log:
        subprocess.run(command, cwd=diagnostics, stdout=log,
                       stderr=subprocess.STDOUT, check=True)

try:
    while not (diagnostics / "guard_off_online.exit").exists():
        time.sleep(2)
    assert (diagnostics / "guard_off_online.exit").read_text().strip() == "0"
    with (diagnostics / "guard_off_desk_start1000_memory_sampler.log").open("w") as log:
        sampler = subprocess.Popen([python, str(diagnostics / "sample_online_memory.py"),
                                    "--protocol", "desk_start1000"], stdout=log,
                                   stderr=subprocess.STDOUT)
        run([python, "run_online.py", "--protocol", "desk_start1000"], "guard_off_desk_start1000_batch.log")
        assert sampler.wait(timeout=10) == 0
    if not (diagnostics / "mlp_backward.exit").exists() or (diagnostics / "mlp_backward.exit").read_text().strip() != "0":
        run(["bash", "run_mlp_backward.sh"], "mlp_backward.log")
    run([python, "run_online_evaluation.py"], "guard_off_evaluation_batch.log")
    run([python, "run_online_evaluation.py", "--protocol", "desk_start1000"], "guard_off_desk_start1000_evaluation_batch.log")
    run([python, "collect_online_results.py"], "guard_off_online_summary.log")
    run([python, "collect_online_results.py", "--protocol", "desk_start1000"], "guard_off_desk_start1000_online_summary.log")
    (diagnostics / "guard_off_remaining_acceptance.exit").write_text("0\n")
except BaseException:
    (diagnostics / "guard_off_remaining_acceptance.exit").write_text("1\n")
    raise
