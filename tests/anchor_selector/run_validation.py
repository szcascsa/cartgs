"""在隔离构建目录运行功能验证并保存完整日志；不做质量调参。"""
import json
import os
from pathlib import Path
import subprocess
import sys

repo = Path(sys.argv[1]).resolve()
output = repo.parent / "validation"
output.mkdir(exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS="2")
results = {}

def run(name, command):
    result = subprocess.run([str(x) for x in command], cwd=output, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (output / (name+".log")).write_text(result.stdout)
    results[name] = dict(exit_code=result.returncode, passed=result.returncode == 0)
    for line in result.stdout.splitlines():
        if line.startswith('{"passed":'):
            results[name]["measurements"] = json.loads(line)
    (output / "correctness.json").write_text(json.dumps(results, indent=2))
    print(name, result.returncode, flush=True)
    if result.returncode:
        print(result.stdout[-6000:])
        raise SystemExit(result.returncode)

for device in ("cpu", "cuda"):
    run("anchor_selector_"+device, [repo/"bin/anchor_selector_test", device])
    run("python_parity_"+device, [sys.executable, repo/"tests/anchor_selector/python_parity.py", output, device])
run("anchor_mapping", [repo/"bin/anchor_mapping_test", "--parity"])
run("legacy_mapping", [repo/"bin/ultrafast_mapping_test", "--parity"])
run("anchor_change_distill", [repo/"bin/anchor_change_distill_test"])
run("legacy_change_distill", [repo/"bin/change_distill_test"])
print(json.dumps(results, indent=2))
