"""性能报告解析测试；输入为合成日志，不能用作 GPU 性能结果。"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
script = root / "scripts" / "report_global_gi.py"
spec = importlib.util.spec_from_file_location("report_global_gi", script)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
assert report.nearest_p95(list(range(1, 101))) == 95
assert report.nearest_p95([]) is None

with tempfile.TemporaryDirectory(prefix="cartgs-sgi-report-test-", dir=root) as directory:
    temp = Path(directory)
    log = temp / "synthetic.log"
    log.write_text(
        "[Mapping PERF] iter=1000 cycles=1000 wall_ms(avg/p95/max)=20.0/30.0/40.0\n"
        "[Mapping PERF] iter=2000 cycles=1000 wall_ms(avg/p95/max)=40.0/50.0/60.0\n"
        "[GlobalGI] event=topology iteration=10 kind=append global_gi_topology_total_ms=1.5\n"
        "[GlobalGI] event=topology iteration=20 kind=prune global_gi_topology_total_ms=4.5\n"
        "[GlobalGI] event=full_refresh iteration=1000 global_gi_full_refresh_ms=70.5\n"
        "[GlobalGI] event=final iteration=2000 global_gi_cache_memory_mb=8.5 "
        "global_gi_valid_mature_gaussians=80 global_gi_guidance_enabled=1\n", encoding="utf-8")
    rows, topology, full, blocks = report.read_log(log)
    assert len(rows) == 4 and topology == [1.5, 4.5] and full == [70.5]
    assert sum(n * ms for n, ms in blocks) / sum(n for n, _ in blocks) == 30
    output = temp / "synthetic-report.md"
    subprocess.run([sys.executable, str(script), "--streaming-log", str(log),
                    "--output", str(output)], check=True, capture_output=True)
    text = output.read_text(encoding="utf-8")
    assert "30.0000" in text and "33.3333" in text and "8.5" in text
    assert output.with_suffix(".csv").exists()
    assert "未提供日志" in text
print("SGI report parser: PASS (synthetic data only)")
