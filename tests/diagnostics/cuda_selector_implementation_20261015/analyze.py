"""汇总本次实际 parity/benchmark 证据；缺失项保持未验收。"""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
summary = {"production_accepted": False, "parity": {}, "benchmark": []}
for path in sorted(root.glob("*.csv")):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        continue
    if "check" in rows[0]:
        failed = [r for r in rows if r["passed"] != "1"]
        hard = [r for r in failed if any(k in r["check"] for k in ("/hard", "/compact", "/rng_hard"))]
        steps = [int(r["check"].split("/")[0][5:]) for r in rows if r["check"].startswith("step_")]
        summary["parity"][path.name] = {
            "checks": len(rows), "failures": len(failed), "hard_or_compact_failures": len(hard),
            "last_recorded_step": max(steps, default=0), "first_failures": failed[:40],
            "hard_failures": hard[:20],
            "max_abs": max(float(r["max_abs"]) for r in rows),
        }
    elif "mean_ms" in rows[0]:
        summary["benchmark"].extend(dict(source=path.name, **row) for row in rows)
(root / "acceptance_status.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({name: {k: v for k, v in item.items() if k not in ("first_failures", "hard_failures")}
                  for name, item in summary["parity"].items()}, ensure_ascii=False, indent=2))
