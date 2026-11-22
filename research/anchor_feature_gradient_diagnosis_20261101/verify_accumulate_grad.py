"""只读取已保存快照，检查 Scatter hook 到 leaf.grad 是否新增位差。"""
from pathlib import Path
import csv
import gc
import hashlib
import json

import torch

root = Path(__file__).resolve().parent
torch.set_num_threads(2)
rows = []
for path in sorted((root / "snapshots").glob("d*_r*_*.pt")):
    module = torch.jit.load(str(path), map_location="cpu")
    for side in ("a", "b"):
        for table in range(6):
            scatter_key = f"scatter_{side}_{table}"
            if not hasattr(module, scatter_key):
                continue
            scatter = getattr(module, scatter_key)
            leaf = getattr(module, f"leaf_{side}_{table}")
            equal = bool(torch.equal(scatter.contiguous().view(torch.int32),
                                     leaf.contiguous().view(torch.int32)))
            rows.append(dict(snapshot=path.name, side=side, table=table,
                             elements=scatter.numel(), bitwise_equal=equal))
    del module, scatter, leaf
    gc.collect()

with (root / "ACCUMULATEGRAD_SELF_CHECK.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
result = dict(passed=bool(rows) and all(row["bitwise_equal"] for row in rows),
              snapshots=len(set(row["snapshot"] for row in rows)),
              checks=len(rows), different_checks=sum(not row["bitwise_equal"] for row in rows),
              note="同一分支 Scatter 输出与 leaf.grad 比较；没有重新执行 GPU 计算。")
(root / "ACCUMULATEGRAD_VERIFICATION.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n")
assert result["passed"], result

manifest = []
for path in sorted((root / "snapshots").glob("*.pt")):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    manifest.append(dict(file=str(path.relative_to(root)), bytes=path.stat().st_size,
                         sha256=digest.hexdigest()))
(root / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False), flush=True)
