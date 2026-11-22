"""审计完成前逐文件核对生产源码；只写审计清单。"""
import hashlib,json,subprocess
from pathlib import Path
dest=Path(__file__).resolve().parent;repo=dest.parents[2]
before=json.loads((dest/'source_hashes_before.json').read_text())
after={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in before}
changes=[name for name in before if before[name]!=after[name]]
checkpoint=Path(json.loads((dest/'inventory.json').read_text())['checkpoint'])
artifacts=['selector.pt','selector_config.yaml','transform_field.pt','transform_state.json']
artifacts += [str(p.relative_to(checkpoint)) for p in checkpoint.rglob('*.ply')]
artifacts += [name for name in ['transform_optimizer.pt','protection_state.json'] if (checkpoint/name).is_file()]
binaries=[repo/'lib/libgaussian_mapper.so',repo/'lib/libcuda_rasterizer.so',dest/'build/native_benchmark',dest/'build/gradient_paths',dest/'build/optimizer_benchmark',dest/'build/isolated_costs']
def cmd(s):
    p=subprocess.run(s,shell=True,capture_output=True,text=True,cwd=repo)
    return dict(exit=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())
payload=dict(changed_production_sources=changes,source_hashes=after,checkpoint_hashes={name:hashlib.sha256((checkpoint/name).read_bytes()).hexdigest() for name in artifacts},
    binary_hashes={str(p.relative_to(repo)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,mtime=p.stat().st_mtime) for p in binaries},
    final_git={s:cmd('git '+s) for s in ['rev-parse HEAD','rev-parse --show-toplevel','status --short -- .']},
    gpu=cmd('nvidia-smi --query-gpu=name,utilization.gpu,memory.used,clocks.sm,clocks.mem --format=csv'),disk=cmd('df -h .'))
(dest/'source_verification.json').write_text(json.dumps(payload,indent=2))
print(json.dumps(dict(changed_sources=changes,binary_count=len(binaries)),indent=2))
