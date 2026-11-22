"""单独测量分配，避免把分配跟踪器开销混入已冻结的speed benchmark。"""
import json,tracemalloc,time,subprocess
from pathlib import Path
import numpy as np
from reference import build
ROOT=Path(__file__).resolve().parent;x=dict(np.load(ROOT/'ranking_inputs.npz'));h=json.loads((ROOT/'population.json').read_text())['h']
tracemalloc.start();before=tracemalloc.take_snapshot();t=time.perf_counter();r,p=build(x['xyz'],x['radius'],x['raw_I'],x['row_ids'],h);current,peak=tracemalloc.get_traced_memory();after=tracemalloc.take_snapshot();stats=after.compare_to(before,'lineno');tracemalloc.stop()
out=dict(reference_traced_current_bytes=current,reference_traced_peak_bytes=peak,reference_live_allocation_count_diff=sum(s.count_diff for s in stats),reference_known_allocation_sites_count=sum(10+5*c['n']for c in p['cells']),instrumented_wall_ms=(time.perf_counter()-t)*1000,note='live snapshot difference is not total allocation traffic; explicit NumPy allocation count is an estimate; ranking RSS includes Python/NumPy runtime and is not comparable to isolated native RSS')
(ROOT/'allocation_profile.json').write_text(json.dumps(out,indent=2))
