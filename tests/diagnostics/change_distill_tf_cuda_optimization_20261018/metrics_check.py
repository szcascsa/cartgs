"""比较正式 10 warm runs 的完整精度 pre/post metrics；保留 audit 日志精度边界。"""
from pathlib import Path
import json
import math
import re
D=Path(__file__).resolve().parent
result={}
fields=['selector_pre','selector_post','tf_pre','tf_post']
for branch in ['tf','selector','both']:
    delta={k:[] for k in fields};audit=[]
    for direction in ['forward','reverse']:
        paths=[D/s/'raw'/('paired_'+direction+'_'+branch) for s in ['baseline_production','production']]
        rows=[[r for r in json.loads((p/'runs.json').read_text()) if r['label'].startswith('warm_')] for p in paths]
        assert len(rows[0])==len(rows[1])==5
        for a,b in zip(*rows):
            for k in fields:
                assert math.isfinite(a[k]) and math.isfinite(b[k])
                assert abs(a[k]-b[k])<=2e-5+2e-4*abs(b[k]),(branch,k,a[k],b[k])
                delta[k].append(abs(a[k]-b[k]))
        pattern=r'untouched_pre=([0-9eE+.,-]+) untouched_post=([0-9eE+.,-]+)'
        values=[re.findall(pattern,(p/'stdout.log').read_text()) for p in paths]
        audit.append({'direction':direction,'equal_at_logged_four_decimals':values[0]==values[1],'lines':len(values[0])})
    result[branch]={'max_abs':{k:max(v) for k,v in delta.items()},'warm_pairs':10,'allclose':True,'audit_log_comparison':audit}
(D/'production_metrics.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
