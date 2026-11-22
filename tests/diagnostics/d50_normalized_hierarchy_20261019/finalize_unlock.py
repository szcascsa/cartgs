"""用实际node AABB补齐P0/P32 unlock物理尺度；不改emit或ranking。"""
from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parent
for s in ['fr1_room','fr1_xyz','fr2_xyz']:
    out=ROOT/s;d50=json.loads((out/'new_tree/hierarchy_telemetry.json').read_text())['d50']
    for arm,tag in [('P0','old'),('P32','new')]:
        bounds={int(a['node']):a for a in csv.DictReader(open(out/(tag+'_tree/hierarchy_tree.csv')))}
        f=out/(arm+'_unlock_trace.csv');records=list(csv.DictReader(open(f)))
        for a in records:
            n=bounds[int(a['node'])];side=[float(n[k+'max'])-float(n[k+'min'])for k in 'xyz'];h=json.loads((out/(tag+'_tree/hierarchy_telemetry.json')).read_text())['h_root']/(2**(int(a['level'])-1))
            assert all(abs(v-h)<=1e-10*max(1,h)for v in side);a['side_length']=side[0];a['side_over_d50']=side[0]/d50
        with open(f,'w',newline='')as stream:w=csv.DictWriter(stream,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
print('ACTUAL_AABB_UNLOCK_SCALE_PASS')
