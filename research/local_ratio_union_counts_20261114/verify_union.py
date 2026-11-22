"""独立核验32组轨迹并集计数、逐帧哈希、ID集合与原始均值。"""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np
from collect_union import ROOT,SOURCE,SOURCE_SHA,RATIOS,records,sha,write


def main():
    finished=json.loads((ROOT/'FINISHED.json').read_text())
    assert finished['complete'] and finished['groups']==32 and finished['ratios']==352
    source=records();lookup={(r['group'],float(r['ratio'])):r for r in source}
    counts=list(csv.DictReader((ROOT/'LOCAL_UNION_COUNTS.csv').open(encoding='utf-8-sig')))
    assert len(counts)==len({(r['group'],float(r['ratio'])) for r in counts})==352
    frames=0;hashes_verified=0;all_inputs=0
    for group in sorted({r['group'] for r in counts}):
        folder=ROOT/'runs'/group
        done=json.loads((folder/'COMPLETE.json').read_text())
        before=json.loads((folder/'INPUTS_BEFORE.json').read_text())
        after=json.loads((folder/'INPUTS_AFTER.json').read_text())
        assert done['complete'] and done['source_unchanged'] and before==after
        assert all(sha(p)==h for p,h in before.items())
        all_inputs+=len(before)
        rows=list(csv.DictReader((folder/'PER_VIEW_UNION.csv').open()))
        by=defaultdict(list)
        for row in rows:by[float(row['ratio'])].append(row)
        assert set(by)==set(RATIOS)
        views=int(done['frames']);frames+=views
        for ratio in RATIOS:
            item=next(r for r in counts if r['group']==group and float(r['ratio'])==ratio)
            original=lookup[group,ratio];view_rows=by[ratio]
            assert len(view_rows)==views==int(item['views'])==int(original['frames'])
            ids=np.load(folder/f'UNION_IDS_{original["ratio"]}.npz')['ids']
            assert ids.dtype==np.int64 and np.array_equal(ids,np.unique(ids))
            assert len(ids)==int(item['local_union_count'])
            assert len(ids)==0 or 0<=ids.min()<=ids.max()<int(original['N'])
            assert sha_ids(ids)==item['union_ids_sha256']
            history=json.loads((Path(original['source'])/'selection_hashes.json').read_text())
            assert [r['selection_sha256'] for r in view_rows]==history
            hashes_verified+=views
            assert [int(r['frame']) for r in view_rows]==list(range(views))
            total=0
            for row in view_rows:
                total+=int(row['new_unique'])
                assert total==int(row['union_count']) and 0<=int(row['new_unique'])<=int(row['K'])
            assert total==len(ids) and max(int(r['K']) for r in view_rows)<=len(ids)
            mean=sum(int(r['K']) for r in view_rows)/views
            assert abs(mean-float(item['local_mean_K']))<1e-9
            assert abs(mean-float(original['K_mean']))<1e-9
    assert hashes_verified==finished['selection_checks'] and frames==finished['views']
    assert sha(SOURCE)==SOURCE_SHA
    write(ROOT/'UNION_VERIFIED.json',dict(complete=True,groups=32,ratios=352,views=frames,
        historical_selection_hashes_verified=hashes_verified,input_hash_checks=all_inputs,
        source_unchanged=True,union_ids_unique=True,union_sizes_match=True,
        per_view_running_union_consistent=True,original_mean_K_reproduced=True,
        scope='每场景／相机模式／Ratio，所有已评价视角实际选中母Gaussian ID的并集',
        training_steps=0,render_calls=0,images_saved=0))
    print('UNION_VERIFIED',32,352,frames,hashes_verified,flush=True)


def sha_ids(ids):
    import hashlib
    return hashlib.sha256(ids.tobytes()).hexdigest()


if __name__=='__main__':main()
