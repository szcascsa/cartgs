"""只读选集诊断；覆盖差异本身不是质量因果证据。"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
rows=[]
for p in sorted((ROOT/'frozen_fixtures').glob('*/selected_sets.npz')):
    sets=np.load(p)
    for ratio,lo,hi in [(.025,.01,.05),(.075,.05,.1),(.125,.1,.15)]:
        s=sets[f'{ratio}_same_k'];a=sets[f'{lo}_same_k'];b=sets[f'{hi}_same_k']
        rows.append(dict(source=p.parent.name,ratio=ratio,K=len(s),in_lower=len(np.intersect1d(s,a)),
            in_upper=len(np.intersect1d(s,b)),in_both=len(np.intersect1d(s,np.intersect1d(a,b))),in_neither=len(np.setdiff1d(s,np.union1d(a,b)))))
data=pd.DataFrame(rows);data.to_csv(ROOT/'endpoint_selection_coverage.csv',index=False);print(data.to_string(index=False))
