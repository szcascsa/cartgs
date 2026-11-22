"""只汇总Bank内探索证据，不打开nonBank。"""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent


def main():
    rows=[]
    for path in sorted((ROOT/'results').glob('*/*/*/*/complete.json')):
        v=json.loads(path.read_text());cfg=v['config']
        rows.append(dict(tier=v['tier'],scene=path.parts[-4],name=cfg['id'],seed=v['seed'],steps=v['steps'],
            parameters=v['parameters'],pure=v['pure_shared'],validation=v['validation_psnr'],train=v['validation_train'],
            unseen=v['validation_unseen'],seconds=v['seconds'],config=json.dumps(cfg,sort_keys=True)))
    f=pd.DataFrame(rows)
    f.to_csv(ROOT/'bank_screen_ledger.csv',index=False)
    print(f.to_string(index=False),flush=True)


if __name__=='__main__':main()
