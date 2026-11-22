"""完全复用上一轮worker/metrics；只传入不同冻结ranking前缀。"""
import csv
from common import *
def write_csv(name,rows):
    with open(ROOT/name,'w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def evaluate_arms(scene,metrics,arm_orders,tag):
    info=read(ROOT/'input_manifest.json');inp=np.load(ROOT/'ranking_inputs.npz');ids=inp['row_ids'];N=len(ids);frames=info['evaluation_frames'];records=[];summaries=[];selected={}
    for arm,order in arm_orders.items():
        for ratio in read(ROOT/'protocol.json')['ratios']:
            K=int(ratio*N+.5);rows=np.sort(ids[order[:K]]);assert len(rows)==K
            selected[f'{arm}_{ratio}']=rows;case=[];save(ROOT/'status.json',dict(stage=tag,arm=arm,ratio=ratio,completed=len(summaries)))
            for start in range(0,len(frames),4):
                indices=frames[start:start+4];out=scene.call(dict(method='images',rows=rows.tolist(),ratio=ratio,frames=indices,apply_tf=True))
                values=metrics.all(torch.from_numpy(np.array(scene.mapping[:len(indices)])).cuda(),indices)
                for j,f in enumerate(indices):case.append(dict(arm=arm,ratio=ratio,K=K,frame_index=f,KF_id=info['camera_ids'][f],photo=out['photo'][j],psnr=float(values[j,0]),ssim=float(values[j,1]),lpips=float(values[j,2])))
            records.extend(case);summary=dict(arm=arm,ratio=ratio,K=K,views=len(case),**{m:float(np.mean([r[m]for r in case]))for m in ['photo','psnr','ssim','lpips']});summaries.append(summary)
            write_csv(tag+'_per_view.csv',records);write_csv(tag+'_quality.csv',summaries);print('QUALITY',json.dumps(summary),flush=True)
    np.savez_compressed(ROOT/(tag+'_selected.npz'),**selected)
    return records,summaries
