"""只用冻结XYZ构建fixed sparse grid；不读取hierarchy、baseline score或quality。"""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'spatial-mlp-implicit-si-20261020'
SCENES=['fr1_room','fr1_xyz','fr2_xyz']
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def tsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(name,rows):
    with open(ROOT/name,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def build(xyz):
    target=2048
    seed=float(np.ptp(xyz,axis=0).max()/np.cbrt(target));assert seed>0
    evaluated={}
    def evaluate(h):
        if h not in evaluated:evaluated[h]=len(np.unique(np.floor(xyz/h).astype(np.int64),axis=0))
        return evaluated[h]
    current=seed;count=evaluate(current);low=high=current
    if count>target:
        for _ in range(256):
            low=current;high=current*2;current=high;count=evaluate(current)
            if count<=target:break
        else:raise RuntimeError('No occupancy bracket')
    elif count<target:
        for _ in range(256):
            low=current*.5;high=current;current=low;count=evaluate(current)
            if count>=target:break
        else:raise RuntimeError('No occupancy bracket')
    else:low=current*.5;high=current*2;evaluate(low);evaluate(high)
    def sample(lo,hi):
        for i in range(128):
            h=lo if i==0 else hi if i==127 else math.exp(math.log(lo)+(math.log(hi)-math.log(lo))*i/127)
            evaluate(h)
    sample(low,high)
    best=lambda:min(evaluated,key=lambda h:(abs(evaluated[h]-target),-h))
    ordered=sorted(evaluated);idx=ordered.index(best());sample(ordered[max(0,idx-1)],ordered[min(len(ordered)-1,idx+1)])
    h=best();keys,inverse=np.unique(np.floor(xyz/h).astype(np.int64),axis=0,return_inverse=True)
    occupancy=np.bincount(inverse,minlength=len(keys)).astype(np.int64)
    centers=(keys.astype(np.float64)+.5)*h;lo=centers.min(0);hi=centers.max(0);assert (hi>lo).all()
    normalized=2*(centers-lo)/(hi-lo)-1
    return dict(h=h,h_seed=seed,keys=keys,cell_id=inverse.astype(np.int64),occupancy=occupancy,
                centers=centers,normalized=normalized,normalization_min=lo,normalization_max=hi,evaluated=evaluated)

def run():
    ROOT.mkdir(exist_ok=True);metadata=[];occupancy_rows=[];all_arrays={};checks={}
    for name in SCENES:
        here=ROOT/name;here.mkdir(exist_ok=True)
        data=np.load(OLD/name/'ranking_inputs.npz');xyz=data['xyz'].astype(np.float64);ids=data['row_ids'];assert np.isfinite(xyz).all()
        grid=build(xyz);repeat=build(xyz)
        for key in ['h','keys','cell_id','occupancy','centers','normalized']:np.testing.assert_array_equal(grid[key],repeat[key])
        n=len(ids);m=len(grid['keys']);assert grid['occupancy'].sum()==n and grid['occupancy'].min()>0 and m<n
        assert grid['cell_id'].min()==0 and grid['cell_id'].max()==m-1
        np.testing.assert_array_equal(grid['keys'][grid['cell_id']],np.floor(xyz/grid['h']).astype(np.int64))
        arrays=dict(gaussian_source_row=ids,cell_id=grid['cell_id'],cell_key=grid['keys'],cell_center=grid['centers'],
                    occupancy=grid['occupancy'],normalized_cell_center=grid['normalized'],h=np.array(grid['h']),
                    normalization_min=grid['normalization_min'],normalization_max=grid['normalization_max'])
        np.savez_compressed(here/'cell_mapping.npz',**arrays)
        for key,value in arrays.items():all_arrays[name+'_'+key]=value
        with open(here/'grid.bin','wb')as f:
            np.array([n,m],dtype=np.int64).tofile(f);grid['cell_id'].tofile(f);grid['normalized'].tofile(f);grid['occupancy'].tofile(f)
        info=dict(scene=name,N=n,M=m,M_over_N=m/n,h=grid['h'],h_seed=grid['h_seed'],world_origin=[0,0,0],target_occupied=2048,
                  h_rule='XYZ max span / cbrt(2048) seed; occupancy bracket;128 log probes +128 fixed refinement; coarser h wins count ties',
                  normalized_center_min=grid['normalization_min'].tolist(),normalized_center_max=grid['normalization_max'].tolist(),
                  Gaussian_XYZ_SHA=tsha(xyz),source_rows_SHA=tsha(ids),mapping_SHA=tsha(grid['cell_id']),normalized_center_SHA=tsha(grid['normalized']),
                  occupancy_min=int(grid['occupancy'].min()),occupancy_median=float(np.median(grid['occupancy'])),occupancy_p90=float(np.quantile(grid['occupancy'],.9)),occupancy_max=int(grid['occupancy'].max()),
                  construction_inputs=['frozen valid AND observed XYZ only'],deterministic_repeat_exact=True)
        save(here/'normalization.json',info);metadata.append({k:v for k,v in info.items()if not isinstance(v,(list,dict))})
        write_search=[dict(h=h,occupied=count)for h,count in sorted(grid['evaluated'].items())]
        with open(here/'occupancy_search.csv','w',newline='')as f:w=csv.DictWriter(f,fieldnames=['h','occupied']);w.writeheader();w.writerows(write_search)
        for c,key in enumerate(grid['keys']):occupancy_rows.append(dict(scene=name,cell_id=c,key_x=int(key[0]),key_y=int(key[1]),key_z=int(key[2]),occupancy=int(grid['occupancy'][c]),center_x=float(grid['centers'][c,0]),center_y=float(grid['centers'][c,1]),center_z=float(grid['centers'][c,2])))
        checks[name]=info;print('GRID',name,n,m,grid['h'],flush=True)
    write('grid_metadata.csv',metadata);write('occupancy_statistics.csv',occupancy_rows)
    np.savez_compressed(ROOT/'cell_mapping.npz',**all_arrays);save(ROOT/'grid_correctness.json',dict(all_passed=True,scenes=checks))

if __name__=='__main__':run()
