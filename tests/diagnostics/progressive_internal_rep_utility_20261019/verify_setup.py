"""渲染前补充核验隔离runtime SHA及S/L/G边界测试。"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
for scene in ['fr1_room','fr1_xyz','fr2_xyz']:
    p=ROOT/scene
    old=Path('/home/prmi/zwj/hierarchical-progressive-marginal-20261019')if scene=='fr1_room'else Path('/home/prmi/zwj/hierarchical-xyz-f1f2-mechanism-20261019')/scene
    if scene=='fr1_room':expected=json.loads((old/'artifact_manifest.json').read_text())
    else:expected={str(Path(k).relative_to(scene)):v['sha256']for k,v in json.loads((old.parent/'bundle_manifest.json').read_text())['files'].items()if k.startswith(scene+'/')}
    for lib in (p/'runtime/lib').glob('*'):
        rel=str(lib.relative_to(p));assert rel in expected,(scene,rel)
        actual=sha(lib);assert actual==expected[rel],('STOP_RUNTIME_SHA',scene,rel)
        checks.append(dict(scene=scene,path=rel,sha256=actual,exact=True))
sys.path.insert(0,str(ROOT/'fr1_room'))
from construct import construct,refill,rank
from progressive_ranker import Node
n0=Node(0,(0,0,0,''),1,np.array([0,1,2,3]),[1]);n1=Node(1,(0,0,0,'0'),2,np.array([0,1,2,3]));n2=Node(2,(1,0,0,''),1,np.array([4,5]))
nodes={0:n0,1:n1,2:n2};P=np.array([0,4,1,2,5,3]);C=np.array([3,4,2,1,5,0]);ev=[dict(index=int(i),node=0 if i==0 else 1 if i<4 else 2,phase='internal_rep'if i==0 else'final_rep',rank=j+1,local_position=1)for j,i in enumerate(P)]
L,t=construct(P,C,ev,nodes,2);np.testing.assert_array_equal(L,[3,4]);assert t[0]['same_subtree']and t[0]['success']
G=refill(P,C,np.array([1,0,0,0,0,0],bool),2);np.testing.assert_array_equal(G,[3,4])
small={0:Node(0,(0,0,0,''),1,np.array([0]),[1])};L,t=construct(np.array([0,1]),np.array([1,0]),[dict(index=0,node=0,phase='internal_rep',rank=1,local_position=1),dict(index=1,node=0,phase='final_rep',rank=2,local_position=1)],small,1);np.testing.assert_array_equal(L,[0]);assert not t[0]['success']
for K in range(1,7):
    a,t=construct(P,C,ev,nodes,K);z,u=construct(P,C,ev,nodes,K);np.testing.assert_array_equal(a,z);assert t==u and len(a)==len(set(a))==K
(ROOT/'setup_checks.json').write_text(json.dumps(dict(all_passed=True,runtime_checks=checks,synthetic=['same_subtree_proxy','missing_descendant_retains_original','G_C_rank_refill','same_K_unique','repeat_exact','all_candidate_ids']),indent=2)+'\n')
print('SETUP_EXACT_PASS',len(checks),flush=True)
