from common import *
from baseline import previous_runtime
from large_scene import LARGE


@torch.no_grad()
def main():
    s=BudgetScene(LARGE,None,5000,0,None,'pytorch');a=previous_runtime(s);out=[]
    for i in np.linspace(0,len(s.views)-1,9,dtype=int):
        _,ids,stats=a.prepare(camera(s,int(i)),ratio_budget=.1);out.append(dict(frame=int(i),**stats))
    save(ROOT/'reports/LARGE_PROBE.json',dict(N=a.total,anchors=len(s.model.anchors.centers),frames=len(s.views),schema=a._base.joint_schema,views=out))
    print('LARGE_PROBE_PASS',a.total,len(s.views),flush=True)


if __name__=='__main__':main()
