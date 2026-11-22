from components import *


@torch.no_grad()
def main():
    ext=deployment_extension()
    for group in ['both_li','replica']:
        s=scene(group) if group!='replica' else BudgetScene(LARGE,None,5000,0,None,'pytorch');a=previous_runtime(s);b=a._base
        for ratio in [.01,.1,.7]:
            for frame in np.linspace(0,len(s.views)-1,8,dtype=int):
                rows,scores,ids,args,_,_=sample(s,a,int(frame),ratio);arguments=(b.selector_static,rows,b.owners,args[7],a._reference.noise,a._selector,ratio,bool(b.joint_schema),b.joint_schema)
                calls={'selector_v11':lambda:previous_extension().selector(*arguments),
                    'selector_optimized':lambda:ext._selector_tiled(*arguments),
                    'comp_v11':lambda:previous_extension()._compensate_tiled(*args,4) if len(ids)>=1024 else previous_extension().compensate(*args),
                    'comp_optimized':lambda:ext._compensate_tiled(*args,16) if len(ids)>=1024 else ext.compensate(*args)}
                for fn in calls.values():fn()
                for tag,fn in calls.items():
                    with torch.cuda.nvtx.range(f'node/{group}/{ratio}/{tag}'):fn()
                    torch.cuda.synchronize()
        del s,a;torch.cuda.empty_cache()


if __name__=='__main__':main()
