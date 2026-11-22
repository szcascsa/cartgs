"""CPU Workspace/错误恢复/共享实例/预算边界和并发请求的行为检查。"""
from cpu_study import *
from concurrent.futures import ThreadPoolExecutor

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);source=load_scene('Replica__office2__rgbd');ref=source['reference']
    engine=CpuRenderer(source['model'],source['pc'],ref.noise,6,selector_prefix=True,factored=True)
    cam=camera(source['records'][len(source['records'])//2]);checks=[]
    for budget in [0,1,3,7,8,9,31,32,33,63,64,65,127,128,129,255,256,257,511,512,513,1000,5000,100000]:
        packet,ids,stats=engine.prepare(cam,budget=budget);assert len(ids)==min(budget,stats['eligible_candidates'])
        assert (ids[1:]>ids[:-1]).all();checks.append(dict(case='strict_budget',value=budget,K=len(ids),passed=True))
    # caller buffer尾部哨兵、结果独立、冻结快照隔离。
    size=engine.total*engine.width;host=torch.full((size+64,),-317.25);index=torch.full((engine.total+8,),-12345,dtype=torch.long)
    packet,ids,stats=engine.prepare(cam,budget=1000,output=host[:-64],id_output=index[:-8]);frozen=packet['inputs']['means3D'].clone()
    assert (host[-64:]==-317.25).all() and (index[-8:]==-12345).all();checks.append(dict(case='workspace_tail_canary',value=size,K=len(ids),passed=True))
    independent,kept,_=engine.prepare(cam,budget=1000)
    for _ in range(8):engine.prepare(cam,ratio_budget=.7)
    assert torch.equal(independent['inputs']['means3D'],frozen)
    checks.append(dict(case='returned_output_independence',value=8,K=len(kept),passed=True))
    original=source['pc']._xyz.clone();source['pc']._xyz.add_(100)
    changed,_,_=engine.prepare(cam,budget=1000);assert torch.equal(changed['inputs']['means3D'],frozen);source['pc']._xyz.copy_(original)
    checks.append(dict(case='frozen_map_isolation',value=1,K=len(ids),passed=True))
    for value in [float('nan'),float('inf'),-1.,1.01]:
        try:engine.prepare(cam,ratio_budget=value)
        except (RuntimeError,ValueError):checks.append(dict(case='invalid_ratio_rejected',value=str(value),K=0,passed=True))
        else:raise AssertionError(value)
    bad=copy.copy(cam);bad.camera_center=cam.camera_center.clone();bad.camera_center[0]=float('nan')
    try:engine.prepare(bad,budget=100)
    except RuntimeError:checks.append(dict(case='nonfinite_camera_rejected',value=1,K=0,passed=True))
    else:raise AssertionError('Accepted nonfinite camera')
    after,_,_=engine.prepare(cam,budget=1000);assert torch.equal(after['inputs']['means3D'],frozen)
    # 相同实例的C++互斥保证请求原子性；不同调用输出不共享。
    def call_task(i):
        with torch.no_grad():return engine.prepare(cam,budget=[1,31,128,1000][i%4])
    with ThreadPoolExecutor(max_workers=3) as executor:results=list(executor.map(call_task,range(12)))
    for i,(packet,ids,stats) in enumerate(results):assert len(ids)==[1,31,128,1000][i%4]
    checks.append(dict(case='concurrent_request_serialization',value=12,K=0,passed=True))
    write_csv(out/'checks.csv',checks);save(out/'SUMMARY.json',dict(passed=True,checks=len(checks),cuda_initialized=torch.cuda.is_initialized(),binary_sha256=sha(extension().__file__)))
    print('CPU_BOUNDARIES_COMPLETE',len(checks),flush=True)

if __name__=='__main__':main()
