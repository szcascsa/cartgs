"""无损FP32原生封包逐字节验证及同工作量性能测量。"""
from common import *
from budget_viewer import native_frame_packet
import statistics
import struct
import importlib.util

spec=importlib.util.spec_from_file_location('original_budget_viewer',ROOT/'original/python/budget_viewer.py')
original_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(original_module)
original_packet=original_module.native_frame_packet


def unpack(packet):
    size=struct.unpack('<I',packet[:4])[0]
    return json.loads(packet[4:4+size]),packet[4+size:]


@torch.no_grad()
def main():
    s=scene();engine=FrozenResidualInference(s.model,s.gaussians);checks=[];bench=[]
    generator=torch.Generator(device='cuda').manual_seed(91)
    for height,width in [(1,1),(17,31),(480,640),(680,1200)]:
        image=torch.randn((3,height,width),device='cuda',generator=generator)
        flat=image.reshape(-1)
        flat[:min(8,len(flat))]=image.new_tensor([0.,-0.,1.,-1.,float('inf'),-float('inf'),float('nan'),1e-40])[:min(8,len(flat))]
        ref=torch.rand((3,height,width),device='cuda',generator=generator)
        for second in [None,ref]:
            a=original_packet(image,second,dict(stats={},request_id=23))
            b=native_frame_packet(image,second,dict(stats={},request_id=23),staging=engine)
            assert a==b,(height,width,second is None)
            assert a==native_frame_packet(image,second,dict(stats={},request_id=23))
            checks.append(dict(height=height,width=width,reference=second is not None,packet_bytes=len(a),bitwise=True))
        # Returned Python bytes must survive later workspace reuse or resizing.
        saved=b;engine.display_bytes(image*.5,None);assert b==saved
    cam=camera(s,396);renderer=FixedBudgetRenderer(s.model,s.gaussians,s.config,0,'cuda_graph')
    data,_,_=renderer.prepare(cam,2000,view_conditioned=True,measure_time=False)
    image=render_prepared(cam,data,s.background)['render'];ref=s.ground_truth(396)
    for second in [None,ref]:
        times={'original':[],'join_only':[],'fused':[]}
        for repeat in range(100):
            names=list(times)
            if repeat%2:names.reverse()
            for name in names:
                torch.cuda.synchronize();start=time.perf_counter()
                packet=(original_packet(image,second,dict(stats={},request_id=23)) if name=='original' else
                    native_frame_packet(image,second,dict(stats={},request_id=23),staging=engine if name=='fused' else None))
                if repeat>=20:times[name].append((time.perf_counter()-start)*1000)
        bench.append(dict(reference=second is not None,original_ms=statistics.median(times['original']),
            join_only_ms=statistics.median(times['join_only']),fused_ms=statistics.median(times['fused']),
            speedup=statistics.median(times['original'])/statistics.median(times['fused']),
            join_only_speedup=statistics.median(times['original'])/statistics.median(times['join_only'])))
    # Full Viewer handler retains image/metadata, reference cache and current conditions.
    original=BudgetScene(CHECKPOINTS['both_li'],DATASET,2000,0,None,'pytorch')
    fast=BudgetScene(CHECKPOINTS['both_li'],DATASET,2000,0,None,'cuda_graph')
    for frame in [0,396,791]:
        for with_reference in [False,True]:
            request=dict(frame=frame,budget=2000,view_conditioned=True,reference_full=with_reference,request_id=9)
            ha,pa=unpack(original.render(request,native=True));hb,pb=unpack(fast.render(dict(request,measure_inference_stages=False),native=True))
            assert pa==pb
            for key in ['active_gaussians','candidate_gaussians','projected_gaussians','ground_truth_psnr','full_reference_psnr']:
                assert ha['stats'][key]==hb['stats'][key],key
            checks.append(dict(viewer_frame=frame,reference_full=with_reference,packet_pixels_bitwise=True,quality_metrics_identical=True))
    save(ROOT/'reports/DISPLAY_VALIDATION.json',dict(passed=True,checks=checks,benchmarks=bench,
        buffer_limit_bytes=64*1024**2,cache_entries=1,transport_dtype='little-endian FP32 RGB',lossy_conversion=False))
    print(json.dumps(bench),flush=True)


if __name__=='__main__':main()
