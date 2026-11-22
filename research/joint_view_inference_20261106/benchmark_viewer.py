"""完整原生Viewer处理函数（含指标和FP32封包）的相同请求对照。"""
from common import *
import importlib.util
import statistics
import struct


def parse(packet):
    length=struct.unpack('<I',packet[:4])[0]
    return json.loads(packet[4:4+length]),packet[4+length:]


@torch.no_grad()
def main():
    spec=importlib.util.spec_from_file_location('viewer_original_benchmark',ROOT/'original/python/budget_viewer.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module.BudgetScene(CHECKPOINTS['both_li'],DATASET,2000,0,None,'pytorch')
    optimized=BudgetScene(CHECKPOINTS['both_li'],DATASET,2000,0,None,'cuda_graph');results=[]
    for budget in [2000,5000]:
        for full in [False,True]:
            values={'original':[],'optimized':[]}
            for frame in [0,198,396,594,791]:
                request=dict(frame=frame,budget=budget,view_conditioned=True,reference_full=full,request_id=12)
                a,pa=parse(original.render(request,native=True));b,pb=parse(optimized.render(dict(request,measure_inference_stages=False),native=True))
                assert pa==pb
                for key in ['active_gaussians','projected_gaussians','ground_truth_psnr','full_reference_psnr']:assert a['stats'][key]==b['stats'][key]
                for repeat in range(24):
                    order=list(values)
                    if repeat%2:order.reverse()
                    for name in order:
                        start=time.perf_counter()
                        packet=(original.render(request,native=True) if name=='original' else
                            optimized.render(dict(request,measure_inference_stages=False),native=True))
                        if repeat>=4:values[name].append((time.perf_counter()-start)*1000)
            row=dict(budget=budget,reference_full=full,samples_per_backend=100,
                original_ms=statistics.median(values['original']),optimized_ms=statistics.median(values['optimized']),
                speedup=statistics.median(values['original'])/statistics.median(values['optimized']),raw_ms=values)
            results.append(row);print(json.dumps({k:v for k,v in row.items() if k!='raw_ms'}),flush=True)
    save(ROOT/'reports/VIEWER_HANDLER_BENCHMARK.json',dict(passed=True,results=results,pixel_and_metric_parity=True,
        scope='含相机、Anchor统计、指标、Full参考开关、原生FP32封包；不含HTTP/客户端/显示器',no_lossy_conversion=True))


if __name__=='__main__':main()
