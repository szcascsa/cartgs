"""默认JointReleaseRenderer(mode=182)：冻结Student推理，不保存渲染图片。"""
import argparse
import csv
from pathlib import Path
import time

import torch
from joint_release_renderer import JointReleaseRenderer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--dataset',type=Path)
    request=parser.add_mutually_exclusive_group(required=True)
    request.add_argument('--budget',type=int)
    request.add_argument('--ratio',type=float)
    request.add_argument('--full',action='store_true')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--student',type=Path,required=True,help='与当前冻结地图指纹匹配的Student checkpoint')
    parser.add_argument('--frames',type=int,default=0,help='0为全部轨迹帧')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.frames<0:parser.error('frames must be nonnegative')
    if args.budget is not None and args.budget<0:parser.error('budget must be nonnegative')
    if args.ratio is not None and not 0<=args.ratio<=1:parser.error('ratio must be finite and in [0,1]')
    if args.output.exists():parser.error('output already exists')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    checkpoint=args.checkpoint.resolve()
    if (checkpoint/'ply').is_dir():checkpoint=checkpoint/'ply'
    from budget_viewer import load_views
    calibration,views=load_views(checkpoint,args.dataset)
    if not views:parser.error('checkpoint has no camera trajectory')
    runtime=JointReleaseRenderer.from_checkpoint(checkpoint,args.seed,student_checkpoint=args.student)
    request=(dict(full=True) if args.full else dict(budget=args.budget) if args.budget is not None else dict(ratio_budget=args.ratio))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',newline='',encoding='utf-8') as f,torch.inference_mode():
        # 模块加载、静态投影和首帧Graph建立不混入稳态逐帧计时。
        runtime.precompile(calibration['width'],calibration['height'])
        for _ in range(3):runtime.render_pose(views[0]['pose'],calibration,**request)
        torch.cuda.synchronize()
        writer=csv.DictWriter(f,fieldnames=['frame','backend','rasterizer_mode','candidate_N','K','ratio','shortfall','prepare_render_ms'])
        writer.writeheader()
        for frame,v in enumerate(views[:args.frames or None]):
            torch.cuda.synchronize();start=time.perf_counter()
            result=runtime.render_pose(v['pose'],calibration,**request)
            torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
            stats=result['stats']
            writer.writerow(dict(frame=frame,backend=runtime.backend,rasterizer_mode=runtime.mode,
                candidate_N=stats['candidate_gaussians'],K=len(result['ids']),
                ratio=stats.get('ratio'),shortfall=stats['budget_shortfall'],prepare_render_ms=elapsed))
    print(f'已完成部署推理：{args.output}',flush=True)


if __name__=='__main__':main()
