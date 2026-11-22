"""独立部署推理入口：不训练，不保存图片，导出轨迹数值与真实墙钟。"""
import argparse
import csv
from pathlib import Path
import time

import numpy as np
import torch
from budget_viewer import load_views
from deployment_runtime import DeploymentRenderer
from utils import MiniCam, focal2fov


def main():
    parser=argparse.ArgumentParser(description='训练完成后的冻结FP32部署推理')
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--dataset',type=Path)
    request=parser.add_mutually_exclusive_group(required=True)
    request.add_argument('--budget',type=int)
    request.add_argument('--ratio',type=float)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--frames',type=int,default=0,help='0为全部轨迹帧')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.frames<0:parser.error('frames must be nonnegative')
    if args.output.exists():parser.error('output already exists')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    checkpoint=args.checkpoint.resolve()
    if (checkpoint/'ply').is_dir():checkpoint=checkpoint/'ply'
    calibration,views=load_views(checkpoint,args.dataset)
    runtime=DeploymentRenderer.from_checkpoint(checkpoint,args.seed)
    request=dict(budget=args.budget) if args.budget is not None else dict(ratio_budget=args.ratio)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',newline='',encoding='utf-8') as f,torch.inference_mode():
        writer=csv.DictWriter(f,fieldnames=['frame','candidate_N','K','ratio','shortfall','prepare_ms','prepare_render_ms'])
        writer.writeheader()
        for frame,v in enumerate(views[:args.frames or None]):
            c=calibration
            camera=MiniCam(c['width'],c['height'],focal2fov(c['fx'],c['width']),
                focal2fov(c['fy'],c['height']),np.linalg.inv(np.asarray(v['pose'])))
            torch.cuda.synchronize();start=time.perf_counter()
            result=runtime.render(camera,**request)
            torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
            stats=result['stats']
            writer.writerow(dict(frame=frame,candidate_N=stats['candidate_gaussians'],K=len(result['ids']),
                ratio=stats['ratio'],shortfall=stats['budget_shortfall'],prepare_ms=stats['prepare_ms'],prepare_render_ms=elapsed))
    print(f'已完成部署推理：{args.output}',flush=True)


if __name__=='__main__':main()
