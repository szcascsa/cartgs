"""补充无隐式原版同步的连续提交、同地址输入更新和256 Tile任务边界。"""
from harness import *
from final_benchmark import native_modules
from final_edges import packet as make_packet
from runtime import SparseRenderer
from graph_runtime import GraphRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False)
    config=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text());native_modules(config)
    c=SparseRenderer(**config['regular']);g=GraphRenderer(**config['graph']);rows=[];torch.manual_seed(57103)
    cam=MiniCam(193,129,1.2,1.,np.eye(4));bg=torch.tensor([.13,-0.,.76],device='cuda')
    cases=[]
    for index in range(48):
        p=make_packet([1,31,127,255,256,257,511,512,513,1025,2049,4097][index%12],
            ['random','overlap','huge','thin'][index%4])
        cases.append((p,render_prepared(cam,p,bg,return_radii=True)))
    g.precompile(4097,cam,bg)
    c.render(cam,cases[-1][0],bg);torch.cuda.synchronize()
    pending=[]
    # oracle已在前面全部计算。这里不调用原入口、不比较Tensor、不主动同步。
    for index,(p,ref) in enumerate(cases):
        for name,engine in [('C',c),('G',g)]:pending.append((f'burst_dynamic_K/{index}',name,engine.render(cam,p,bg),ref))
    torch.cuda.synchronize()
    def verify(values):
        for label,name,out,ref in values:
            row=dict(case=label,variant=name,K=len(out[1]),rgb_bitwise=bits_equal(out[0],ref['render']),radii_bitwise=bits_equal(out[1],ref['radii']))
            rows.append(row);assert row['rgb_bitwise'] and row['radii_bitwise'],row
    verify(pending)
    # 指针不变而内容每帧更新，验证计划没有偷用旧几何或背景。
    cases=[]
    for index in range(32):
        p=make_packet(257,['random','overlap','huge','thin'][index%4]);background=torch.tensor([index/32,.2,.7-index/64],device='cuda')
        cases.append((p,background,render_prepared(cam,p,background,return_radii=True)))
    destination=make_packet(257);background=bg.clone();pending=[];torch.cuda.synchronize()
    for index,(p,b,ref) in enumerate(cases):
        for key,value in p['inputs'].items():
            if isinstance(value,torch.Tensor):destination['inputs'][key].copy_(value)
        background.copy_(b)
        for name,engine in [('C',c),('G',g)]:pending.append((f'same_address_changed_values/{index}',name,engine.render(cam,destination,background),ref))
    torch.cuda.synchronize();verify(pending)
    # 横向整图footprint：实际bbox正好255/256/257个Tile，跨调度任务容量边界。
    for tiles in [255,256,257]:
        camera=MiniCam(16*tiles,7,1.2,1.,np.eye(4));p=make_packet(1,'huge')
        p['inputs']['means3D'].zero_();p['inputs']['means3D'][:,2]=2.;p['inputs']['opacities'].fill_(.1)
        ref=render_prepared(camera,p,bg,return_radii=True);out=c.render(camera,p,bg,True)
        assert int(out[5][0])==tiles,(tiles,out[5])
        verify([(f'footprint_task_boundary/{tiles}','C',out,ref),(f'footprint_task_boundary/{tiles}','G',g.render(camera,p,bg),ref)])
    write_csv(output/'checks.csv',rows);write_json(output/'SUMMARY.json',dict(passed=True,comparisons=len(rows),
        single_sync_per_burst=True,original_oracle_outside_burst=True,output_storage_retained=True,
        same_input_addresses_changed_every_frame=True,footprint_task_boundaries=[255,256,257]))
    print('ASYNC_LIFECYCLE_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
