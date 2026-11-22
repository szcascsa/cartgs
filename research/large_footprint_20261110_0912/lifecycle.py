"""最终候选的容量、跨Stream、输出独立、足迹任务和异常边界。"""
from study_common import *
from footprint_runtime import Renderer,MODES


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--modes',default='1,7')
    parser.add_argument('--sanitizer',action='store_true');args=parser.parse_args();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    native_modules(BASELINE_CONFIG);engines={MODES[k]:Renderer(k) for k in map(int,args.modes.split(','))};rows=[];torch.manual_seed(583021)
    camera=MiniCam(97,65,1.2,1.,np.eye(4));bg=torch.tensor([.13,-0.,.76],device='cuda')
    def compare(label,cam,p,background):
        ref=render_prepared(cam,p,background,return_radii=True)
        for name,engine in engines.items():
            result=engine.render(cam,p,background);row=dict(case=label,variant=name,K=len(result[1]),rgb_bitwise=bits_equal(result[0],ref['render']),radii_bitwise=bits_equal(result[1],ref['radii']))
            rows.append(row);assert row['rgb_bitwise'] and row['radii_bitwise'],row
    for P in [0,1,31,32,33,255,256,257,511,512,513,1023,1024,1025,4097]:
        for color,covariance in [(False,False),(True,False),(False,True),(True,True)]:
            compare(f'formats/{P}/{color}/{covariance}',camera,synthetic_packet(P,'huge',color,covariance),bg)
    # 强制超过256项短列表，同时Alpha足够低避免所有像素提前结束。
    for P in [255,256,257,513]:compare('list_boundary/'+str(P),camera,synthetic_packet(P,'overlap'),bg)
    # 真正的无中间同步：先完成所有oracle，然后连续提交变化形状，保存所有输出。
    cases=[]
    for index,P in enumerate([1,257,513,31,1025,4097,33,129]):
        cam=MiniCam(97+index,65+index%2,1.2,1.,np.eye(4));p=synthetic_packet(P,'thin' if index%2 else 'huge')
        cases.append((cam,p,render_prepared(cam,p,bg,return_radii=True)))
    torch.cuda.synchronize();pending=[]
    for cam,p,ref in cases:
        for name,engine in engines.items():pending.append((name,engine.render(cam,p,bg),ref))
    torch.cuda.synchronize()
    for name,result,ref in pending:
        assert bits_equal(result[0],ref['render']) and bits_equal(result[1],ref['radii'])
        rows.append(dict(case='async_shape_and_output_lifecycle',variant=name,K=len(result[1]),rgb_bitwise=True,radii_bitwise=True))
    # 多Stream显式输入依赖，错用旧Stream实例必须拒绝。
    p=synthetic_packet(257,'huge');ref=render_prepared(camera,p,bg,return_radii=True);ready=torch.cuda.Event();ready.record()
    for stream in [torch.cuda.Stream(),torch.cuda.Stream()]:
        with torch.cuda.stream(stream):
            stream.wait_event(ready);local={MODES[k]:Renderer(k) for k in map(int,args.modes.split(','))}
            values={name:e.render(camera,p,bg) for name,e in local.items()}
        stream.synchronize()
        for name,value in values.items():
            assert bits_equal(value[0],ref['render']) and bits_equal(value[1],ref['radii'])
            try:local[name].render(camera,p,bg)
            except RuntimeError as error:assert 'per device/stream' in str(error)
            else:raise AssertionError('Cross stream reuse accepted')
            rows.append(dict(case='stream_dependency_and_rejection',variant=name,K=257,rgb_bitwise=True,radii_bitwise=True))
    for tiles in [255,256,257]:
        cam=MiniCam(tiles*16,7,1.2,1.,np.eye(4));p=synthetic_packet(1,'huge')
        p['inputs']['means3D'].zero_();p['inputs']['means3D'][:,2]=2.;compare('footprint_tiles/'+str(tiles),cam,p,bg)
    # 稀疏背景copy标志必须覆盖P>256但少量可见/全无Alpha。
    for P in [257,513,1025]:
        for style in ['outside','no_alpha','barely_alpha']:
            compare('sparse_background/'+str(P)+'/'+style,camera,synthetic_packet(P,style),bg)
    write_csv(out/'checks.csv',rows);save_json(out/'SUMMARY.json',dict(passed=True,comparisons=len(rows),modes=args.modes,
        retained_outputs=True,explicit_stream_checks=True,no_images=True))
    print('FOOTPRINT_LIFECYCLE_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
