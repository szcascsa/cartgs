"""交付只读加载入口与Q组合的真实运行身份。"""
from harness import *
from frozen_renderer import create_renderer,create_camera_preparer
from camera_graph_preparation import GraphCameraPreparer
import ctypes


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False)
    scene,source=load_scene('Replica__office2__rgbd')
    camera_variant='R' if (HERE/'CAMERA_ASYNC_FINAL_VERIFICATION.json').exists() and json.loads((HERE/'CAMERA_ASYNC_FINAL_VERIFICATION.json').read_text())['passed'] else 'Q'
    factory=create_camera_preparer(MiniCam,variant=camera_variant)
    camera=factory.from_source(source,31);g=create_renderer(camera,source.scene.background,source.runtime.total)
    c=create_renderer(camera,source.scene.background,source.runtime.total,use_graph=False)
    checks=[]
    for frame,condition in [(31,'local0.01'),(120,'local0.05'),(713,'local0.1'),(612,'full'),(901,'fixed0')]:
        camera=factory.from_source(source,frame);original=source.camera(frame)
        packet,ids,_=prepare(source,camera,frame,condition);ref=render_prepared(original,packet,source.scene.background,return_radii=True)
        for name,engine in [('C',c),('G+'+camera_variant,g)]:
            image,radii=engine.render(camera,packet,source.scene.background)
            row=dict(frame=frame,condition=condition,variant=name,K=len(ids),RGB_bitwise=bits_equal(image,ref['render']),radii_bitwise=bits_equal(radii,ref['radii']))
            checks.append(row);assert row['RGB_bitwise'] and row['radii_bitwise'],row
    maps=Path('/proc/self/maps').read_text();paths=sorted({line.split()[-1] for line in maps.splitlines() if '/' in line})
    selected=[path for path in paths if any(name in path for name in ['mycartgs_sparse_','diff_gaussian','libcudart','libtorch','libcublas','libcusolver'])]
    runtimes=[path for path in selected if 'libcudart.so' in path];assert runtimes
    runtime_version=ctypes.c_int();library=ctypes.CDLL(runtimes[0]);assert library.cudaRuntimeGetVersion(ctypes.byref(runtime_version))==0
    binaries=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text())['binaries']
    for row in binaries.values():assert row['path'] in selected
    result=dict(passed=True,checks=checks,loaded_paths=selected,python=sys.version,python_executable=sys.executable,
        torch_version=torch.__version__,torch_cuda=torch.version.cuda,cuda_runtime_version=runtime_version.value,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),default_dtype=str(torch.get_default_dtype()),
        autograd_enabled=torch.is_grad_enabled(),current_stream=int(torch.cuda.current_stream().cuda_stream),
        GPU=torch.cuda.get_device_name(),graph_stats=g.native.stats(),production_unchanged=True,training_not_invoked=True)
    modules={}
    for name,module in list(sys.modules.items()):
        filename=getattr(module,'__file__',None)
        if filename and str(filename).startswith('/home/prmi/zwj/') and Path(filename).is_file():
            path=Path(filename)
            if path.suffix in ['.py','.so']:
                modules[str(path)]=dict(module=name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for path in selected:
        if path.startswith('/home/prmi/zwj/') and Path(path).is_file():
            modules.setdefault(path,dict(module='proc_maps',bytes=Path(path).stat().st_size,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()))
    write_json(HERE/'reports/DEPENDENCY_IDENTITY.json',dict(passed=True,modules=modules,
        scope='实际加载的项目源码/二进制；不复制原checkpoint、数据或共享环境'))
    write_json(output/'SUMMARY.json',result);write_json(HERE/'reports/RUNTIME_IDENTITY.json',result)
    print('DELIVERY_SMOKE_COMPLETE',len(checks),runtime_version.value,flush=True)


if __name__=='__main__':main()
