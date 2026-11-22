"""独立算子微测：真实权重、预打包oneDNN、同FP32；不改变冻结候选。"""
from common import *

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:'+os.environ.get('PATH','');os.environ['MAX_JOBS']='1'
    from torch.utils.cpp_extension import load
    from torch.utils.mkldnn import to_mkldnn
    core=(HERE/'source/cpu_deep.cpp').read_text();prelude=core[:core.index('class Pool')];dense=core[core.index('struct Dense {'):core.index('\nstruct Key')]
    code=prelude+dense+r'''
struct DenseOperator {
  Dense value;DenseOperator(T w,T b,int fast):value(w,b){value.fast=fast;}
  T forward(T x,T y){TORCH_CHECK(x.device().is_cpu()&&x.scalar_type()==at::kFloat&&x.is_contiguous()&&x.dim()==2&&x.size(1)==value.in,"Dense input");
    TORCH_CHECK(y.device().is_cpu()&&y.scalar_type()==at::kFloat&&y.is_contiguous()&&y.size(0)==x.size(0)&&y.size(1)==value.out,"Dense output");
    auto a=x.data_ptr<float>();auto b=y.data_ptr<float>();for(int i=0;i<x.size(0);i+=4)value.apply(a+i*value.in,std::min<int64_t>(4,x.size(0)-i),value.in,b+i*value.out,value.out);return y;}
};
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){pybind11::class_<DenseOperator>(m,"Dense").def(pybind11::init<T,T,int>()).def("forward",&DenseOperator::forward,pybind11::call_guard<pybind11::gil_scoped_release>());}
'''
    build=HERE/'build_dense';build.mkdir(exist_ok=True);(build/'dense.cpp').write_text(code)
    module=load(name='cpu_deep_dense_probe',sources=[str(build/'dense.cpp')],extra_cflags=['-O3','-mavx2','-mfma','-fno-math-errno'],build_directory=str(build),verbose=True)
    scene=load_scene('Replica__office2__rgbd');model=scene['model'];s=model.spatial_compensation
    matrices=[('selector_body',model.local_emd[2].weight,model.local_emd[2].bias),('comp_body',s.body.weight,s.body.bias),
        ('four_head_hidden',torch.cat([h[0].weight for h in s.heads.values()]),torch.cat([h[0].bias for h in s.heads.values()]))]
    rows=[];checks=[];torch.manual_seed(0)
    for name,w,b in matrices:
        linear=torch.nn.Linear(w.shape[1],w.shape[0]);linear.weight.copy_(w);linear.bias.copy_(b);packed=to_mkldnn(linear.eval())
        operators=[module.Dense(w,b,0),module.Dense(w,b,1)]
        for n in [4,32,256,4096]:
            x=torch.randn(n,w.shape[1]);opaque=x.to_mkldnn();outputs=[torch.empty(n,w.shape[0]),torch.empty(n,w.shape[0])]
            expected=torch.nn.functional.linear(x,w,b)
            functions=dict(avx4x8=lambda:operators[0].forward(x,outputs[0]),avx4x16=lambda:operators[1].forward(x,outputs[1]),
                aten=lambda:torch.nn.functional.linear(x,w,b),onednn_dense_io=lambda:packed(x),onednn_opaque_io=lambda:packed(opaque))
            for variant,function in functions.items():
                y=function();y=y.to_dense() if y.is_mkldnn else y
                checks.append(dict(matrix=name,n=n,variant=variant,max_abs_error=float((y-expected).abs().max()),all_finite=bool(torch.isfinite(y).all())))
            repeats=100 if n<256 else 30
            for block in range(3):
                for variant in rotate(list(functions),block):
                    function=functions[variant]
                    for _ in range(12):function()
                    for repeat in range(repeats):
                        tick=time.perf_counter_ns();function();elapsed=(time.perf_counter_ns()-tick)/1e6
                        rows.append(dict(matrix=name,n=n,inputs=w.shape[1],outputs=w.shape[0],variant=variant,block=block,sample=repeat,ms=elapsed))
            print('ONEDNN_PAIRED',name,n,flush=True)
    write_csv(out/'samples.csv',rows);write_csv(out/'checks.csv',checks)
    save(out/'SUMMARY.json',dict(passed=all(x['all_finite'] for x in checks),oneDNN_available=torch.backends.mkldnn.is_available(),
        threads=1,binary_sha256=sha(module.__file__),note='AVX输出复用；oneDNN权重预打包；opaque输入仅为理想布局算子上限，未含动态输入转换，不能当作端到端收益'))

if __name__=='__main__':main()
