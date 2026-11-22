"""使用CUDA Graph原生DOT转储计数节点，绕过本环境CUPTI后处理停滞。"""
import inspect
import json
import argparse
import torch
from baseline_profile import ROOT,fixture
from implementation import Execution


def graphs(value,seen):
    if id(value) in seen:return
    seen.add(id(value))
    if isinstance(value,torch.cuda.CUDAGraph):yield value;return
    if isinstance(value,torch.nn.Module):yield from graphs(value.forward,seen)
    elif isinstance(value,type) and issubclass(value,torch.autograd.Function):
        yield from graphs(value.forward,seen);yield from graphs(value.backward,seen)
    elif inspect.ismethod(value):yield from graphs(value.__func__,seen)
    elif inspect.isfunction(value):
        for cell in value.__closure__ or ():
            try:content=cell.cell_contents
            except ValueError:continue
            yield from graphs(content,seen)


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',default='graph_all_fused');args=p.parse_args()
    debug=torch.cuda.CUDAGraph();debug.enable_debug_mode()
    sc,f,net,payload=fixture()
    if args.method=='decoded_graph':
        from decoded_graph import DecodedExecution
        execution=DecodedExecution(net,sc.render_bases)
    else:execution=Execution(net,args.method)
    rows=[];prefix='' if args.method=='graph_all_fused' else args.method+'_'
    for ratio in [.01,.15]:
        ids=f.sets[ratio,'same_k'];execution(ratio,ids)
        key=next(k for k in execution.graphs if k[0 if args.method=='decoded_graph' else 1]==len(ids));found=list(graphs(execution.graphs[key],set()))
        for j,graph in enumerate(found):
            target=ROOT/f'{prefix}captured_K{len(ids)}_{j}.dot';graph.debug_dump(str(target));rows.append(dict(method=args.method,K=len(ids),graph=j,path=target.name))
    (ROOT/(prefix+'graph_nodes.json')).write_text(json.dumps(rows,indent=2));print('GRAPH_DUMPS',rows,flush=True)


if __name__=='__main__':main()
