"""一次性确认现有Renderer能否完整Capture；失败不变更Renderer或预算。"""
import copy
import json
import torch
from baseline_profile import ROOT,r,fixture
from implementation import Core,Execution


def main():
    sc,f,net,payload=fixture();ids=f.sets[.01,'same_k'];ex=Execution(net)
    x=ex.pack(ids);scale=f.features.anchor_scale[ids];condition=ex.condition(.01);core=Core(net)
    opt=torch.optim.Adam(net.parameters(),lr=.001,eps=1e-15,fused=True);opt.load_state_dict(copy.deepcopy(payload['optimizer']))
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            opt.zero_grad(set_to_none=True);loss=r.rt.photo(sc.render(ids,core(x,scale,condition),sc.train[0])[0],sc.train[0]['gt']);loss.backward()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();opt.zero_grad(set_to_none=True)
    graph=torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            loss=r.rt.photo(sc.render(ids,core(x,scale,condition),sc.train[0])[0],sc.train[0]['gt']);loss.backward();opt.step()
        result=dict(capture_succeeded=True,recommended=False,reason='single capture alone is insufficient for dynamic renderer semantics')
    except Exception as error:
        result=dict(capture_succeeded=False,recommended=False,error=str(error),scope='existing Renderer unchanged; no approximation or fixed tile-budget workaround')
    (ROOT/'full_graph_probe.json').write_text(json.dumps(result,indent=2));print('FULL_GRAPH_PROBE',result,flush=True)


if __name__=='__main__':main()
