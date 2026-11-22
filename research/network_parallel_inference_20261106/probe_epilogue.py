from common import *
@torch.no_grad()
def main():
 s=scene();g=torch.Generator(device='cuda').manual_seed(582);rows=[]
 layers={'selector0':s.model.local_emd[0],'selector1':s.model.local_emd[2],'body':s.model.spatial_compensation.body}
 layers.update({'head_'+name:s.model.spatial_compensation.heads[name][0] for name in ['xyz','scale','rotation','other']})
 for name,layer in layers.items():
  for n in [0,1,7,32,129,1000,5000,13298]:
   for scale in [0.,.01,1.,100.]:
    x=torch.randn(n,layer.in_features,device='cuda',generator=g)*scale
    a=torch.relu(torch.nn.functional.linear(x,layer.weight,layer.bias))
    b=torch._addmm_activation(layer.bias,x,layer.weight.t(),use_gelu=False)
    same=torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
    row=dict(layer=name,N=n,scale=scale,bitwise=same)
    if not same:row.update(max_abs=float((a-b).abs().max()),different=int((a.view(torch.int32)!=b.view(torch.int32)).sum()))
    rows.append(row)
  print(name,'passed',sum(v['bitwise'] for v in rows if v['layer']==name),'/',sum(v['layer']==name for v in rows),flush=True)
 save(ROOT/'reports/EPILOGUE_PROBE.json',dict(passed=all(v['bitwise'] for v in rows),checks=rows))
if __name__=='__main__':main()
