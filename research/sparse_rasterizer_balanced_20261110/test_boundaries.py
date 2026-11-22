"""显式验证128成员溢出、1024位目录边界、同深度次序及空Tile。"""
from validate import *


@torch.no_grad()
def main():
    torch.manual_seed(410)
    cam=MiniCam(97,65,1.2,1.0,np.eye(4))
    bg=torch.tensor([.13,.25,.76],device='cuda')
    control=SparseRenderer(0);engines=[SparseRenderer(3),SparseRenderer(4)]
    rows=[]
    for P in [1,31,32,33,127,128,129,1023,1024,1025,4097]:
        for depth_order in ['equal','mixed','outside']:
            xyz=torch.zeros(P,3,device='cuda');xyz[:,2]=2.
            if depth_order=='mixed':xyz[:,2]=torch.rand(P,device='cuda')+1.5
            if depth_order=='outside':xyz[:,2]=-.5
            q=torch.zeros(P,4,device='cuda');q[:,0]=1.
            packet=dict(sh_degree=0,inputs=dict(means3D=xyz,means2D=torch.zeros_like(xyz),
                opacities=torch.full((P,1),.02,device='cuda'),scales=torch.full((P,3),.08,device='cuda'),
                rotations=q,dc=None,shs=None,colors_precomp=torch.rand(P,3,device='cuda'),cov3D_precomp=None))
            ref=render_prepared(cam,packet,bg,return_radii=True)
            old=control.render(cam,packet,bg,True)
            tiles=((cam.image_width+15)//16)*((cam.image_height+15)//16)
            ids=old[7].cpu().tolist();ranges=old[8].cpu().tolist() if old[8].numel() else [(0,0)]*tiles
            for engine in engines:
                out=engine.render(cam,packet,bg,True)
                assert bits_equal(out[0],ref['render']) and bits_equal(out[1],ref['radii']),(P,depth_order,engine.mode)
                bits=out[9].cpu().numpy().astype(np.uint32)
                directory=out[10].cpu().numpy().astype(np.uint32)
                lists=out[11].cpu().tolist();counts=out[12].cpu().tolist()
                sorted_ids=out[7].cpu().tolist()
                for tile,words in enumerate(bits):
                    expected=ids[slice(*ranges[tile])]
                    enumerated=[]
                    for w,word in enumerate(words):
                        value=int(word)
                        assert bool(directory[tile,w//32] & (1<<(w%32)))==bool(value)
                        while value:
                            bit=(value&-value).bit_length()-1;enumerated.append(sorted_ids[w*32+bit]);value&=value-1
                    assert enumerated==expected,(P,depth_order,tile,'order')
                    if engine.mode==4:
                        assert counts[tile]==len(expected)
                        assert lists[tile][:min(128,len(expected))]==expected[:128]
                rows.append(dict(P=P,depth_order=depth_order,mode=engine.mode,
                    overflow_tiles=sum(c>128 for c in counts),max_count=max(counts,default=0),passed=True))
    result=dict(passed=True,checks=len(rows),pixel_bitwise=True,directory_bitwise=True,
        compact_prefix_and_overflow_counts_exact=True,rows=rows)
    (HERE/'reports/BOUNDARY_VALIDATION.json').write_text(json.dumps(result,indent=2))
    print('BOUNDARY_VALIDATION_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
