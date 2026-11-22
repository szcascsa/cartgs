"""独立复现正式输入差异，分离旧B和新增C/G，不用近似质量放行。"""
from harness import *
from runtime import SparseRenderer
from graph_runtime import GraphRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--ordered',action='store_true');args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False)
    scene,source=load_scene('Replica__office0__rgbd');frame=1291;condition='local0.05';cam=source.camera(frame)
    p,ids,stats=prepare(source,cam,frame,condition);bg=source.scene.background
    ref=render_prepared(cam,p,bg,return_radii=True);rows=[];debug={}
    engines={'serial':SparseRenderer(mode=0),'B':baseline_B(),'C':SparseRenderer(render_variant=5,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2,ordered_coverage=args.ordered),
        'G':GraphRenderer(plan_granularity=512,max_plans=512,ordered_coverage=args.ordered)}
    for name,engine in engines.items():
        out=engine.render(cam,p,bg,True) if name!='G' else engine.render(cam,p,bg)
        rgb=out[0];a=rgb.contiguous().view(torch.int32);b=ref['render'].contiguous().view(torch.int32)
        where=(a!=b).nonzero();positions=where[:20].cpu().tolist();indices=tuple(where[:20].unbind(1)) if len(where) else None
        delta=(rgb-ref['render']).abs();row=dict(variant=name,K=len(ids),rgb_bitwise=bits_equal(rgb,ref['render']),radii_bitwise=bits_equal(out[1],ref['radii']),
            max_abs=float(delta.max()),different_values=len(where),positions=positions,
            values=rgb[indices].cpu().tolist() if indices else [],reference=ref['render'][indices].cpu().tolist() if indices else [],
            bits=a[indices].cpu().tolist() if indices else [],reference_bits=b[indices].cpu().tolist() if indices else [])
        rows.append(row)
        if name!='G':debug[name]=out
    # 直接调用只读原二进制的Full forward，解析其原始几何与排序缓冲；不修改原二进制。
    import diff_gaussian_rasterization
    inputs=p['inputs'];empty=bg.new_empty(0)
    tensor=lambda name:inputs[name] if inputs[name] is not None else empty
    raw=diff_gaussian_rasterization._C.rasterize_gaussians(bg,inputs['means3D'],tensor('colors_precomp'),inputs['opacities'],tensor('scales'),tensor('rotations'),
        1.,tensor('cov3D_precomp'),cam.world_view_transform,cam.full_proj_transform,math.tan(cam.FoVx*.5),math.tan(cam.FoVy*.5),cam.image_height,cam.image_width,
        tensor('dc'),tensor('shs'),p['sh_degree'],cam.camera_center,False,False)
    rows.append(dict(variant='A_training_forward',rgb_bitwise=bits_equal(raw[2],ref['render']),max_abs=float((raw[2]-ref['render']).abs().max())))
    def fields(buffer,layout):
        offset=0;values={}
        for label,count,dtype in layout:
            offset=(offset+127)//128*128;size=count*torch.empty(0,dtype=dtype).element_size()
            values[label]=buffer[offset:offset+size].view(dtype);offset+=size
        return values
    N=len(ids);M=raw[0]
    geom=fields(raw[4],[('depth',N,torch.float32),('clamped',3*N,torch.uint8),('radii',N,torch.int32),('xy',2*N,torch.float32),
        ('covariance',6*N,torch.float32),('conic',4*N,torch.float32),('rgb',3*N,torch.float32),('bbox',N,torch.int32)])
    original=[]
    for label,index,width in [('depth',4,1),('xy',2,2),('conic',3,4),('bbox',5,1),('rgb',13,3)]:
        first=geom[label].reshape(N,width);second=debug['C'][index].reshape(N,width);mask=ref['radii']>0
        original.append(dict(field=label,equal=bits_equal(first[mask],second[mask]),max_abs=float((first[mask]-second[mask]).abs().max())))
    # 原bin前四字段均在CUB scratch之前，直接定位当前像素所属Tile有序点表。
    binning=fields(raw[5],[('sorted_ids',M,torch.int32),('unsorted_ids',M,torch.int32),('sorted_keys',M,torch.int64),('unsorted_keys',M,torch.int64)])
    tile=(447//16)*math.ceil(cam.image_width/16)+798//16
    mask=(binning['sorted_keys']>>32)==tile;old_ids=binning['sorted_ids'][mask].cpu().tolist()
    bits=debug['C'][9][tile].cpu().numpy().view(np.uint32);sorted_ids=debug['C'][7].cpu().tolist();new_ids=[]
    for word,bitmask in enumerate(bits):
        value=int(bitmask)
        while value:
            bit=(value&-value).bit_length()-1;new_ids.append(sorted_ids[word*32+bit]);value&=value-1
    lists=dict(tile=tile,original=old_ids,candidate=new_ids,original_only=[i for i in old_ids if i not in new_ids],candidate_only=[i for i in new_ids if i not in old_ids],same=old_ids==new_ids)
    suspects=[]
    for i in lists['original_only']+lists['candidate_only']:
        suspects.append(dict(id=i,xy=debug['C'][2][i].cpu().tolist(),conic=debug['C'][3][i].cpu().tolist(),
            xy_bits=debug['C'][2][i].view(torch.int32).cpu().tolist(),conic_bits=debug['C'][3][i].view(torch.int32).cpu().tolist(),
            radius=int(debug['C'][1][i]),bbox=int(debug['C'][5][i])))
    internals=[]
    if debug['B'][2].shape==debug['C'][2].shape:
        for index,label in [(1,'radii'),(2,'xy'),(3,'conic'),(4,'depth'),(5,'bbox'),(7,'sorted_id'),(9,'bits'),(10,'directory')]:
            left,right=debug['B'][index],debug['C'][index]
            if index in [2,3,4]:left,right=left[debug['B'][1]>0],right[debug['C'][1]>0]
            internals.append(dict(field=label,equal=bits_equal(left,right),shape=list(left.shape)))
    value=dict(scene=scene['id'],frame=frame,condition=condition,N=source.runtime.total,K=len(ids),background=bg.cpu().tolist(),rows=rows,internals=internals,
        original_geometry=original,tile_lists=lists,suspects=suspects)
    write_json(output/'SUMMARY.json',value);print(json.dumps(value,indent=2),flush=True)


if __name__=='__main__':main()
