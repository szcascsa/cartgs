"""从原二进制实际排序缓冲重建Tile成员，核对全部位而不只比较最终图像。"""
from harness import *
from runtime import SparseRenderer
from graph_runtime import GraphRenderer


@torch.no_grad()
def original_buffers(camera,packet,bg):
    import diff_gaussian_rasterization
    p=packet['inputs'];empty=bg.new_empty(0)
    tensor=lambda name:p[name] if p[name] is not None else empty
    return diff_gaussian_rasterization._C.rasterize_gaussians(bg,p['means3D'],tensor('colors_precomp'),p['opacities'],tensor('scales'),tensor('rotations'),
        1.,tensor('cov3D_precomp'),camera.world_view_transform,camera.full_proj_transform,math.tan(camera.FoVx*.5),math.tan(camera.FoVy*.5),
        camera.image_height,camera.image_width,tensor('dc'),tensor('shs'),packet['sh_degree'],camera.camera_center,False,False)


def parse_binning(raw):
    count=raw[0];buffer=raw[5];offset=0;result={}
    for name,dtype in [('ids',torch.int32),('unsorted_ids',torch.int32),('keys',torch.int64),('unsorted_keys',torch.int64)]:
        offset=(offset+127)//128*128;size=count*torch.empty(0,dtype=dtype).element_size()
        result[name]=buffer[offset:offset+size].view(dtype);offset+=size
    return result


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--all-scenes',action='store_true');args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False);rows=[]
    c=SparseRenderer(render_variant=5,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2,ordered_coverage=True)
    g=GraphRenderer(plan_granularity=512,max_plans=512,ordered_coverage=True,device_clear=True)
    def check(scene,frame,condition,camera,packet,bg):
        out=c.render(camera,packet,bg,True);P=len(out[1]);ref=render_prepared(camera,packet,bg,return_radii=True)
        image,radii=g.render(camera,packet,bg)
        row=dict(scene=scene,frame=frame,condition=condition,K=P,
            C_rgb_bitwise=bits_equal(out[0],ref['render']),C_radii_bitwise=bits_equal(out[1],ref['radii']),
            G_rgb_bitwise=bits_equal(image,ref['render']),G_radii_bitwise=bits_equal(radii,ref['radii']),
            tile_bits_equal=True,missing_members=0,extra_members=0,valid_instances=0)
        if P:
            raw=original_buffers(camera,packet,bg);binning=parse_binning(raw)
            words=(P+31)//32;tiles=math.ceil(camera.image_width/16)*math.ceil(camera.image_height/16)
            keys=binning['keys'].cpu().numpy().view(np.uint64);ids=binning['ids'].cpu().numpy()
            tile=(keys>>np.uint64(32)).astype(np.uint32);valid=tile<tiles;tile=tile[valid];ids=ids[valid]
            ranks=np.empty(P,dtype=np.uint32);ranks[out[7].cpu().numpy()]=np.arange(P,dtype=np.uint32)
            ranks=ranks[ids];expected=np.zeros((tiles,words),dtype=np.uint32)
            np.bitwise_or.at(expected,(tile,ranks//32),np.left_shift(np.uint32(1),ranks%32))
            actual=out[9].cpu().numpy().view(np.uint32)
            missing=np.bitwise_and(expected,np.bitwise_not(actual));extra=np.bitwise_and(actual,np.bitwise_not(expected))
            row.update(tile_bits_equal=np.array_equal(actual,expected),valid_instances=len(ids),
                missing_members=int(np.unpackbits(missing.view(np.uint8)).sum()),extra_members=int(np.unpackbits(extra.view(np.uint8)).sum()))
            assert bits_equal(raw[2],ref['render']),'Original training/inference RGB differ; buffer oracle invalid'
        rows.append(row)
        if not all(row[name] for name in ['C_rgb_bitwise','C_radii_bitwise','G_rgb_bitwise','G_radii_bitwise','tile_bits_equal']):
            write_json(output/'FAILURE.json',row);write_csv(output/'checks.csv',rows);raise AssertionError(row)
        return row
    for scene in SCENES:
        if not args.all_scenes and scene['split']!='tune' and scene['slug']!='Replica__office0__rgbd':continue
        _,source=load_scene(scene['slug']);views=views_for(source,'final' if args.all_scenes else 'tune')
        chosen=[views[0],views[-1]]
        if scene['slug']=='Replica__office0__rgbd':chosen.append(('regression',1291))
        for motion,frame in chosen:
            camera=source.camera(frame)
            for condition in ['local0.01','local0.05','local0.1','local0.03','local0.7','global0.01','full']:
                packet,ids,stats=prepare(source,camera,frame,condition)
                check(scene['id'],frame,condition,camera,packet,source.scene.background)
        write_csv(output/'checks.csv',rows)
        del source,packet;torch.cuda.empty_cache();print('COVERAGE_PARITY',scene['id'],len(rows),flush=True)
    write_json(output/'SUMMARY.json',dict(passed=True,comparisons=len(rows),all_scenes=args.all_scenes,
        exact_original_tile_members=True,rows=rows,images_saved=False))
    print('COVERAGE_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
