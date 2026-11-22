"""原A为oracle，覆盖大footprint/排序/容量/负零和真实调优视角。"""
from study_common import *
from footprint_runtime import Renderer,MODES,extension
from coverage_check import original_buffers,parse_binning


@torch.no_grad()
def exact_members(engine,camera,packet,bg):
    raw=original_buffers(camera,packet,bg);original=parse_binning(raw);snap=engine.native.snapshot()
    bits=snap[0].cpu().numpy().view(np.uint32);sorted_ids=snap[4].cpu().numpy();capacity=len(sorted_ids)
    ranks=np.empty(capacity,dtype=np.uint32);ranks[sorted_ids]=np.arange(capacity,dtype=np.uint32)
    keys=original['keys'].cpu().numpy().view(np.uint64);ids=original['ids'].cpu().numpy()
    tiles=bits.shape[0];tile=(keys>>np.uint64(32)).astype(np.uint32);valid=tile<tiles;tile=tile[valid];ids=ids[valid]
    expected=np.zeros_like(bits);rank=ranks[ids]
    np.bitwise_or.at(expected,(tile,rank//32),np.left_shift(np.uint32(1),rank%32))
    assert np.array_equal(bits,expected),'Tile bit mismatch'
    counts=snap[3].cpu().numpy();assert int(counts.sum())==len(ids)
    lists=snap[2].cpu().numpy();directory=snap[1].cpu().numpy().view(np.uint32)
    for tile_id in range(tiles):
        members=[]
        for word,mask in enumerate(expected[tile_id]):
            mask=int(mask)
            while mask:
                bit=(mask&-mask).bit_length()-1;members.append(int(sorted_ids[word*32+bit]));mask&=mask-1
        assert counts[tile_id]==len(members)
        assert np.array_equal(lists[tile_id,:min(256,len(members))],np.asarray(members[:256],dtype=np.uint32))
        for group in range(directory.shape[1]):
            value=sum((int(expected[tile_id,word]!=0)<<(word-group*32)) for word in range(group*32,min((group+1)*32,expected.shape[1])))
            assert int(directory[tile_id,group])==value
    return len(ids)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--modes',default='0,1,2,3,4,5,6,7')
    parser.add_argument('--quick',action='store_true');args=parser.parse_args();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    native_modules(BASELINE_CONFIG);engines={MODES[k]:Renderer(k) for k in map(int,args.modes.split(','))}
    rows=[];failed={};torch.manual_seed(104719)
    def check(label,camera,packet,bg,inspect=False):
        reference=render_prepared(camera,packet,bg,return_radii=True)
        for name,engine in engines.items():
            if name in failed:continue
            image,radii=engine.render(camera,packet,bg)
            row=dict(case=label,variant=name,K=len(radii),rgb_bitwise=bits_equal(image,reference['render']),radii_bitwise=bits_equal(radii,reference['radii']),tile_instances=-1)
            rows.append(row)
            if not row['rgb_bitwise'] or not row['radii_bitwise']:
                row['max_abs']=float((image-reference['render']).abs().max());save_json(out/'FAILURE.json',row)
                write_csv(out/'checks.csv',rows);failed[name]=dict(row)
                if name=='control':raise AssertionError(row)
                continue
            if inspect and len(radii):row['tile_instances']=exact_members(engine,camera,packet,bg)
    for W,H in [(97,65),(641,479)] if not args.quick else [(97,65)]:
        camera=MiniCam(W,H,1.2,1.,np.eye(4));bg=torch.tensor([.13,-0.,.76],device='cuda')
        for P in [0,1,31,32,33,127,255,256,257,511,512,513,1025] if not args.quick else [0,1,255,256,257,513]:
            for style in ['random','overlap','equal','outside','no_alpha','barely_alpha','huge','thin','nonfinite']:
                check(f'{W}x{H}/{P}/{style}',camera,synthetic_packet(P,style),bg,inspect=P in [1,33,257] and style in ['overlap','thin'])
    for scene in SCENES:
        if scene['split']!='tune' or args.quick:continue
        _,source=load_scene(scene['slug']);views=views_for(source,'tune')[-2:]
        for _,frame in views:
            camera=source.camera(frame)
            for condition in ['local0.01','local0.05','local0.1','local0.03','full']:
                packet,ids,_=prepare(source,camera,frame,condition);check(scene['id']+'/'+condition+'/'+str(frame),camera,packet,source.scene.background,
                    inspect=frame==views[-1][1] and condition=='local0.01')
        del source,packet;torch.cuda.empty_cache();write_csv(out/'checks.csv',rows)
        print('FOOTPRINT_VALIDATION',scene['id'],len(rows),flush=True)
    # 原失败极薄投影根例，当前候选必须沿用修复后的orderedTilePower。
    if not args.quick:
        _,source=load_scene('Replica__office0__rgbd');camera=source.camera(1291)
        packet,_,_=prepare(source,camera,1291,'local0.05');check('known_FMA_regression',camera,packet,source.scene.background,inspect=True)
    # 同输入新内容/独立输出，不允许通过旧视角内容缓存获得结果。
    retained=[];camera=MiniCam(97,65,1.2,1.,np.eye(4));bg=torch.tensor([.13,-0.,.76],device='cuda')
    packets=[synthetic_packet(P,'huge') for P in [31,257,513,1]]
    refs=[render_prepared(camera,p,bg,return_radii=True) for p in packets];torch.cuda.synchronize()
    for packet,ref in zip(packets,refs):
        for name,engine in engines.items():
            if name not in failed:retained.append((name,engine.render(camera,packet,bg),ref))
    torch.cuda.synchronize()
    for name,value,ref in retained:assert bits_equal(value[0],ref['render']) and bits_equal(value[1],ref['radii'])
    write_csv(out/'checks.csv',rows)
    save_json(out/'SUMMARY.json',dict(passed=True,comparisons=len(rows),modes=args.modes,valid_modes=[n for n in engines if n not in failed],failed_variants=failed,
        source_sha256=hashlib.sha256((HERE/'source/footprint_build.cuh').read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(Path(extension().__file__).read_bytes()).hexdigest(),retained_output_checks=len(retained)))
    print('FOOTPRINT_VALIDATION_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
