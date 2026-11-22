"""补齐全部16场景测速输入的radii逐位校验，并记录实际Tile工作量。"""
from benchmark import *


@torch.no_grad()
def main():
    engine=SparseRenderer(4);rows=[]
    conditions=[('local1pct',dict(ratio_budget=.01)),('local5pct',dict(ratio_budget=.05)),
        ('local10pct',dict(ratio_budget=.1)),('local70pct',dict(ratio_budget=.7)),
        ('local1000',dict(budget=1000)),('local5000',dict(budget=5000)),('full',{})]
    for item in SCENES:
        scene,source=load_scene(item['slug'])
        for condition,request in conditions:
            for i in range(8):
                frame=(19+137*i)%len(source.scene.views);cam=source.camera(frame)
                if condition=='full':packet,_=source.runtime.full()
                else:packet,_,_=source.runtime.prepare(cam,**request)
                ref=render_prepared(cam,packet,source.scene.background,return_radii=True)
                inspect=i==0 and condition in ['full','local1pct','local10pct']
                out=engine.render(cam,packet,source.scene.background,inspect)
                assert torch.equal(out[0].contiguous().view(torch.uint8),ref['render'].contiguous().view(torch.uint8))
                assert torch.equal(out[1],ref['radii'])
                row=dict(scene=scene['id'],condition=condition,frame=frame,K=len(out[1]),
                    width=cam.image_width,height=cam.image_height,pixel_bitwise=True,radii_bitwise=True)
                if inspect:
                    counts=out[12].cpu();words=(len(out[1])+31)//32
                    bits=out[9].cpu().numpy().astype(np.uint32)
                    nonempty=(bits!=0).sum(axis=1)
                    row.update(tiles=len(counts),active_tiles=int((counts>0).sum()),
                        compact_nonempty_tiles=int(((counts>0)&(counts<=128)).sum()),overflow_tiles=int((counts>128).sum()),
                        tile_count_median=float(counts.float().median()),tile_count_max=int(counts.max()),
                        words_per_tile=words,nonempty_words_mean=float(nonempty.mean()))
                rows.append(row)
        del source;torch.cuda.empty_cache()
        print('ALL_INPUTS_PARITY',item['slug'],len(rows),flush=True)
    result=dict(passed=True,cases=len(rows),pixel_bitwise=True,radii_bitwise=True,rows=rows)
    (HERE/'reports/ALL_RENDER_INPUTS_VALIDATION.json').write_text(json.dumps(result,indent=2))
    print('ALL_RENDER_INPUTS_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
