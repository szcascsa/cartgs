"""固定真实模型分层逐bit检查，失败停在首个不等价节点。"""
from common import *


@torch.no_grad()
def main():
    test=Comparisons('LAYER_VALIDATION.json');generator=torch.Generator(device='cuda').manual_seed(729)
    for group in ['both_li','selector_li','compensator_li','mode','legacy']:
        s=scene(group);model,pc=s.model,s.gaussians;engine=FrozenResidualInference(model,pc)
        frame=396;cam=camera(s,frame);code=engine.view_code(cam) if engine.joint_schema else None
        if code is not None:test.same(group+'/view_code',code,model.joint_view_code(cam))
        ratios=RATIOS+[max(1e-5,min(1.,r+e)) for r in model.anchors.compensation_values for e in [-2e-8,0,2e-8]]
        for size in [0,1,7,32,129,1000,len(pc.get_xyz)]:
            rows=torch.randperm(len(pc.get_xyz),device='cuda',generator=generator)[:size].sort().values
            compact=compact_gaussians(pc,rows);owners=model.anchors.rows[rows];reference=model.anchors.offset_scales[owners]
            for ratio in ratios:
                controls=engine.controls(ratio);features=model.anchors.compensation_lookup(rows,ratio)
                offset=(compact.get_xyz-model.anchors.centers[owners])/reference
                packed=torch.cat((((torch.asinh(offset)-engine.spatial.mean[3:6])/engine.spatial.stddev[3:6]).clamp(-15,15),features),1)
                test.same(f'{group}/{size}/{ratio}/comp_pack',engine.compensation_input(rows,controls),packed)
                for local in ([False,True] if engine.joint_schema else [False]):
                    label=f'{group}/{size}/{ratio}/local{local}'
                    values=[]
                    hook=model.local_emd[0].register_forward_pre_hook(lambda m,args:values.append(args[0]))
                    args=(compact.get_xyz,compact.get_scaling,compact.get_features_dc,compact.get_opacity,pc.get_xyz.new_tensor([[ratio]]))
                    if local:expected=model.forward_view_selection_features(*args,cam,source_rows=rows,return_logits=True,anchor_code=code)[2]
                    else:expected=model.forward_selection_features(*args,source_rows=rows,return_logits=True)[2]
                    hook.remove()
                    test.same(label+'/selector_input',engine.selector_input(rows,ratio,code,local),values[0])
                    test.same(label+'/logits',engine.logits(rows,ratio,code,local),expected)
                    condition=None
                    if local:
                        branch=model.joint_branch_view_code(code,compensation=True)
                        condition=torch.cat((branch[owners],code.new_ones((len(rows),1))),1)
                        test.same(label+'/condition',engine.ext.condition_pack(rows,engine.owners,code,engine.joint_schema in (2,4)),condition)
                    residual=model.spatial_compensation(compact.get_xyz,compact._scaling,compact.get_rotation,compact.get_features_dc,
                        compact._opacity,model.anchors.centers[owners],reference,ratio,log_scale_input=True,anchor_feature=features,view_condition=condition)
                    test.same(label+'/delta14',engine.residual(rows,controls,code,local),residual)
            print('layers',group,size,flush=True)
        test.finish(group_completed=group)
        del engine,s;torch.cuda.empty_cache()
    test.finish(groups=5)


if __name__=='__main__':main()
