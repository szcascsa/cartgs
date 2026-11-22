"""核对teacher自定义打包与原Renderer路径，及Full alpha质量守恒。"""
import json
from pathlib import Path
import types
import torch
import evaluate as e


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    config=json.loads((e.ROOT/'configs/experiment.json').read_text())
    scene=e.BudgetScene(config['checkpoint'],config['dataset'],5000,0,None,'pytorch')
    binding=e.load_binding();model=scene.model;pc=scene.gaussians;anchors=model.anchors
    gi=torch.jit.load(str(Path(config['checkpoint'])/'flexgs_importance.pt'),map_location='cuda').state_dict()['0']
    records=[];conservation=[]
    for frame in [0,169,396,791]:
        v=scene.views[frame];c=scene.calibration
        camera=e.MiniCam(c['width'],c['height'],e.focal2fov(c['fx'],c['width']),e.focal2fov(c['fy'],c['height']),e.np.linalg.inv(e.np.asarray(v['pose'])))
        full=e.native_render(binding,camera,scene.full_data,scene.background,training=True)
        q=full[2]/float(camera.image_height*camera.image_width)
        alpha_inputs=dict(scene.full_data['inputs']);alpha_inputs['colors_precomp']=torch.ones((13298,3),device='cuda')
        alpha_inputs['dc']=None;alpha_inputs['shs']=None
        alpha=e.native_render(binding,camera,dict(sh_degree=scene.full_data['sh_degree'],inputs=alpha_inputs),scene.background,statistic=0,training=True)[0]
        error=abs(float(q.sum())-float(alpha[0].mean()));assert error<1e-4
        conservation.append(dict(frame=frame,gaussian_alpha_mass=float(q.sum()),pixel_alpha_mean=float(alpha[0].mean()),absolute_error=error))
        visible=e.anchor_frustum_mask(anchors.centers,anchors.offset_scales,camera.full_proj_transform,scene.renderer.occupied)
        rows=torch.nonzero(visible[anchors.rows]).flatten();compact=e.compact_gaussians(pc,rows)
        eligible=e.slot_eligibility(model,compact.get_opacity,rows)
        for ratio in [.01,.05]:
            labels=binding.teacher(q[rows],gi[rows],rows,eligible,ratio);assert labels is not None
            mask=labels.bool()
            original=model.forward_selection_features
            def forced(self,*args,**kwargs):
                source=kwargs['source_rows'];assert source.equal(rows)
                noise=scene.renderer.noise[source]
                # Scores saturate only in this test to force exactly the same
                # existing teacher IDs through the unmodified Renderer API.
                keep=torch.where(mask,torch.full_like(labels,100),torch.full_like(labels,-100))-(noise[:,1]-noise[:,0])
                logits=torch.stack((torch.zeros_like(keep),keep),1)
                return logits.softmax(1)[:,1],torch.zeros((len(rows),32),device='cuda'),logits
            model.forward_selection_features=types.MethodType(forced,model)
            try:
                for comp in [False,True]:
                    own,ids=e.selected_data(scene,compact,rows,mask,ratio,comp)
                    reference,ref_ids,_=scene.renderer.prepare(camera,ratio_budget=ratio,compensation=comp,view_conditioned=False)
                    assert ids.equal(ref_ids)
                    for name,value in own['inputs'].items():
                        other=reference['inputs'][name]
                        if isinstance(value,torch.Tensor):assert torch.equal(value.view(torch.uint8),other.view(torch.uint8)),name
                        else:assert value is other
                    a=e.native_render(binding,camera,own,scene.background)[0]
                    b=e.render_prepared(camera,reference,scene.background)['render']
                    assert torch.equal(a.view(torch.int32),b.view(torch.int32))
                    records.append(dict(frame=frame,ratio=ratio,compensation=comp,K=len(ids),all_attributes_bitwise=True,image_bitwise=True))
            finally:model.forward_selection_features=original
    scene.renderer.check_state()
    e.save(e.ROOT/'reports/TEACHER_PAYLOAD_PARITY.json',dict(passed=True,cases=len(records),records=records,alpha_conservation=conservation))
    print('TEACHER_PAYLOAD_BITWISE_PASS',len(records))


if __name__=='__main__':main()
