"""区分产品入口的空集背景行为与旧研究primitive的零输出行为。"""
from harness import *
from runtime import SparseRenderer
from graph_runtime import GraphRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False)
    p=dict(sh_degree=0,inputs=dict(means3D=torch.empty((0,3),device='cuda'),opacities=torch.empty((0,1),device='cuda'),
        scales=torch.empty((0,3),device='cuda'),rotations=torch.empty((0,4),device='cuda'),dc=torch.empty((0,1,3),device='cuda'),
        shs=torch.empty((0,15,3),device='cuda'),colors_precomp=None,cov3D_precomp=None))
    rows=[]
    for cam in [MiniCam(17,19,1.2,1.,np.eye(4)),MiniCam(97,65,1.2,1.,np.eye(4))]:
        for bg in [torch.tensor([0.,0.,0.],device='cuda'),torch.tensor([.13,-0.,.76],device='cuda')]:
            ref=render_prepared(cam,p,bg,return_radii=True)
            for name,engine in [('C',SparseRenderer()),('G',GraphRenderer(plan_granularity=512)),('B',baseline_B())]:
                image,radii=engine.render(cam,p,bg)
                passed=bits_equal(image,ref['render']) and bits_equal(radii,ref['radii'])
                if name!='B':assert passed
                rows.append(dict(variant=name,width=cam.image_width,nonblack_bg=bool(bg[0]),matches_product_entry=passed))
    write_json(output/'SUMMARY.json',dict(passed=True,rows=rows,
        scope='C/G包装层沿用A产品入口空集背景，B旧primitive空集为零，单独披露'))
    print('EMPTY_ENTRY_PASS',len(rows),flush=True)


if __name__=='__main__':main()
