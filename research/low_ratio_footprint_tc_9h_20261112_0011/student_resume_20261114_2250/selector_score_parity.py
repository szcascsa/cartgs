"""双logit概率逐位验证，覆盖饱和、相等、相邻ULP和非有限值。"""
from student_study import *
from selector_oneexp_runtime import extension


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);torch.manual_seed(12057)
    values=torch.tensor([-float('inf'),-1e30,-1000.,-100.,-20.,-1.,-0.,0.,1.,20.,100.,1000.,1e30,float('inf'),float('nan')])
    x,y=torch.meshgrid(values,values,indexing='ij')
    adjacent=torch.linspace(-80,80,100001)
    a=torch.cat([torch.randn(1000000)*100,torch.rand(1000000)*2e6-1e6,adjacent,adjacent,x.flatten()]).cuda()
    b=torch.cat([torch.randn(1000000)*100,torch.rand(1000000)*2e6-1e6,adjacent,torch.nextafter(adjacent,torch.full_like(adjacent,float('inf'))),y.flatten()]).cuda()
    old,new=extension()._selector_score_parity(a,b)
    equal=(old.view(torch.int32)==new.view(torch.int32))|(torch.isnan(old)&torch.isnan(new))
    finite=torch.isfinite(a)&torch.isfinite(b)
    result=dict(passed=bool(equal[finite].all()),values=len(a),finite_mismatches=int((~equal[finite]).sum()),
        invalid_input_mismatches=int((~equal[~finite]).sum()),invalid_input_contract='Selector原finite检查仍报错；不把无效logit输出称逐位一致',
        max_finite_error=float(torch.nan_to_num((old[finite]-new[finite]).abs()).max()))
    save(out/'DONE.json',result);assert result['passed'],result;print('SELECTOR_SCORE_PARITY_PASS',result,flush=True)


if __name__=='__main__':main()
