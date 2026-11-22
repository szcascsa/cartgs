"""合并bit-exp拒绝边界，含阈值相邻ULP、极端有限值和非有限值。"""
from student_study import *
from branchless_runtime import extension


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    torch.manual_seed(7851)
    cut=torch.tensor([-5.541263545158426],dtype=torch.float32).view(torch.int32).item()
    boundary=torch.arange(cut-8192,cut+8193,dtype=torch.int32).view(torch.float32)
    powers=torch.cat([boundary,torch.linspace(-1e6,1e6,1000001),torch.rand(1000000)*20-15,
        torch.tensor([float('nan'),float('inf'),-float('inf'),0.,-0.,-6.])]).cuda()
    old,new=extension()._alpha_parity(powers)
    mismatch=~(old.view(torch.int32)==new.view(torch.int32))
    record=dict(passed=not bool(mismatch.any()),values=len(powers),mismatches=int(mismatch.sum()))
    save(out/'DONE.json',record)
    assert record['passed'],(record,powers[mismatch][:20],old[mismatch][:20],new[mismatch][:20])
    print('MERGED_ALPHA_PARITY_PASS',len(powers),flush=True)


if __name__=='__main__':main()
