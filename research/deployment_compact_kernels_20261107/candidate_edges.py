from common import *
from deployment_runtime import deployment_extension


@torch.no_grad()
def main():
    torch.manual_seed(96);ext=deployment_extension();checks=Comparisons('CANDIDATE_EDGES.json')
    for anchors in [0,1,31,255,256,257,2790,19936]:
        for kind in ['random','all_hidden','all_visible','nonfinite']:
            n=anchors*3
            owners=torch.randint(anchors,(n,),device='cuda',dtype=torch.int64) if anchors else torch.empty(0,device='cuda',dtype=torch.int64)
            scales=torch.rand(anchors,device='cuda')*10
            occupied=torch.rand(anchors,device='cuda')>.1
            eligible=torch.randint(100,(anchors,),device='cuda',dtype=torch.int64)
            if anchors:eligible[0]=2**40
            distance=torch.randn(anchors,6,device='cuda')
            planes=torch.randn(6,4,device='cuda')
            if kind=='all_hidden':occupied.zero_()
            if kind=='all_visible':distance.fill_(1e6);occupied.fill_(True)
            if kind=='nonfinite' and anchors:distance[0,0]=float('nan')
            rows=torch.empty(n,device='cuda',dtype=torch.int64);counts=torch.empty(3,device='cuda',dtype=torch.int64)
            work=torch.empty(0,device='cuda',dtype=torch.uint8)
            partial=torch.empty(2*((anchors+255)//256),device='cuda',dtype=torch.int64)
            visible=ext._frustum_test(distance,planes,scales,occupied)
            work=ext._visible_candidates(visible,owners,eligible,rows,counts,work)[2]
            old_rows=rows[:int(counts[0])].clone();old_counts=counts.clone()
            result,work=ext._fused_candidates(distance,planes,scales,occupied,owners,eligible,rows,counts,work,partial)
            ext._finish_counts(partial,counts)
            label=f'{anchors}/{kind}'
            checks.same(label+'/visible',visible,result);checks.same(label+'/counts',old_counts,counts)
            checks.same(label+'/rows',old_rows,rows[:int(counts[0])])
    checks.finish(cases=32,int64_counts=True)
    print('CANDIDATE_EDGE_PASS',len(checks.checks),flush=True)


if __name__=='__main__':main()
