from components import *
from deployment_runtime import deployment_extension


@torch.no_grad()
def main():
    s=scene();runtime=previous_runtime(s);ext=deployment_extension();old=previous_extension()
    ids,args=sample(s,runtime,396,.7)
    checks=Comparisons('COMPENSATION_NUMERIC_EDGES.json');states=[]
    for kind in ['checkpoint','zero','signed_zero','large']:
        values=list(args)
        if kind!='checkpoint':
            values[0]=values[0].clone();values[4]=[x.clone() for x in values[4]]
            if kind=='zero':
                values[0].zero_()
                for x in values[4]:x.zero_()
            elif kind=='signed_zero':
                values[0].fill_(-0.)
                for x in values[4]:x.fill_(-0.)
            else:
                values[0].mul_(1e3)
                for x in values[4]:x.mul_(100.)
        for k in [0,1,7,15,16,17,255,1023,1024,1031]:
            values[1]=ids[:k]
            for tile in [16,6416]:
                expected=old._compensate_tiled(*values,16)
                actual=ext._compensate_tiled(*values,tile)
                checks.same(f'{kind}/{k}/{tile}/status',expected[2],actual[2])
                finite=int(expected[2])==0
                if finite:
                    checks.same(f'{kind}/{k}/{tile}/attributes',expected[0],actual[0])
                    checks.same(f'{kind}/{k}/{tile}/heads',expected[1],actual[1])
                states.append(dict(kind=kind,K=k,tile=tile,finite=finite))
    checks.finish(cases=len(states),states=states)
    print('NUMERIC_EDGE_PASS',len(states),len(checks.checks),flush=True)


if __name__=='__main__':main()
