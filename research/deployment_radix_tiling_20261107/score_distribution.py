from components import *


@torch.no_grad()
def main():
    ext=deployment_extension();out=[]
    for group in ['both_li','replica']:
        s=scene(group) if group!='replica' else BudgetScene(LARGE,None,5000,0,None,'pytorch');a=previous_runtime(s)
        for frame in [0,len(s.views)//2,len(s.views)-1]:
            for ratio in [.01,.1,.7]:
                rows,scores,ids,_,_,_=sample(s,a,frame,ratio);work=scratch(len(rows))
                selected=ext._radix_select(scores[0],rows,a._base.eligible,len(ids),work,scores[2]);assert torch.equal(ids,selected)
                score=scores[0];valid=a._base.eligible[rows];values=score[valid]
                out.append(dict(group=group,frame=frame,ratio=ratio,N=len(rows),K=len(ids),eligible=int(valid.sum()),
                    unique_scores=len(torch.unique(values)),zero_scores=int((values==0).sum()),one_scores=int((values==1).sum()),
                    radix_histogram_passes=int(work[2][5]) if len(ids) else 0,
                    final_uncertain_candidates=int(work[2][0]) if len(ids) else 0))
        del a,s;torch.cuda.empty_cache()
    save(ROOT/'reports/SCORE_DISTRIBUTION.json',dict(rows=out));print('SCORE_DISTRIBUTION',len(out),flush=True)


if __name__=='__main__':main()
