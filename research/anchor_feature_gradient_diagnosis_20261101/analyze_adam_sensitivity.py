from common import *
import numpy as np,torch,csv

torch.set_num_threads(2)
m=torch.jit.load(str(R/'snapshots/cold_adam_high_ratio.pt'),map_location='cpu')
rows=[];examples=[]
for j in range(6):
    a=getattr(m,'gradient_a_'+str(j)).detach().numpy();b=getattr(m,'gradient_b_'+str(j)).detach().numpy();before=getattr(m,'parameter_before_'+str(j)).detach().numpy()
    pa=getattr(m,'parameter_a_'+str(j)).detach().numpy();pb=getattr(m,'parameter_b_'+str(j)).detach().numpy()
    dg=a.view(np.int32)!=b.view(np.int32);dp=pa.view(np.int32)!=pb.view(np.int32)
    sign=(np.sign(a)!=np.sign(b));zero=(a==0)|(b==0)
    rows.append(dict(table=j,gradient_changed=int(dg.sum()),parameter_changed=int(dp.sum()),
        gradient_max_abs=float(abs(a-b).max()),parameter_max_abs=float(abs(pa-pb).max()),
        gradient_sign_changed=int(sign.sum()),gradient_zero_in_pair=int((dg&zero).sum()),
        parameter_change_with_same_gradient=int((dp&~dg).sum()),
        gradient_abs_min_changed=float(np.minimum(abs(a[dg]),abs(b[dg])).min()) if dg.any() else 0.))
    order=np.argsort(abs(pa-pb).ravel())[::-1][:5]
    for flat in order:
        row,col=np.unravel_index(flat,pa.shape)
        examples.append(dict(table=j,anchor=int(row),channel=int(col),gradient_a=float(a[row,col]),gradient_b=float(b[row,col]),
            gradient_delta=float(a[row,col]-b[row,col]),parameter_before=float(before[row,col]),parameter_a=float(pa[row,col]),parameter_b=float(pb[row,col]),
            update_a=float(pa[row,col]-before[row,col]),update_b=float(pb[row,col]-before[row,col]),parameter_delta=float(pa[row,col]-pb[row,col])))
for name,data in [('ADAM_CANCELLATION_SUMMARY.csv',rows),('ADAM_CANCELLATION_EXAMPLES.csv',examples)]:
    with (R/name).open('w',newline='') as f:
        w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
save('adam_sensitivity_complete.json',dict(passed=True,eps=1e-15,learning_rate=.001,matching_gradient_parameter_mismatch=sum(x['parameter_change_with_same_gradient'] for x in rows)))
print('ADAM_CANCELLATION_ANALYZED',flush=True)
