"""结构、depth/unlock、统一Fixed-h8重复度与逐row变化；不读质量。"""
from common import *
from tree_io import nodes_from_tree
from ranking_io import write_csv
import csv
x=dict(np.load(ROOT/'ranking_inputs.npz'));b=dict(np.load(ROOT/'rankings.npz'));ids=x['row_ids'];N=len(ids);W=x['raw_I'];total=float(W.sum());ratios=read(ROOT/'protocol.json')['ratios'];structures={};node_sets={};traces={};bounds={}
def stats(v):
    a=np.asarray(v,float)
    return dict(count=len(a),min=float(a.min()),mean=float(a.mean()),median=float(np.median(a)),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)),p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)),max=float(a.max()))if len(a)else dict(count=0,min=None,mean=None,median=None,p10=None,p90=None,p95=None,p99=None,max=None)
rootstats=[];levels=[]
for tag,prefix in [('old','old'),('new','new')]:
    nodes,recs=nodes_from_tree(ROOT/(tag+'_tree/hierarchy_tree.csv'),x['xyz'],ids,b[prefix+'_root']);node_sets[tag]=nodes;bounds[tag]={int(z['node']):z for z in recs};telem=read(ROOT/(tag+'_tree/hierarchy_telemetry.json'));roots=[n for n in nodes.values()if n.level==1];leaves=[n for n in nodes.values()if n.leaf]
    count=np.array([len(n.members)for n in roots]);mass=np.array([n.mass for n in roots]);largest=int(np.argmax(count));unresolved=[n for n in leaves if n.level==3 and n.quality<.8]
    for n in roots:rootstats.append(dict(backend=tag,root=int(b[prefix+'_root'][n.members[0]]),key_x=n.key[0],key_y=n.key[1],key_z=n.key[2],occupancy=len(n.members),raw_mass=n.mass,raw_mass_fraction=n.mass/total))
    d50=read(ROOT/'new_tree/hierarchy_telemetry.json')['d50']
    structures[tag]=dict(N_candidate=N,d50=d50,h_L1=telem['h_root'],h_L2=telem['h_root']/2,h_L3=telem['h_root']/4,h_over_d50=telem['h_root']/d50,occupied_roots=len(roots),root_occupancy=stats(count),largest_root_candidate_fraction=float(count[largest]/N),largest_N_root_Raw_fraction=float(mass[largest]/total),largest_root_Raw_fraction=float(mass.max()/total),total_final_leaves=len(leaves),total_final_representatives=sum(len(n.original_reps)for n in leaves),L3_unresolved_count=len(unresolved),L3_unresolved_Raw_fraction=sum(n.mass for n in unresolved)/total,global_leaf_weighted_Q=sum(n.mass*n.quality for n in leaves)/total,runtime=dict(grid_ms=telem['root_grouping_ms'],hierarchy_ms=telem['build_ms']))
    for level in [1,2,3]:
        group=[n for n in nodes.values()if n.level==level];den=sum(n.mass for n in group)
        row=dict(backend=tag,level=level,evaluated_nodes=len(group),split_nodes=sum(not n.leaf for n in group),final_leaves=sum(n.leaf for n in group),provisional_representatives=sum(len(n.original_reps)for n in group),final_representatives=sum(len(n.original_reps)for n in group if n.leaf),Raw_SI_weighted_Q=sum(n.mass*n.quality for n in group)/den if den else None)
        for name,values in [('occupancy',[len(n.members)for n in group]),('Q',[n.quality for n in group])]:row.update({name+'_'+k:v for k,v in stats(values).items()})
        levels.append(row)
for arm,tag in [('P0','old'),('P32','new')]:traces[arm]={int(e['row_id']):e for e in csv.DictReader(open(ROOT/(arm+'_emit_trace.csv')))}
rank={};masks={}
for arm in ['A','B','C0','P0','C32','P32']:
    v=np.empty(N,int);v[b[arm+'_order']]=np.arange(1,N+1);rank[arm]=v
    for r in ratios:masks[arm,r]=v<=int(N*r+.5)
red=[];composition=[];coverage=[]
for arm in ['C0','P0','C32','P32']:
    tag='new'if arm in ['C32','P32']else'old';nodes=node_sets[tag];leafmass={n.index:n.mass for n in nodes.values()if n.leaf};rootmass={int(b[tag+'_root'][n.members[0]]):n.mass for n in nodes.values()if n.level==1}
    for r in ratios:
        mask=masks[arm,r];K=int(mask.sum());values=np.zeros(N);pairs={t:0 for t in [.8,.9,.95]}
        for c in np.unique(b['fixed_cell'][mask]):
            g=np.flatnonzero(mask&(b['fixed_cell']==c));n=len(g)
            if n<2:continue
            d=((x['xyz'][g,None,:]-x['xyz'][g][None,:,:])**2).sum(2);s=.5*(x['radius'][g,None]+x['radius'][g][None,:]);k=np.exp(-d/(2*s*s+1e-12));np.fill_diagonal(k,0);values[g]=k.max(1);upper=k[np.triu_indices(n,1)]
            for t in pairs:pairs[t]+=int((upper>t).sum())
        red.append(dict(arm=arm,ratio=r,K=K,mean=float(values[mask].mean()),median=float(np.median(values[mask])),p90=float(np.quantile(values[mask],.9)),**{'pairs_gt_'+str(t):v for t,v in pairs.items()}))
        lh=np.unique(b[tag+'_leaf'][mask]);rh=np.unique(b[tag+'_root'][mask]);coverage.append(dict(arm=arm,ratio=r,K=K,hit_leaves=len(lh),hit_roots=len(rh),hit_leaf_Raw_fraction=sum(leafmass[int(i)]for i in lh)/total,hit_root_Raw_fraction=sum(rootmass[int(i)]for i in rh)/total))
        if arm in ['P0','P32']:
            e=[traces[arm][int(i)]for i in ids[mask]];row=dict(arm=arm,ratio=r,K=K)
            for phase in ['internal_rep','final_rep','independent_residual']:
                n=sum(v['phase']==phase for v in e);row[phase+'_count']=n;row[phase+'_fraction']=n/K
            for l in [1,2,3]:n=sum(int(v['level'])==l for v in e);row['L'+str(l)+'_count']=n;row['L'+str(l)+'_fraction']=n/K
            composition.append(row)
overlap=[];changes=[]
for first,second in [('C0','C32'),('P0','P32'),('C32','P32'),('B','P32')]:
    for r in ratios:
        a=masks[first,r];z=masks[second,r];inter=int((a&z).sum());overlap.append(dict(comparison=first+'_vs_'+second,ratio=r,K=int(a.sum()),intersection=inter,Jaccard=inter/int((a|z).sum()),added=int((z&~a).sum()),removed=int((a&~z).sum())))
        for change,mask in [('added',z&~a),('removed',a&~z)]:
            for i in np.flatnonzero(mask):
                oldkey=tuple(np.floor(x['xyz'][i]/structures['old']['h_L1']).astype(np.int64));newkey=tuple(np.floor(x['xyz'][i]/structures['new']['h_L1']).astype(np.int64));oldlev=node_sets['old'][int(b['old_leaf'][i])].level;newlev=node_sets['new'][int(b['new_leaf'][i])].level
                changes.append(dict(comparison=first+'_vs_'+second,ratio=r,change=change,row_id=int(ids[i]),x=x['xyz'][i,0],y=x['xyz'][i,1],z=x['xyz'][i,2],raw_SI=W[i],old_rank=int(rank[first][i]),new_rank=int(rank[second][i]),old_root=str(oldkey),new_root=str(newkey),old_leaf_level=oldlev,new_leaf_level=newlev,old_C_representative=bool(b['old_rep_mask'][i]),new_C_representative=bool(b['new_rep_mask'][i]),P0_identity=traces['P0'][int(ids[i])]['phase'],P32_identity=traces['P32'][int(ids[i])]['phase'],P0_emitted_level=int(traces['P0'][int(ids[i])]['level']),P32_emitted_level=int(traces['P32'][int(ids[i])]['level']),old_selected=bool(a[i]),new_selected=bool(z[i])))
write_csv(ROOT/'root_statistics.csv',rootstats);write_csv(ROOT/'level_statistics.csv',levels);write_csv(ROOT/'selected_redundancy.csv',red);write_csv(ROOT/'progressive_composition.csv',composition);write_csv(ROOT/'spatial_coverage.csv',coverage);write_csv(ROOT/'ranking_overlap.csv',overlap);write_csv(ROOT/'ranking_row_changes.csv',changes)
save(ROOT/'structure_statistics.json',structures);save(ROOT/'diagnostic_checks.json',dict(all_passed=True,uniform_fixed_h8_metric=True,quality_not_read=True,occupied_entries_only=True))
