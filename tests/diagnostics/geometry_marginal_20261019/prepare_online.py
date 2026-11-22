import importlib.util,json,re,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;R=Path('/home/prmi/zwj/mycartgs')
manifest=json.loads((R/'resultsall/batch_20261018_tum_aligned/manifest.json').read_text())
datasets=[x for x in manifest['datasets']if x['id']in ['tum/rgbd/fr1_room','tum/rgbd/fr1_desk']]
def override(t,d):
    for k,v in d.items():
        pattern=r'^'+re.escape(k)+r'\s*:.*$';line=k+': '+str(v)
        t=re.sub(pattern,lambda _:line,t,flags=re.M)if re.search(pattern,t,re.M)else t+'\n'+line+'\n'
    return t
(ROOT/'configs').mkdir(exist_ok=True)
base=(R/'cfg/selector/fast.yaml').read_text()
for d in datasets:
    scene=d['scene']
    for k in ['orb','mapper']:
        dest=ROOT/'configs'/(scene+'_'+k+'.yaml');shutil.copy2(d[k],dest);d[k]=str(dest)
    for arm in ['raw','marginal','shadow']:
        values={'Selector.max_mapping_iterations':d['target_mapping_iterations'],'GlobalGI.mode':'"streaming_marginal_geo"'if arm=='marginal'else'"streaming"','GlobalGI.performance_logging':1,'GlobalGI.verbose_logging':1,'GlobalGI.marginal_shadow':int(arm=='shadow'),'GlobalGI.marginal_diagnostics':1}
        (ROOT/'configs'/(scene+'_'+arm+'.yaml')).write_text(override(base,values))
runtime=ROOT/'runtime/python';runtime.mkdir(parents=True,exist_ok=True)
for p in(R/'python').glob('*.py'):shutil.copy2(p,runtime/p.name)
spec=importlib.util.find_spec('diff_gaussian_rasterization');package=Path(next(iter(spec.submodule_search_locations)));dest=runtime/'diff_gaussian_rasterization';dest.mkdir(exist_ok=True)
for p in package.iterdir():
    if p.is_file()and p.suffix in ['.so','.py']:shutil.copy2(p,dest/p.name)
protocol=dict(datasets=datasets,seeds=[2026,2027],ratios=[.01,.05],evaluation_views=128,
    evaluation_frames='128 evenly spaced unique common GT RGB indices shared by all four models in a scene; fixed before quality; all associated poses used for ATE',
    train_config='current local project fast.yaml, only fixed mapping budget + GI mode + symmetric diagnostic logging overrides; same original Selector/TF/Protection/Maturity/CD',
    shadow='complete fr1_room seed2026 Raw teacher, exact reference comparison every existing full refresh; require early/middle/late before AB',
    h='first existing full refresh with >=8 valid observed candidates; h=16*median(cbrt(sx*sy*sz)); fixed per map until reset',
    teacher='same percentile/maturity builder; complete marginal sequence encoded as unique FP32 order scores, all candidates, no approximate tail',
    refresh='existing full SGI refresh only, publication remapped on exact topology callbacks; warmup explicitly Raw until first publication',
    primary='production cached-u selected rendering at1/5%; Full quality and SE3 ATE',
    secondary='same absoluteK of production probability using minimum final N across four scene models; diagnostic only',
    order=[dict(scene=s,seed=seed,arm=arm)for s in ['fr1_room','fr1_desk']for seed,arms in [(2026,['raw','marginal']),(2027,['marginal','raw'])]for arm in arms],
    quality_gate='No stable 1/5% regression on fr1_room; no systematic Full/ATE degradation; mechanistic redundancy/coverage evidence; classify >10% mapping overhead or visible refresh stall as performance problem, independent of algorithm quality. n=2 no significance claim.',
    seed_scope='Mapper std/torch RNG and OpenCV RNG use same paired seed; async ORB/SLAM is not bitwise deterministic',commit=False,push=False)
(ROOT/'online_protocol.json').write_text(json.dumps(protocol,indent=2))
print('ONLINE_PROTOCOL_FROZEN',flush=True)
