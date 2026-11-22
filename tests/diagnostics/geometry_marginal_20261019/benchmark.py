"""所有online测量结束后，在隔离CPU条件下完成配对5次微测。"""
import json,subprocess,time,struct,os
from pathlib import Path
import numpy as np
from reference import build
ROOT=Path(__file__).resolve().parent;x=dict(np.load(ROOT/'ranking_inputs.npz'));ref=dict(np.load(ROOT/'reference.npz'));h=json.loads((ROOT/'population.json').read_text())['h'];rows=x['row_ids'][ref['order']];N=len(rows);profiles=[]
# 独立benchmark-only无cache control，不修改已验证的production header。
text=(ROOT/'geometry_ranker.h').read_text();start=text.index('k.resize(n*n);');end=text.index('stat.similarity_ms=ms(t);',start)
text=text[:start]+'input=&p;cover.assign(n,0);taken.assign(n,false);stat.pairs=0;'+text[end:]
text=text.replace('std::vector<size_t> rows;','const std::vector<Point>* input=nullptr;\n double at(size_t i,size_t j)const{if(i==j)return 1.;const auto&a=(*input)[rows[i]];const auto&b=(*input)[rows[j]];double d=0;for(int z=0;z<3;++z)d+=(a.x[z]-b.x[z])*(a.x[z]-b.x[z]);double s=.5*(a.radius+b.radius);return std::exp(-d/(2*s*s+1e-12));}\n std::vector<size_t> rows;')
text=text.replace('k[i*n+j]','at(i,j)').replace('k[v.row*rows.size()+j]','at(v.row,j)').replace('stat.hits+=n;','stat.hits+=0;')
(ROOT/'geometry_ranker_uncached.h').write_text(text);cli=(ROOT/'rank_cli.cpp').read_text().replace('"geometry_ranker.h"','"geometry_ranker_uncached.h"');(ROOT/'rank_cli_uncached.cpp').write_text(cli)
subprocess.run(['g++-9','-std=c++17','-O3',str(ROOT/'rank_cli_uncached.cpp'),'-o',str(ROOT/'rank_cli_uncached')],check=True)
for repeat in range(5):
    t=time.perf_counter();r,p=build(x['xyz'],x['radius'],x['raw_I'],x['row_ids'],h);elapsed=1000*(time.perf_counter()-t);np.testing.assert_array_equal(r['order'],ref['order']);profiles.append(dict(mode='python_reference',repeat=repeat,wall_ms=elapsed))
    (ROOT/'final_python_reference_profile.json').write_text(json.dumps(p,indent=2))
    for mode,limit in [('reference',N),('global',int(.15*N+.5)),('optimized',int(.15*N+.5)),('optimized_full',N),('uncached',int(.15*N+.5))]:
        tag='final_'+mode+'_'+str(repeat);binary='rank_cli_uncached'if mode=='uncached'else'rank_cli';arg=mode if mode in ['reference','global']else'optimized'
        subprocess.run([str(ROOT/binary),str(ROOT/'ranking_input.bin'),str(ROOT/tag),arg,str(limit)],check=True)
        with open(ROOT/(tag+'.bin'),'rb')as f:n=struct.unpack('<q',f.read(8))[0];selected=np.frombuffer(f.read(n*8),dtype='<i8')
        np.testing.assert_array_equal(selected,rows[:n]);d=json.loads((ROOT/(tag+'.json')).read_text());d.update(mode=mode,repeat=repeat);profiles.append(d)
    print('FINAL_BENCHMARK_REPEAT',repeat,flush=True)
(ROOT/'final_offline_profiles.json').write_text(json.dumps(profiles,indent=2))
# 独立timer-only instrumented control：测global heap，不进入speedup或production。
text=(ROOT/'geometry_ranker.h').read_text().replace('global.push({c.head(p,true),ci});','{auto head=c.head(p,true);auto ht=Clock::now();global.push({head,ci});out.merge_ms+=ms(ht);}')
text=text.replace('auto head=global.top();global.pop();','auto ht=Clock::now();auto head=global.top();global.pop();out.merge_ms+=ms(ht);')
text=text.replace('global.push({c.head(p,lazy_local),head.cell});','{auto next=c.head(p,lazy_local);auto ht=Clock::now();global.push({next,head.cell});out.merge_ms+=ms(ht);}')
(ROOT/'geometry_ranker_heap_profile.h').write_text(text);(ROOT/'rank_cli_heap_profile.cpp').write_text((ROOT/'rank_cli.cpp').read_text().replace('"geometry_ranker.h"','"geometry_ranker_heap_profile.h"'))
subprocess.run(['g++-9','-std=c++17','-O3',str(ROOT/'rank_cli_heap_profile.cpp'),'-o',str(ROOT/'rank_cli_heap_profile')],check=True)
subprocess.run([str(ROOT/'rank_cli_heap_profile'),str(ROOT/'ranking_input.bin'),str(ROOT/'heap_profile'),'optimized',str(int(.15*N+.5))],check=True)
with open(ROOT/'heap_profile.bin','rb')as f:
    n=struct.unpack('<q',f.read(8))[0];order=np.frombuffer(f.read(n*8),dtype='<i8');np.testing.assert_array_equal(order,rows[:n])
lib=Path('/lib/x86_64-linux-gnu/libmemusage.so')
if lib.exists():
    for mode,limit in [('reference',N),('optimized',int(.15*N+.5)),('optimized_full',N)]:
        with open(ROOT/('native_allocations_'+mode+'.log'),'w')as log:
            subprocess.run([str(ROOT/'rank_cli'),str(ROOT/'ranking_input.bin'),str(ROOT/('allocation_'+mode)),mode if mode=='reference'else'optimized',str(limit)],stderr=log,env=dict(os.environ,LD_PRELOAD=str(lib)),check=True)
else:(ROOT/'native_allocation_profiler_unavailable.json').write_text(json.dumps(dict(path=str(lib),note='native allocation count unavailable; explicit vector/cache bytes and peak RSS still recorded')))
subprocess.run(['/home/prmi/anaconda3/envs/zwj/bin/python',str(ROOT/'profile_extra.py')],check=True)
