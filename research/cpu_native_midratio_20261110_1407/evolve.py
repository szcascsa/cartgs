"""从上一版冻结源码生成两个独立候选，不修改上一版或生产。"""
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'cpu_native_small_20261110_1318'
s=(BASE/'source/cpu_small.cpp').read_text(encoding='utf-8').replace('CpuPipelineSmall','CpuPipelineMid')

def swap(old,new):
    global s
    assert s.count(old)==1,(old[:70],s.count(old));s=s.replace(old,new)

swap('a30=a00,a31=a00;','a30=a00,a31=a00,a40=a00,a41=a00,a50=a00,a51=a00;')
swap('if constexpr(Rows>3){t=_mm256_set1_ps(x[xs*3+j]);a30=_mm256_fmadd_ps(t,v0,a30);a31=_mm256_fmadd_ps(t,v1,a31);}',
'''if constexpr(Rows>3){t=_mm256_set1_ps(x[xs*3+j]);a30=_mm256_fmadd_ps(t,v0,a30);a31=_mm256_fmadd_ps(t,v1,a31);}
      if constexpr(Rows>4){t=_mm256_set1_ps(x[xs*4+j]);a40=_mm256_fmadd_ps(t,v0,a40);a41=_mm256_fmadd_ps(t,v1,a41);}
      if constexpr(Rows>5){t=_mm256_set1_ps(x[xs*5+j]);a50=_mm256_fmadd_ps(t,v0,a50);a51=_mm256_fmadd_ps(t,v1,a51);}''')
swap('__m256 lo[4]={a00,a10,a20,a30},hi[4]={a01,a11,a21,a31};','__m256 lo[6]={a00,a10,a20,a30,a40,a50},hi[6]={a01,a11,a21,a31,a41,a51};')
swap('__m128 a0=_mm_setzero_ps(),a1=a0,a2=a0,a3=a0;','__m128 a0=_mm_setzero_ps(),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0;')
swap('if constexpr(Rows>3)a3=_mm_fmadd_ps(_mm_set1_ps(x[xs*3+j]),w,a3);',
'''if constexpr(Rows>3)a3=_mm_fmadd_ps(_mm_set1_ps(x[xs*3+j]),w,a3);
    if constexpr(Rows>4)a4=_mm_fmadd_ps(_mm_set1_ps(x[xs*4+j]),w,a4);
    if constexpr(Rows>5)a5=_mm_fmadd_ps(_mm_set1_ps(x[xs*5+j]),w,a5);''')
swap('__m128 values[4]={a0,a1,a2,a3},bias=', '__m128 values[6]={a0,a1,a2,a3,a4,a5},bias=')
swap('if(rows==4)small_rows<4>', 'if(rows==6)small_rows<6>(x,xs,y,ys,activation);else if(rows==5)small_rows<5>(x,xs,y,ys,activation);else if(rows==4)small_rows<4>')
swap('if(rows==4)fast_rows<4>', 'if(rows==6)fast_rows<6>(x,xs,y,ys,activation);else if(rows==5)fast_rows<5>(x,xs,y,ys,activation);else if(rows==4)fast_rows<4>')
swap('    if(rows==4)apply_rows<4>', '    if(rows>4){apply(x,4,xs,y,ys,activation);apply(x+4*xs,rows-4,xs,y+4*ys,ys,activation);return;}\n    if(rows==4)apply_rows<4>')
swap('int kernel_mode_,ordered_mode_;','int kernel_mode_,ordered_mode_,selection_mode_,comp_batch_;bool histogram_active_=false;\n  std::vector<int> histogram_;std::vector<Key> boundary_;')
# Histogram is built while original scores are produced. No output value changes.
swap('    pool_.run((M_+3)/4,16,[&](int begin,int end,int){',
'''    if(histogram_active_)std::fill(histogram_.begin(),histogram_.end(),0);
    pool_.run((M_+3)/4,16,[&](int begin,int end,int worker){''')
swap('if(!std::isfinite(logits[r][0])||!std::isfinite(logits[r][1])||!std::isfinite(scores_[base+r]))bad_.store(2,std::memory_order_relaxed);',
'''if(!std::isfinite(logits[r][0])||!std::isfinite(logits[r][1])||!std::isfinite(scores_[base+r]))bad_.store(2,std::memory_order_relaxed);
          else if(histogram_active_&&eligible()[id])++histogram_[worker*256+std::min(255,int(scores_[base+r]*256.f))];''')
swap('  void controls(double ratio,int& full', '''  bool histogram_select(){
    int bins[256]={};for(int worker=0;worker<pool_.count();++worker)for(int b=0;b<256;++b)bins[b]+=histogram_[worker*256+b];
    int bucket=255,above=0;for(;bucket>=0;--bucket){if(above+bins[bucket]>=K_)break;above+=bins[bucket];}
    TORCH_CHECK(bucket>=0,"Histogram selection count mismatch");
    // Saturated scores do not benefit: preserve original exact partition path.
    if(bins[bucket]>M_/4)return false;
    boundary_.resize(M_);
    pool_.run(M_,0,[&](int begin,int end,int worker){int n=0;for(int i=begin;i<end;++i){int id=rows_[i];
      if(eligible()[id]&&std::min(255,int(scores_[i]*256.f))==bucket)boundary_[begin+n++]={scores_[i],uint32_t(id)};
    }counts_[worker]=n;});
    int size=0;for(int worker=0;worker<pool_.count();++worker){int begin=int(int64_t(M_)*worker/pool_.count()),n=counts_[worker];std::memmove(boundary_.data()+size,boundary_.data()+begin,n*sizeof(Key));size+=n;}
    TORCH_CHECK(size==bins[bucket]&&K_>above&&K_-above<=size,"Histogram boundary size mismatch");
    int rank=K_-above-1;std::nth_element(boundary_.begin(),boundary_.begin()+rank,boundary_.begin()+size,Better{});
    Key cutoff=boundary_[rank];++cutoff.id; // IDs < INT_MAX; strict Better includes exactly the worst selected ID.
    ordered_compact(cutoff,false,0,K_);return true;
  }
  void controls(double ratio,int& full''')
swap('    pool_.run((K_+3)/4,16,[&](int begin,int end,int){','    int batch=comp_batch_;pool_.run((K_+batch-1)/batch,16,[&](int begin,int end,int){')
swap('h0[4][64],h1[4][64],hidden[4][256],small[4][4],delta[4][14],input[4][56]', 'h0[6][64],h1[6][64],hidden[6][256],small[6][4],delta[6][14],input[6][56]')
swap('int base=block*4,R=std::min(4,K_-base);','int base=block*batch,R=std::min(batch,K_-base);')
swap('bool small_kernel=false,int ordered_mode=0)', 'bool small_kernel=false,int ordered_mode=0,int selection_mode=0,int comp_batch=4)')
swap('ordered_mode_(ordered_mode),pool_(threads)', 'ordered_mode_(ordered_mode),selection_mode_(selection_mode),comp_batch_(comp_batch),pool_(threads)')
swap('    activated_scale_.resize(N_*3);', '    TORCH_CHECK(selection_mode>=0&&selection_mode<=1&&(comp_batch==4||comp_batch==6),"Midratio optimization modes");\n    if(selection_mode_){histogram_.resize(threads*256);boundary_.resize(N_);}\n    activated_scale_.resize(N_*3);')
swap('    bytes_+=int64_t(activated_scale_.size()+activated_opacity_.size())*4;', '    bytes_+=int64_t(activated_scale_.size()+activated_opacity_.size()+histogram_.size())*4+int64_t(boundary_.size())*sizeof(Key);')
swap('      keys_.resize(M_);selector(', '      histogram_active_=selection_mode_&&kind<2&&M_>=2048;keys_.resize(M_);selector(')
swap('      }else{\n        if(K_<int(keys_.size()))std::nth_element', '      }else if(!(histogram_active_&&histogram_select())){\n        if(K_<int(keys_.size()))std::nth_element')
swap('int,bool,bool,bool,bool,int,bool,int>()', 'int,bool,bool,bool,bool,int,bool,int,int,int>()')
swap('pybind11::arg("ordered_mode")=0)', 'pybind11::arg("ordered_mode")=0,pybind11::arg("selection_mode")=0,pybind11::arg("comp_batch")=4)')
(HERE/'source/cpu_mid.cpp').write_text(s,encoding='utf-8')
(HERE/'source/native_handoff.h').write_text((BASE/'source/native_handoff.h').read_text(encoding='utf-8').replace('CpuPipelineSmall','CpuPipelineMid').replace('NativeHandoffSmall','NativeHandoffMid').replace('NativeTicketSmall','NativeTicketMid'),encoding='utf-8')
r=(BASE/'small_runtime.py').read_text(encoding='utf-8').replace('mycartgs_cpu_small','mycartgs_cpu_mid').replace('cpu_small.cpp','cpu_mid.cpp')
r=r.replace('small_kernel=True,ordered_mode=3):','small_kernel=True,ordered_mode=3,selection_mode=0,comp_batch=4):').replace('kernel_mode,small_kernel,ordered_mode)','kernel_mode,small_kernel,ordered_mode,selection_mode,comp_batch)')
(HERE/'mid_runtime.py').write_text(r,encoding='utf-8')
print('MID_GENERATED',len(s),flush=True)
