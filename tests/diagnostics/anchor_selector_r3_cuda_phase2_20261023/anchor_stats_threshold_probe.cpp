// ULP级阈值专项：同时记录Legacy自身重复波动，不能用FP64 allclose替代prune决策。
#define main anchor_statistics_original_main
#include "tests/anchor_selector/fused_anchor_stats_test.cpp"
#undef main
int main(int argc,char** argv){try{
  TORCH_CHECK(argc==4,"threshold_probe checkpoint fixture output");torch::set_num_threads(2);fs::path cp=argv[1],out=argv[3];fs::create_directories(out);
  Net net(.1,true,false,true);net->to(torch::kCUDA);torch::serialize::InputArchive ar;ar.load_from((cp/"selector.pt").string(),torch::kCUDA);net->load(ar);
  auto ma=net->cloneFrozen(),mb=net->cloneFrozen();auto a=ma->anchors(),b=mb->anchors();a->setFusedFullStats(false);b->setFusedFullStats(true);
  torch::serialize::InputArchive f;f.load_from(argv[2],torch::kCUDA);T logits;f.read("parameter_before_3",logits);auto opacity=logits.sigmoid()*.001,visible=torch::arange(logits.size(0),a->rows().options());
  auto increment=initial(a);legacy(a,visible,opacity,increment);auto target=.005*torch::full({a->size()},402.,increment[0].options());auto base=target-increment[0];
  std::ofstream csv(out/"threshold_prune.csv");csv<<"direction,ulp_steps,legacy_repeat_mask_differences,fused_mask_differences,max_opacity_abs,worst_anchor,legacy_O,fused_O,threshold\n";
  csv.precision(17);int64_t failures=0;
  for(int direction:{-1,1})for(int steps:{0,1,2,8,64})for(int repeat=0;repeat<8;++repeat){auto sa=initial(a),sb=initial(b),sc=initial(a);auto before=base.clone();
    for(int i=0;i<steps;++i)before=torch::nextafter(before,torch::full_like(before,direction*std::numeric_limits<double>::infinity()));
    sa[0].copy_(before);sb[0].copy_(before);sc[0].copy_(before);sa[1].fill_(401);sb[1].fill_(401);sc[1].fill_(401);
    legacy(a,visible,opacity,sa);legacy(a,visible,opacity,sc);fused(b,visible,opacity,sb);
    auto pa=sa[0]<target,pb=sb[0]<target,pc=sc[0]<target;auto bad=pa!=pb;auto count=bad.sum().item<int64_t>();failures+=count;
    auto id=count?torch::nonzero(bad).reshape({-1})[0].item<int64_t>():0;
    csv<<direction<<','<<steps<<','<<(pa!=pc).sum().item<int64_t>()<<','<<count<<','<<(sa[0]-sb[0]).abs().max().item<double>()<<','<<id<<','<<sa[0][id].item<double>()<<','<<sb[0][id].item<double>()<<','<<target[id].item<double>()<<'\n';
  }
  std::cout<<"threshold mask differences="<<failures<<'\n';return failures?2:0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
