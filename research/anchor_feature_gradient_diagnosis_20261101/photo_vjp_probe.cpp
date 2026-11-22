#define main diagnostic_adam_main
#include "adam_probe.cpp"
#undef main
#include "include/gaussian_rasterizer.h"
#include "include/anchor_selector_cuda.h"
#include "include/loss_utils.h"
int main(int argc,char** argv) {
  torch::set_num_threads(2);at::globalContext().setAllowTF32CuBLAS(false);const std::string root=argv[1];Scene scene(root,argv[2]);
  scene.fixture=torch::jit::load(std::string(argv[3])+"/office2_render_fixture.pt");scene.fixture.to(torch::kCUDA);scene.xyz=scene.fixture.attr("xyz").toTensor();
  auto t=[&](const std::string& name){return scene.fixture.attr(name).toTensor();};auto a=scene.net->anchors();auto m=scene.net->spatialCompensation();
  auto view=t("view_0"),proj=t("projection_0"),camera=t("camera_0"),fov=t("fov").cpu(),bg=torch::zeros({3},scene.xyz.options()),empty=torch::empty({0},scene.xyz.options()),truth=t("truth_0");
  GaussianRasterizationSettings settings(680,1200,fov[0].item<float>(),fov[1].item<float>(),bg,1,view,proj,3,camera,false,false);
  std::ofstream nodes(root+"/photo_nodes.csv"),adam(root+"/photo_adam.csv"),cases(root+"/photo_cases.csv");
  nodes<<"deterministic,seed,ratio,comparison,rep,node,bits,total,different,max_abs,relative_l2\n";
  adam<<"deterministic,ratio_index,warm,comparison,node,bits,total,different,max_abs,relative_l2\n";
  cases<<"ratio,K,photo_loss,upstream_max_abs,upstream_l2,psnr\n";
  ssim_backend::CacheScope cache(true);ssim_backend::Scope split(true);mapping_cuda::L1FusionScope l1(true);
  std::vector<double> ratios{.01,.05,.1,.2,.3,.4};
  for(int j=0;j<6;++j) {
    auto ratio=ratios[j];auto ids=scene.ids(ratio,j),owners=a->rows().index_select(0,ids),c=a->centers().index_select(0,owners),r=a->offsetScales().index_select(0,owners);
    auto x=scene.xyz.index_select(0,ids),s=t("log_scale").index_select(0,ids),q=t("rotation").index_select(0,ids),op=t("opacity_logit").index_select(0,ids),dc=t("dc").index_select(0,ids),sh=t("sh").index_select(0,ids);
    // Render exactly once, then freeze its VJP to prevent renderer atomics from contaminating compensation localization.
    at::globalContext().setDeterministicAlgorithms(false,false);
    auto delta=m->forward(x,s,q,dc,op,c,r,ratio,true,a->compensationInput(ids,ratio));
    auto mask=torch::ones({ids.numel()},scene.xyz.options());auto decoded=mapping_cuda::decode14AdditiveScale(x,s,q,op,dc,delta,mask);auto xy=torch::zeros_like(x);
    auto image=rasterizeGaussians(decoded[0],xy,decoded[4],sh,empty,decoded[3],decoded[1],decoded[2],empty,settings)[0];
    auto loss=.8*loss_utils::l1_loss(image,truth)+.2*(1-loss_utils::fast_ssim(image,truth));auto upstream=torch::autograd::grad({loss},{delta})[0].detach();
    cases<<ratio<<','<<ids.numel()<<','<<loss.item<double>()<<','<<upstream.abs().max().item<double>()<<','<<upstream.norm().item<double>()<<','<<(-10*torch::log10((image-truth).square().mean())).item<double>()<<'\n';cases.flush();
    for(int deterministic:{0,1}) {
      at::globalContext().setDeterministicAlgorithms(deterministic,false);auto reference=scene.capture(ids,ratio,upstream,false,true);
      auto repeat=scene.capture(ids,ratio,upstream,false,true),candidate=scene.capture(ids,ratio,upstream,true,true);
      compare_capture(nodes,key(deterministic,0,ratio,"baseline_repeat",0),reference,repeat);compare_capture(nodes,key(deterministic,0,ratio,"baseline_vs_fused",0),reference,candidate);nodes.flush();
      AdamBundle initial(scene.tables),pa(scene.tables),pr(scene.tables),pb(scene.tables),rescue(scene.tables);
      for(auto* b:{&pa,&pr,&pb,&rescue})b->copy(initial);
      pa.assign(reference.leaf_gradients);pr.assign(repeat.leaf_gradients);pb.assign(candidate.leaf_gradients);rescue.assign(reference.leaf_gradients);
      for(auto* b:{&pa,&pr,&pb,&rescue})b->step();
      const auto prefix=std::to_string(deterministic)+","+std::to_string(j)+",0,";
      compare(adam,prefix+"baseline_repeat",pa,pr,initial);compare(adam,prefix+"baseline_vs_fused",pa,pb,initial);compare(adam,prefix+"same_gradient_rescue",pa,rescue,initial);adam.flush();
    }
    std::cout<<"FROZEN_REAL_PHOTO_VJP_COMPLETE ratio="<<ratio<<" K="<<ids.numel()<<std::endl;
  }
}
