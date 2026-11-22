#include "include/anchor_si.h"
#include "include/gaussian_renderer.h"
#include "include/selector_handoff.h"
#include "include/selector_sampling_cuda.h"
#include "include/anchor_selector_cuda.h"
#include "include/mapping_cuda_optimizer.h"
#include "include/loss_utils.h"
#include "cuda_rasterizer/anchor_si_capture.h"
#include "cuda_rasterizer/ssim_backend.h"
#include "cuda_rasterizer/render_workspace.h"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <algorithm>
#include <json/json.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

int main(int argc,char** argv) { try {
  TORCH_CHECK(argc==7,"anchor_si_frozen_train checkpoint dataset output arm seed steps");
  namespace fs=std::filesystem;const fs::path cp=argv[1],data=argv[2],out=argv[3];fs::create_directories(out);
  const std::string arm=argv[4];TORCH_CHECK(arm=="off"||arm=="anchor"||arm=="gaussian","Unknown arm");
  const int seed=std::stoi(argv[5]),steps=std::stoi(argv[6]);torch::set_num_threads(2);torch::manual_seed(2026);
  std::mt19937 order_rng(seed);renderer_workspace::Scope workspace(true);ssim_backend::Scope ssim(true);
  auto g=std::make_shared<GaussianModel>(3);fs::path cloud;for(auto p:fs::directory_iterator(cp/"point_cloud"))if(p.is_directory())cloud=p.path();
  g->loadPly(cloud/"point_cloud.ply");g->loadSelectorMetadataPly(cloud/"selector_metadata.ply");
  auto net=improvements::selector::GumbelNetwork(.1,true,false,true);net->to(torch::kCUDA);
  torch::serialize::InputArchive archive;archive.load_from((cp/"selector.pt").string(),torch::kCUDA);net->load(archive);
  net->setFusedInputPack(true);net->anchors()->setFusedFullStats(true);net->anchors()->assertBlocks(g->getXYZ().size(0));
  std::vector<torch::Tensor> head;for(auto& item:net->named_parameters()) {
    bool train=item.key().rfind("local_emd.2.",0)==0 || item.key().rfind("soft_net.",0)==0;
    item.value().requires_grad_(train);if(train)head.push_back(item.value());
  }
  torch::optim::Adam optimizer(head,torch::optim::AdamOptions(.001).eps(1e-15));
  std::vector<std::pair<torch::Tensor,torch::Tensor>> fixed;
  for(auto parameter:net->parameters())if(!parameter.requires_grad())fixed.push_back({parameter,parameter.detach().clone()});
  for(auto parameter:{g->xyz_,g->scaling_,g->rotation_,g->opacity_,g->features_dc_,g->features_rest_})fixed.push_back({parameter,parameter.detach().clone()});
  auto original_uids=net->privateCompensation()->uids().clone(),original_owner=net->anchors()->rows().clone();
  Json::Value cameras;std::ifstream input(cp/"cameras.json");input>>cameras;
  struct View { std::shared_ptr<GaussianKeyframe> camera;torch::Tensor gt; };
  std::vector<View> views;auto opts=g->getXYZ().options().requires_grad(false);
  for(const auto& c:cameras) {
    auto camera=std::make_shared<GaussianKeyframe>(views.size());camera->image_width_=c["width"].asInt();camera->image_height_=c["height"].asInt();
    camera->FoVx_=2*std::atan(camera->image_width_/(2*c["fx"].asDouble()));camera->FoVy_=2*std::atan(camera->image_height_/(2*c["fy"].asDouble()));
    Eigen::Matrix3d r;Eigen::Vector3d position;for(int i=0;i<3;++i){position[i]=c["position"][i].asDouble();for(int j=0;j<3;++j)r(i,j)=c["rotation"][i][j].asDouble();}
    auto rt=r.transpose().eval();camera->setPose(Eigen::Quaterniond(rt),-rt*position);camera->set_camera_=true;camera->computeTransformTensors();
    auto image=cv::imread((data/c["img_name"].asString()).string());TORCH_CHECK(!image.empty(),"Missing Bank image");cv::cvtColor(image,image,cv::COLOR_BGR2RGB);
    auto gt=torch::from_blob(image.data,{image.rows,image.cols,3},torch::kUInt8).clone().to(torch::kCUDA).permute({2,0,1}).to(torch::kFloat32).contiguous()/255;
    views.push_back({camera,gt});
  }
  auto xyz=g->getXYZ().detach(),scale=g->getScalingActivation().detach(),rotation=g->getRotationActivation().detach(),opacity=g->getOpacityActivation().detach();
  auto dc=g->features_dc_.detach(),sh=g->features_rest_.detach(),logit=g->opacity_.detach();
  GaussianRenderInput full{xyz,opacity,scale,rotation,dc,sh,g->active_sh_degree_,g->max_sh_degree_};
  auto bg=torch::zeros({3},opts),override_color=torch::Tensor();GaussianPipelineParams pipeline;
  anchor_si::Config config;config.enabled=arm!="off";anchor_si::State state(config);
  const auto n=xyz.size(0);auto row_options=net->anchors()->rows().options();auto knots=torch::tensor({.01f,.05f,.1f,.15f},opts);
  auto gaussian_ema=torch::zeros({n},opts),gaussian_active=torch::zeros_like(gaussian_ema);
  auto gaussian_visits=torch::zeros({n},row_options),gaussian_last=torch::zeros_like(gaussian_visits),gaussian_labels=torch::Tensor();
  auto order=std::vector<int>(views.size());std::iota(order.begin(),order.end(),0);std::shuffle(order.begin(),order.end(),order_rng);
  std::ofstream curve(out/"training.csv");curve<<"step,view,ratio,N,K,photo,budget,si,si_grad_norm,photo_budget_grad_norm,gradient_cosine,expected_K\n";
  std::ofstream timing(out/"timing.csv");timing<<"step,cuda_ms\n";
  cudaEvent_t begin,end;cudaEventCreate(&begin);cudaEventCreate(&end);std::vector<float> ratios{.01,.05,.1,.15};
  for(int step=0;step<steps;++step) {
    if(step && step%views.size()==0)std::shuffle(order.begin(),order.end(),order_rng);
    const auto id=order[step%views.size()];auto& view=views[id];const int ri=step%4;const float ratio=ratios[ri];
    optimizer.zero_grad(true);cudaEventRecord(begin);
    bool capture=arm!="off"&&state.captureNextFull();
    {
      torch::NoGradGuard guard;anchor_si::CaptureScope scope(capture);
      auto rendered=GaussianRenderer::render(view.camera,view.camera->image_height_,view.camera->image_width_,full,pipeline,bg,override_color);
      if(capture && arm=="anchor")state.observe(net,rendered.frame_importance,opacity,int64_t(view.camera->image_width_)*view.camera->image_height_,1,ratios);
      if(capture && arm=="gaussian") {
        auto inactive=net->anchors()->inactiveSlots();auto active=~(inactive&(opacity.reshape({-1})<1.f/255.f));
        auto value=rendered.frame_importance/(double(view.camera->image_width_)*view.camera->image_height_);
        auto seen=(value>0)&active;gaussian_visits.masked_fill_(gaussian_active!=active.to(torch::kFloat32),0);
        gaussian_ema.copy_(torch::where(seen,torch::where(gaussian_visits>0,.9*gaussian_ema+.1*value,value),gaussian_ema));
        gaussian_visits.add_(seen.to(torch::kLong));gaussian_last.masked_fill_(seen,step+1);gaussian_active=active.to(torch::kFloat32);
        gaussian_labels=anchor_si::targets(gaussian_ema,gaussian_visits,gaussian_last,gaussian_active,knots,step+1,n,4,128,1);
      }
    }
    auto selection=net->forwardSelectionFeatures(xyz,ratio,1,{},scale,dc,opacity);
    auto sampled=mapping_cuda::selectorSampleCompact(selection.probability,g->protection_.cached_uniform,opacity,net->anchors()->inactiveSlots());
    auto probability=sampled[0],mask=sampled[1],ids=sampled[2];auto attributes=full.indexSelect(ids);
    auto residual=net->compensationResidual(selection.shared.index_select(0,ids),ids,ratio);
    auto decoded=mapping_cuda::decode14(attributes.xyz,attributes.scaling,attributes.rotation,logit.index_select(0,ids),attributes.features_dc,residual,mask.index_select(0,ids));
    attributes.xyz=decoded[0];attributes.scaling=decoded[1];attributes.rotation=decoded[2];attributes.opacity=decoded[3];attributes.features_dc=decoded[4];
    auto rendered=GaussianRenderer::render(view.camera,view.camera->image_height_,view.camera->image_width_,attributes,pipeline,bg,override_color);
    auto photo=.8*loss_utils::l1_loss(rendered.image,view.gt,1)+.2*(1-loss_utils::fast_ssim(rendered.image,view.gt));
    auto budget=selectorRatioLoss(probability,ratio),loss=photo+.01*budget,si=torch::Tensor();
    if(arm=="anchor")si=state.loss(net,probability,opacity,ri,1);
    if(arm=="gaussian"&&gaussian_labels.defined()) {
      auto weights=((gaussian_visits>=4)&(step+1-gaussian_last<=128)&(gaussian_active>0)&(gaussian_labels[ri]>=0)).to(torch::kFloat32);
      si=.01*((probability-gaussian_labels[ri].clamp_min(0)).square()*weights).sum()/weights.sum().clamp_min(1);
    }
    state.accumulateHeadGradients(si,net);std::vector<torch::Tensor> si_grad;
    if(step%16==0)for(auto p:head)si_grad.push_back(p.grad().defined()?p.grad().clone():torch::zeros_like(p));
    loss.backward();double gs=0,gp=0,dot=0;
    if(step%16==0)for(size_t j=0;j<head.size();++j) {
      auto pg=head[j].grad()-si_grad[j];gs+=si_grad[j].square().sum().item<double>();gp+=pg.square().sum().item<double>();dot+=(pg*si_grad[j]).sum().item<double>();
    }
    TORCH_CHECK(mapping_cuda::tryFusedDenseAdam(optimizer),"Fused head Adam unsupported");
    cudaEventRecord(end);cudaEventSynchronize(end);float ms;cudaEventElapsedTime(&ms,begin,end);timing<<step<<','<<ms<<'\n';
    if(step%16==0||step==steps-1)curve<<step<<','<<id<<','<<ratio<<','<<n<<','<<ids.numel()<<','<<photo.item<double>()<<','<<budget.item<double>()<<','<<(si.defined()?si.item<double>():0)<<','<<std::sqrt(gs)<<','<<std::sqrt(gp)<<','<<dot/std::max(1e-30,std::sqrt(gs*gp))<<','<<probability.sum().item<double>()<<'\n';
    if(step%200==0)std::cout<<"STEP "<<step<<" arm="<<arm<<" photo="<<photo.item<double>()<<" SI="<<(si.defined()?si.item<double>():0)<<std::endl;
  }
  cudaEventDestroy(begin);cudaEventDestroy(end);torch::serialize::OutputArchive saved;net->save(saved);saved.save_to((out/"selector.pt").string());
  torch::serialize::OutputArchive optimizer_state;optimizer.save(optimizer_state);optimizer_state.save_to((out/"optimizer.pt").string());
  for(const auto& pair:fixed)TORCH_CHECK(torch::equal(pair.first,pair.second)&&!pair.first.grad().defined(),"Frozen parameter changed or received gradient");
  TORCH_CHECK(torch::equal(original_uids,net->privateCompensation()->uids())&&torch::equal(original_owner,net->anchors()->rows()),"Frozen Gaussian identity/owner changed");
  if(arm=="anchor")state.save(out/"anchor_si_state.pt");
  Json::Value summary;summary["arm"]=arm;summary["seed"]=seed;summary["steps"]=steps;summary["N"]=Json::Int64(n);summary["anchors"]=Json::Int64(net->anchors()->size());summary["bank_views"]=int(views.size());summary["head_adam_steps"]=steps;
  std::ofstream metadata(out/"metadata.json");metadata<<summary;
  std::cout<<"PASS frozen map/head-only training complete\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;} }
