// 真实checkpoint隔离重放；仅调用生产算子。计时含host提交间隙，kernel纯时间另用nsys。
#include "include/gaussian_renderer.h"
#include "include/mapping_cuda_optimizer.h"
#include "include/anchor_selector_cuda.h"
#include "include/anchor_input_cuda.h"
#include "include/anchor_stats_cuda.h"
#include "include/selector_sampling_cuda.h"
#include "cuda_rasterizer/render_workspace.h"
#include "cuda_rasterizer/ssim_backend.h"
#include "include/loss_utils.h"
#include "include/selector_handoff.h"
#include "improvements/selector/gumbel_network.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <nvToolsExt.h>
#include <json/json.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <functional>
#include <iostream>

namespace fs=std::filesystem;
using Net=improvements::selector::GumbelNetwork;
using Clock=std::chrono::steady_clock;
struct EventPair {
  cudaEvent_t begin,end; double wall;
  EventPair(){cudaEventCreate(&begin);cudaEventCreate(&end);}
  ~EventPair(){cudaEventDestroy(begin);cudaEventDestroy(end);}
};
struct Timer {
  std::map<std::string,std::unique_ptr<EventPair>> events;
  template<class F> void measure(const std::string& name,F f) {
    auto& e=events[name]; if(!e)e=std::make_unique<EventPair>();
    nvtxRangePushA(name.c_str());
    cudaEventRecord(e->begin,at::cuda::getCurrentCUDAStream());auto start=Clock::now();
    f();e->wall=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    cudaEventRecord(e->end,at::cuda::getCurrentCUDAStream());nvtxRangePop();
  }
  void write(std::ofstream& out,const std::string& mode,int rep,int64_t n,int64_t k,double total) {
    for(auto& p:events){float gpu;cudaEventElapsedTime(&gpu,p.second->begin,p.second->end);
      out<<mode<<','<<rep<<','<<n<<','<<k<<','<<p.first<<','<<p.second->wall<<','<<gpu<<','<<total<<'\n';}
  }
};
std::string safe(std::string s){for(auto& c:s)if(c=='.')c='_';return s;}
void put(torch::serialize::OutputArchive& a,const std::string& name,const torch::Tensor& t){
  if(t.defined())a.write(safe(name),t.detach().cpu());
}
void optimizerState(torch::serialize::OutputArchive& a,torch::optim::Adam& optimizer,
                    const std::vector<torch::Tensor>& params,const std::string& prefix){
  for(size_t i=0;i<params.size();++i){auto it=optimizer.state().find(params[i].unsafeGetTensorImpl());
    if(it==optimizer.state().end())continue;
    auto& state=static_cast<torch::optim::AdamParamState&>(*it->second);
    auto key=prefix+std::to_string(i);put(a,key+"_m",state.exp_avg());put(a,key+"_v",state.exp_avg_sq());
    put(a,key+"_step",torch::tensor(state.step(),torch::kLong));}
}

using Stats=std::vector<torch::Tensor>;
// PLY does not carry live window/optimizer state. Restore only fresh window
// buffers in this isolated fixed-topology replay, then call the real model API.
struct ReplayModel:GaussianModel {
  using GaussianModel::GaussianModel;
  void attachBank(improvements::selector::AnchorBank bank){
    explicit_anchors_=bank;auto opts=xyz_.options().requires_grad(false);auto a=bank->size(),n=xyz_.size(0);
    anchor_opacity_accum_=torch::zeros({a},opts.dtype(torch::kFloat64));anchor_visits_=torch::zeros({a},opts.dtype(torch::kLong));
    anchor_slot_opacity_accum_=torch::zeros({n},opts.dtype(torch::kFloat64));anchor_slot_below_prune_=torch::zeros({n},opts.dtype(torch::kLong));anchor_slot_below_cutoff_=torch::zeros_like(anchor_slot_below_prune_);
  }
};
int main(int argc,char** argv){
  TORCH_CHECK(argc>=4,"native_benchmark checkpoint_ply_dir dataset output [repeats]");
  torch::set_num_threads(2);torch::manual_seed(2026);
  const fs::path cp=argv[1],data=argv[2],outdir=argv[3];fs::create_directories(outdir);
  const int repeats=argc>4?std::stoi(argv[4]):60;
  const int bits=argc>5?std::stoi(argv[5]):0;
  const double resolution=argc>6?std::stod(argv[6]):1.;
  renderer_workspace::Scope workspace_scope(bits&32);ssim_backend::Scope ssim_scope(bits&64);
  fs::path cloud;for(auto& entry:fs::directory_iterator(cp/"point_cloud"))if(entry.is_directory())cloud=entry.path();
  auto g=std::make_shared<ReplayModel>(3);g->loadPly(cloud/"point_cloud.ply");g->loadSelectorMetadataPly(cloud/"selector_metadata.ply");
  Net net(.1,true,false,true);net->to(g->getXYZ().device());
  torch::serialize::InputArchive checkpoint;checkpoint.load_from((cp/"selector.pt").string(),g->getXYZ().device());net->load(checkpoint);
  net->anchors()->setFusedFullStats(bits&8);g->attachBank(net->anchors());
  TORCH_CHECK(net->anchors()->variableMembers()&&net->privateCompensation()->independent(),"Wrong topology/R3");
  Json::Value cameras;std::ifstream cameras_in(cp/"cameras.json");cameras_in>>cameras;
  const auto c=cameras[cameras.size()/2];auto cam=std::make_shared<GaussianKeyframe>(0);
  cam->image_width_=int(c["width"].asInt()*resolution);cam->image_height_=int(c["height"].asInt()*resolution);
  cam->FoVx_=2*std::atan(cam->image_width_/(2*(c["fx"].asDouble()*resolution)));
  cam->FoVy_=2*std::atan(cam->image_height_/(2*(c["fy"].asDouble()*resolution)));
  Eigen::Matrix3d r;Eigen::Vector3d pos;
  for(int i=0;i<3;++i){pos[i]=c["position"][i].asDouble();for(int j=0;j<3;++j)r(i,j)=c["rotation"][i][j].asDouble();}
  auto rt=r.transpose().eval();cam->setPose(Eigen::Quaterniond(rt),-rt*pos);cam->set_camera_=true;cam->computeTransformTensors();
  auto rgb=cv::imread((data/c["img_name"].asString()).string());cv::cvtColor(rgb,rgb,cv::COLOR_BGR2RGB);if(resolution!=1.)cv::resize(rgb,rgb,cv::Size(cam->image_width_,cam->image_height_),0,0,cv::INTER_LINEAR);
  auto gt=torch::from_blob(rgb.data,{rgb.rows,rgb.cols,3},torch::kUInt8).clone().to(g->getXYZ().device()).permute({2,0,1}).to(torch::kFloat32).contiguous()/255;
  const auto opts=g->getXYZ().options();const int64_t n=g->getXYZ().size(0);
  // loadPly is an inference loader. Explicitly restore all six production leaves.
  for(auto p:{g->xyz_,g->features_dc_,g->features_rest_,g->opacity_,g->scaling_,g->rotation_})p.requires_grad_(true);
  GaussianOptimizationParams optimization;optimization.opacity_reg_=.001;g->spatial_lr_scale_=1;g->trainingSetup(optimization);
  auto params=std::vector<torch::Tensor>{g->xyz_,g->features_dc_,g->features_rest_,g->opacity_,g->scaling_,g->rotation_};
  auto selector_params=net->parameters();params.insert(params.end(),selector_params.begin(),selector_params.end());
  torch::optim::OptimizerParamGroup group(selector_params);g->optimizer_->add_param_group(group);
  static_cast<torch::optim::AdamOptions&>(g->optimizer_->param_groups().back().options()).lr(.001);
  std::vector<torch::Tensor> original;for(auto p:params)original.push_back(p.detach().clone());
  Stats anchor_stats{g->anchorOpacityAccum(),g->anchorVisits(),g->anchorSlotOpacityAccum(),g->anchorSlotBelowPrune(),g->anchorSlotBelowCutoff()};
  GaussianPipelineParams pipeline;auto bg=torch::zeros({3},opts),override_color=torch::Tensor();
  auto names=net->named_parameters();auto linear=[&](torch::Tensor x,const std::string& p){return torch::linear(x,names[p+".weight"],names[p+".bias"]);};
  std::ofstream csv(outdir/"native_stages.csv");csv<<"case,repeat,N,K,stage,host_submit_ms,cuda_elapsed_ms,cycle_wall_ms\n";
  std::ofstream memory(outdir/"memory.csv");memory<<"case,repeat,allocated_bytes,reserved_bytes,peak_allocated_bytes,allocation_requests,device_allocations,device_frees\n";
  Json::Value metadata;metadata["gaussians"]=Json::Int64(n);metadata["anchors"]=Json::Int64(net->anchors()->size());
  metadata["camera"]=c;metadata["repeats"]=repeats;metadata["warmup"]=10;
  metadata["optimizer"]= "production base Adam; eps1e-15;fresh state warmed;weights restored before each replay, moments advance;no checkpoint optimizer import";
  metadata["gpu_timer"]= "current default stream CUDA events, includes CPU submission gaps and external contention";
  metadata["fixed_input"]=true;metadata["fused_bits"]=bits;metadata["full_opacity_regularizer"]=.001;
  for(const auto& item:names){Json::Value row;row["name"]=item.key();for(auto s:item.value().sizes())row["shape"].append(Json::Int64(s));for(auto s:item.value().strides())row["stride"].append(Json::Int64(s));row["grad"]=item.value().requires_grad();metadata["parameters"].append(row);}
  std::vector<float> ratios{0,.01f,.05f,.1f,.15f};
  for(size_t mode=0;mode<ratios.size();++mode){
    float ratio=ratios[mode];auto label=mode?"lod_"+std::to_string(ratio):"Full";
    {torch::NoGradGuard guard;for(size_t i=0;i<params.size();++i)params[i].copy_(original[i]);}
    torch::Tensor reference_ids,reference_p;
    if(mode){torch::NoGradGuard guard;auto encoded=net->forwardSelectionFeatures(g->getXYZ(),ratio,1,{},g->getScalingActivation(),g->features_dc_,g->getOpacityActivation());
      reference_p=net->maskInactiveSlots(encoded.probability,g->getOpacityActivation());reference_ids=torch::nonzero(g->protection_.cached_uniform<reference_p).reshape({-1});}
    int64_t k=mode?reference_ids.numel():n;
    for(int rep=-10;rep<repeats;++rep){
      {torch::NoGradGuard guard;for(size_t i=0;i<params.size();++i)params[i].copy_(original[i]);}
      g->optimizer_->zero_grad(true);torch::cuda::synchronize();
      auto memory_before=c10::cuda::CUDACachingAllocator::getDeviceStats(0);c10::cuda::CUDACachingAllocator::resetPeakStats(0);
      Timer timer;auto start=Clock::now();nvtxRangePushA(label.c_str());
      torch::Tensor scale,opacity,rotation,feature,offset,input,shared,p,mask,ratio_loss,ids,baseline,residual,loss,l1_value,ssim_value,photo;
      GaussianRenderInput attributes;RenderPackage rendered;
      timer.measure("activations",[&]{scale=g->getScalingActivation();opacity=g->getOpacityActivation();rotation=g->getRotationActivation();});
      if(mode){
        if(bits&4) timer.measure("anchor_input_pack",[&]{input=mapping_cuda::anchorInput27(g->getXYZ(),net->anchors()->features(),net->anchors()->rows(),net->anchors()->centers(),net->anchors()->offsetScales(),scale,g->features_dc_,opacity,ratio);});
        else {
        timer.measure("anchor_lookup",[&]{auto pair=net->anchors()->inputs(g->getXYZ());feature=pair.first;offset=pair.second;});
        timer.measure("input27",[&]{auto attr=torch::cat({scale,g->features_dc_.squeeze(1),opacity},1);input=torch::cat({feature,offset,attr,torch::full({n,1},ratio,opts).log()},1);});
        }
        timer.measure("encoder",[&]{shared=torch::relu(linear(input,"local_emd.0"));});
        timer.measure("selector_head",[&]{p=(linear(torch::relu(linear(shared,"local_emd.2")),"soft_net")/1.0).softmax(-1).select(1,1);});
        timer.measure("sample_budget_compact",[&]{if(bits&16){auto sampled=mapping_cuda::selectorSampleCompact(p,g->protection_.cached_uniform,opacity,net->anchors()->inactiveSlots());p=sampled[0];mask=sampled[1];ids=sampled[2];}
          else{p=net->maskInactiveSlots(p,opacity);mask=sampleSelectorST(p,g->protection_.cached_uniform);ids=torch::nonzero(mask.detach()>0).reshape({-1});}ratio_loss=selectorRatioLoss(p,ratio);});
        timer.measure("selected_gather",[&]{attributes=GaussianRenderInput{g->getXYZ(),opacity,scale,rotation,g->features_dc_,g->features_rest_,g->active_sh_degree_,g->max_sh_degree_}.indexSelect(ids);});
        timer.measure("frozen_compensation_head",[&]{baseline=linear(torch::relu(linear(shared.index_select(0,ids),"compensation.0")),"compensation.2");});
        timer.measure("r3_gather_add",[&]{residual=baseline.detach()+net->privateCompensation()->forward(ids,ratio);});
        timer.measure("attribute_transform",[&]{if(bits&2) {
          auto decoded=mapping_cuda::decode14(attributes.xyz,attributes.scaling,attributes.rotation,g->opacity_.index_select(0,ids),attributes.features_dc,residual,mask.index_select(0,ids));
          attributes.xyz=decoded[0];attributes.scaling=decoded[1];attributes.rotation=decoded[2];attributes.opacity=decoded[3];attributes.features_dc=decoded[4];
        }else {auto transformed=Net::Impl::applyCompensationResidual({attributes.xyz,attributes.scaling.log(),attributes.rotation},residual);
          attributes.xyz=transformed.xyz;attributes.scaling=attributes.scaling*residual.slice(1,3,6).exp();attributes.rotation=transformed.rotation;
          auto appearance=Net::Impl::applyAppearanceResidual(attributes.features_dc,g->opacity_.index_select(0,ids),residual);
          attributes.features_dc=appearance.first;attributes.opacity=appearance.second.sigmoid()*mask.index_select(0,ids).unsqueeze(1);}});
      }else attributes={g->getXYZ(),opacity,scale,rotation,g->features_dc_,g->features_rest_,g->active_sh_degree_,g->max_sh_degree_};
      timer.measure("renderer_forward",[&]{rendered=GaussianRenderer::render(cam,cam->image_height_,cam->image_width_,attributes,pipeline,bg,override_color);});
      if(!mode)timer.measure("anchor_full_statistics",[&]{g->addAnchorPruningStats(rendered.visibility_indices);});
      timer.measure("photo_budget_loss",[&]{l1_value=loss_utils::l1_loss(rendered.image,gt,1);ssim_value=loss_utils::fast_ssim(rendered.image,gt);photo=.8*l1_value+.2*(1-ssim_value);loss=photo;
        if(mode)loss=loss+.01*ratio_loss;else loss=loss+.001*opacity.abs().mean();});
      timer.measure("backward",[&]{loss.backward();});
      torch::serialize::OutputArchive fixture;
      if(rep==0){
        torch::cuda::synchronize();
        for(size_t i=0;i<6;++i)TORCH_CHECK(params[i].grad().defined(),"Missing Master gradient in replay");
        TORCH_CHECK(!mode||torch::equal(ids,reference_ids),"Benchmark changed Cached set");
        TORCH_CHECK(!mode||torch::allclose(p,reference_p,1e-6,1e-7),"Staged network differs from production");
        if(mode){torch::NoGradGuard guard;auto expected=net->compensationResidual(shared.index_select(0,ids),ids,ratio);
          TORCH_CHECK(torch::allclose(residual,expected,1e-6,1e-7),"R3 parity mismatch");}
        put(fixture,"ratio",torch::tensor(ratio));put(fixture,"gt",gt);put(fixture,"owner",net->anchors()->rows());
        put(fixture,"anchor_member_counts",net->anchors()->memberCounts());put(fixture,"anchor_uid",net->anchors()->blockUIDs());put(fixture,"anchor_centers",net->anchors()->centers());put(fixture,"anchor_levels",net->anchors()->levels());
        put(fixture,"gaussian_uid",net->privateCompensation()->uids());put(fixture,"uniform",g->protection_.cached_uniform);
        put(fixture,"input27",input);put(fixture,"shared",shared);put(fixture,"probability",p);put(fixture,"mask",mask);put(fixture,"selected_ids",ids);
        put(fixture,"residual",residual);put(fixture,"xyz",attributes.xyz);put(fixture,"scale",attributes.scaling);put(fixture,"rotation",attributes.rotation);
        put(fixture,"opacity",attributes.opacity);put(fixture,"dc",attributes.features_dc);put(fixture,"render",rendered.image);put(fixture,"loss",loss);
        put(fixture,"l1",l1_value);put(fixture,"ssim",ssim_value);put(fixture,"photo",photo);put(fixture,"budget",ratio_loss);
        put(fixture,"view",cam->world_view_transform_);put(fixture,"projection",cam->full_proj_transform_);put(fixture,"radii",rendered.radii);
        auto layout=[&](const std::string& name,const torch::Tensor& t){if(!t.defined())return;
          Json::Value x;x["name"]=name;x["dtype"]=c10::toString(t.scalar_type());x["device"]=t.device().str();
          for(auto v:t.sizes())x["shape"].append(Json::Int64(v));for(auto v:t.strides())x["stride"].append(Json::Int64(v));
          metadata["layouts"][label].append(x);};
        layout("input27",input);layout("shared",shared);layout("probability",p);layout("selected_ids",ids);
        layout("selected_xyz",attributes.xyz);layout("selected_scale",attributes.scaling);layout("selected_rotation",attributes.rotation);
        layout("selected_opacity",attributes.opacity);layout("selected_dc",attributes.features_dc);layout("render",rendered.image);
        layout("owner",net->anchors()->rows());layout("feature",net->anchors()->features());
        for(size_t i=0;i<params.size();++i){put(fixture,"parameter_before_"+std::to_string(i),params[i]);put(fixture,"gradient_"+std::to_string(i),params[i].grad());}
        optimizerState(fixture,*g->optimizer_,params,"before_");
      }
      timer.measure("adam",[&]{if(!(bits&1)||!mapping_cuda::tryFusedDenseAdam(*g->optimizer_))g->optimizer_->step();});
      nvtxRangePop();torch::cuda::synchronize();
      double wall=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
      auto memory_after=c10::cuda::CUDACachingAllocator::getDeviceStats(0);if(rep>0)memory<<label<<','<<rep<<','<<memory_after.allocated_bytes[0].current<<','<<memory_after.reserved_bytes[0].current<<','<<memory_after.allocated_bytes[0].peak<<','<<memory_after.allocation[0].allocated-memory_before.allocation[0].allocated<<','<<memory_after.segment[0].allocated-memory_before.segment[0].allocated<<','<<memory_after.segment[0].freed-memory_before.segment[0].freed<<'\n';
      if(rep>0)timer.write(csv,label,rep,n,k,wall); // rep0 fixture I/O excluded
      if(rep==0){for(size_t j=0;j<anchor_stats.size();++j)put(fixture,"anchor_stats_"+std::to_string(j),anchor_stats[j]);
        for(size_t i=0;i<params.size();++i)put(fixture,"parameter_after_"+std::to_string(i),params[i]);
        optimizerState(fixture,*g->optimizer_,params,"after_");fixture.save_to((outdir/(label+"_parity.pt")).string());}
    }
    std::cout<<"CASE "<<label<<" N="<<n<<" K="<<k<<std::endl;
  }
  // 真实Anchor布局的低频原语；each trial fresh clone，计时不含clone。
  std::ofstream topo(outdir/"topology_stages.csv");topo<<"repeat,operation,host_ms,completed_ms,N,A\n";
  for(int rep=0;rep<12;++rep){
    auto copy=net->cloneFrozen();copy->anchors()->setFusedFullStats(bits&8);torch::cuda::synchronize();auto t=Clock::now();
    nvtxRangePushA("topology/assert_blocks");copy->anchors()->assertBlocks(n);nvtxRangePop();auto host=std::chrono::duration<double,std::milli>(Clock::now()-t).count();torch::cuda::synchronize();
    topo<<rep<<",assert_blocks,"<<host<<','<<std::chrono::duration<double,std::milli>(Clock::now()-t).count()<<','<<n<<','<<copy->anchors()->size()<<'\n';
    auto keep=torch::ones({n},opts.dtype(torch::kBool));torch::cuda::synchronize();t=Clock::now();nvtxRangePushA("topology/noop_prune_rebuild");
    copy->anchors()->pruneVariableMembers(keep,{});nvtxRangePop();host=std::chrono::duration<double,std::milli>(Clock::now()-t).count();torch::cuda::synchronize();
    topo<<rep<<",noop_prune_rebuild,"<<host<<','<<std::chrono::duration<double,std::milli>(Clock::now()-t).count()<<','<<n<<','<<copy->anchors()->size()<<'\n';
  }
  auto workspace_stats=renderer_workspace::stats();metadata["workspace_hits"]=Json::Int64(workspace_stats.hits);metadata["workspace_growths"]=Json::Int64(workspace_stats.growths);metadata["workspace_retained_bytes"]=Json::Int64(workspace_stats.retained_bytes);metadata["resolution"]=resolution;
  std::ofstream meta(outdir/"native_metadata.json");meta<<metadata;
  std::cout<<"PASS native staged parity and optimizer fixtures"<<std::endl;
}
