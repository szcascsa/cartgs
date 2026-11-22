#include <torch/torch.h>
#include <ATen/Context.h>
#include "include/gaussian_renderer.h"
#include "include/transform_field.h"
#include "include/loss_utils.h"
#include "include/selector_handoff.h"
#include "calibrated_selection.h"
#include <json/json.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
using T=torch::Tensor;
Json::Value readJson(const std::string& p){std::ifstream f(p);Json::Value v;f>>v;return v;}
void emit(Json::Value v){Json::StreamWriterBuilder w;w["indentation"]="";std::cout<<"PG "<<Json::writeString(w,v)<<std::endl;}
class FrozenScene {
  std::shared_ptr<GaussianModel> model_;
  GaussianRenderInput raw_;
  GaussianPipelineParams pipe_;
  TransformField tf_{nullptr};
  T gt_,world_,proj_,center_,params_,bg_,color_,maturity_;
  std::vector<T> originals_,copies_;
  std::vector<int> bank_;
 public:
  FrozenScene(const Json::Value& cfg){
    torch::NoGradGuard guard;
    model_=std::make_shared<GaussianModel>(3);model_->loadPly(cfg["master"].asString());
    originals_={model_->xyz_,model_->features_dc_,model_->features_rest_,model_->opacity_,model_->scaling_,model_->rotation_};
    raw_={model_->getXYZ().detach(),model_->getOpacityActivation().detach(),model_->getScalingActivation().detach(),model_->getRotationActivation().detach(),model_->features_dc_.detach(),model_->features_rest_.detach(),model_->active_sh_degree_,model_->max_sh_degree_};
    maturity_=torch::ones({raw_.xyz.size(0)},raw_.xyz.options());bg_=torch::zeros({3},raw_.xyz.options());
    cv::FileStorage f(cfg["config"].asString(),cv::FileStorage::READ);TORCH_CHECK(f.isOpened(),"Missing config");
    TransformFieldConfig c;c.backend=std::string(f["TransformField.backend"]);c.hexplane.spatial_resolution=int(f["TransformField.spatial_resolution"]);c.hexplane.ratio_resolution=int(f["TransformField.ratio_resolution"]);c.hexplane.feature_dim=int(f["TransformField.feature_dim"]);c.hexplane.multires.clear();for(auto v:f["TransformField.multires"])c.hexplane.multires.push_back(int(v));c.hidden_dim=int(f["TransformField.hidden_dim"]);
    c.maturity_compute_epsilon=0;c.elastic_entry_maturity=0;
    tf_=TransformField(torch::zeros({3}),torch::ones({3}),c);tf_->to(torch::kCUDA);torch::serialize::InputArchive ar;ar.load_from(cfg["tf"].asString(),torch::kCUDA);tf_->load(ar);tf_->eval();
    for(auto& t:tf_->parameters())originals_.push_back(t);for(auto& t:tf_->buffers())originals_.push_back(t);
    for(auto& t:originals_){t.set_requires_grad(false);copies_.push_back(t.clone());}
    torch::serialize::InputArchive cams;cams.load_from(cfg["cameras"].asString(),torch::kCUDA);auto get=[&](const std::string& n){T t;cams.read(n,t,true);return t;};
    gt_=get("gt");world_=get("world");proj_=get("projection");center_=get("center");params_=get("params").cpu();
    for(auto i:cfg["bank_frames"])bank_.push_back(i.asInt());
    originals_.insert(originals_.end(),{gt_,world_,proj_,center_,params_,maturity_});
    copies_.insert(copies_.end(),{gt_.clone(),world_.clone(),proj_.clone(),center_.clone(),params_.clone(),maturity_.clone()});
  }
  T normalized(T rows){auto xyz=raw_.xyz.index_select(0,rows).to(torch::kFloat64);auto lo=tf_->aabbMin().to(torch::kFloat64),hi=tf_->aabbMax().to(torch::kFloat64);TORCH_CHECK((hi>lo).all().item<bool>(),"Invalid frozen AABB");return (2*(xyz-lo)/(hi-lo)-1).detach();}
  T aabbMin(){return tf_->aabbMin();}
  T aabbMax(){return tf_->aabbMax();}
  const std::vector<int>& bank(){return bank_;}
  void verify(){for(size_t i=0;i<originals_.size();++i){TORCH_CHECK(!originals_[i].requires_grad()&&!originals_[i].grad().defined(),"Frozen gradient ",i);TORCH_CHECK(originals_[i].equal(copies_[i]),"Frozen tensor changed ",i);}TORCH_CHECK(maturity_.eq(1).all().item<bool>(),"Maturity changed");}
  T photo(T rows,T mask,int frame,float ratio, double* psnr){
    TORCH_CHECK(std::find(bank_.begin(),bank_.end(),frame)!=bank_.end(),"NON_BANK_TRAINING_FORBIDDEN");
    GaussianRenderInput x;
    {torch::NoGradGuard guard;x=raw_.indexSelect(rows);
      auto local=torch::nonzero(tf_->candidateMask(x.xyz,maturity_.index_select(0,rows))).flatten();
      if(local.numel()){
        auto y=tf_->applyResidual(x.xyz.index_select(0,local),x.scaling.index_select(0,local),x.rotation.index_select(0,local),ratio,torch::ones({local.numel()},x.xyz.options()));
        x.xyz=x.xyz.index_copy(0,local,y.xyz);x.scaling=x.scaling.index_copy(0,local,y.scaling);x.rotation=x.rotation.index_copy(0,local,y.rotation);
      }
    }
    x.opacity=x.opacity*mask.to(x.opacity.scalar_type()).unsqueeze(1);
    auto cam=std::make_shared<GaussianKeyframe>(frame);cam->world_view_transform_=world_[frame];cam->full_proj_transform_=proj_[frame];cam->camera_center_=center_[frame];cam->FoVx_=params_[frame][2].item<float>();cam->FoVy_=params_[frame][3].item<float>();
    auto image=GaussianRenderer::render(cam,params_[frame][0].item<int>(),params_[frame][1].item<int>(),x,pipe_,bg_,color_,1.f,false,false,RasterizeMode::Training).image;
    if(psnr)*psnr=(-10*torch::log10((image.detach().clamp(0,1)-gt_[frame]).square().mean().clamp_min(1e-30))).item<double>();
    auto truth=gt_[frame];return .8f*loss_utils::l1_loss(image,truth,1.f)+.2f*(1-loss_utils::fast_ssim(image,truth));
  }
};
std::vector<int64_t> ranking(T score,T rows){
  auto s=score.detach().cpu().contiguous(),r=rows.cpu().contiguous();auto* sp=s.data_ptr<double>();auto* rp=r.data_ptr<int64_t>();
  std::vector<int64_t> o(score.numel());std::iota(o.begin(),o.end(),0);
  std::sort(o.begin(),o.end(),[&](auto i,auto j){return sp[i]==sp[j]?rp[i]<rp[j]:sp[i]>sp[j];});return o;
}
void snapshot(std::string prefix,torch::nn::Sequential& net,T normalized,T a,T selected_count,T uniform){
  torch::serialize::OutputArchive out;
  for(auto kv:std::vector<std::pair<std::string,T>>{{"normalized_xyz",normalized},{"correction",a},{"direct_selected_count",selected_count},{"cached_uniform",uniform}})out.write(kv.first,kv.second.detach().cpu(),true);
  out.save_to(prefix+".pt");torch::serialize::OutputArchive weights;net->save(weights);weights.save_to(prefix+"_weights.pt");
}
int main(int argc,char** argv){
  TORCH_CHECK(argc==8,"train_worker manifest input.bin outdir ratio seed steps lr");torch::set_num_threads(2);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
  int seed=std::stoi(argv[5]),steps=std::stoi(argv[6]);double ratio=std::stod(argv[4]),lr=std::stod(argv[7]);
  torch::manual_seed(seed);FrozenScene scene(readJson(argv[1]));scene.verify();
  std::ifstream f(argv[2],std::ios::binary);int64_t n;f.read((char*)&n,8);std::vector<int64_t> ids(n);std::vector<double> base(n);f.read((char*)ids.data(),n*8);f.read((char*)base.data(),n*8);TORCH_CHECK(f.good(),"Input truncated");
  auto rows=torch::tensor(ids,torch::kInt64).cuda(),b=torch::tensor(base,torch::kFloat64).cuda();auto uniform=torch::rand_like(b);
  auto normalized=scene.normalized(rows),original_normalized=normalized.clone();
  torch::nn::Sequential net(torch::nn::Linear(3,32),torch::nn::SiLU(),torch::nn::Linear(32,32),torch::nn::SiLU(),torch::nn::Linear(torch::nn::LinearOptions(32,1).bias(false)));net->to(torch::kCUDA,torch::kFloat64);
  {torch::NoGradGuard guard;net->at<torch::nn::LinearImpl>(4).weight.zero_();}
  auto correction=[&](){auto r=net->forward(normalized).flatten();return r-r.mean();};auto a=correction();TORCH_CHECK(a.eq(0).all().item<bool>(),"STEP0_NONZERO_MLP");
  auto raw=ranking(b,rows);TORCH_CHECK(ranking(b+a,rows)==raw,"STEP0_NOT_RAW");
  torch::optim::Adam adam(net->parameters(),torch::optim::AdamOptions(lr));int64_t K=int64_t(ratio*n+.5);auto direct_count=torch::zeros({n},rows.options());std::mt19937 generator(seed);auto frames=scene.bank();size_t cursor=frames.size();std::vector<int64_t> previous;
  std::vector<bool> raw_member(n,false);for(int i=0;i<K;++i)raw_member[raw[i]]=true;
  auto hardBank=[&](int step){torch::NoGradGuard guard;auto a=correction();auto o=ranking(b+a,rows);o.resize(K);auto selected=std::get<0>(rows.index_select(0,torch::tensor(o,torch::kInt64).cuda()).sort());double total=0,ps=0;
    for(auto frame:scene.bank()){double p;auto loss=scene.photo(selected,torch::ones({K},b.options()),frame,float(ratio),&p);double l=loss.item<double>();total+=l;ps+=p;Json::Value z;z["kind"]="hard_view";z["step"]=step;z["frame"]=frame;z["photo"]=l;z["psnr"]=p;emit(z);}
    auto sorted=o;std::sort(sorted.begin(),sorted.end());int64_t intersection=0;if(!previous.empty()){std::vector<int64_t> v;std::set_intersection(sorted.begin(),sorted.end(),previous.begin(),previous.end(),std::back_inserter(v));intersection=v.size();}
    int64_t rawhit=0;for(auto index:o)if(raw_member[index])++rawhit;
    Json::Value z;z["kind"]="hard_summary";z["step"]=step;z["photo"]=total/scene.bank().size();z["psnr"]=ps/scene.bank().size();z["Jaccard_previous"]=previous.empty()?1.:double(intersection)/(2*K-intersection);z["Jaccard_Raw"]=double(rawhit)/(2*K-rawhit);for(auto index:sorted)z["selected_source_rows"].append(Json::Int64(ids[index]));emit(z);previous=sorted;
  };
  hardBank(0);snapshot(std::string(argv[3])+"/step_0",net,normalized,a,direct_count,uniform);
  for(int step=1;step<=steps;++step){
    if(cursor==frames.size()){std::shuffle(frames.begin(),frames.end(),generator);cursor=0;}int frame=frames[cursor++];a=correction();a.retain_grad();auto score=b+a;
    auto calibrated=calibratedSelectionProbability(score,K);auto probability=calibrated.first;double offset=calibrated.second;
    auto mask=sampleSelectorST(probability,uniform);auto local=torch::nonzero(mask.detach()>0).flatten();TORCH_CHECK(local.numel()>0,"Empty Bernoulli sample");
    direct_count.index_add_(0,local,torch::ones({local.numel()},direct_count.options()));double psnr;auto loss=scene.photo(rows.index_select(0,local),mask.index_select(0,local),frame,float(ratio),&psnr);TORCH_CHECK(torch::isfinite(loss).item<bool>(),"NONFINITE_LOSS");
    adam.zero_grad();loss.backward();double grad2=0;std::vector<double> layer2(3,0);int ix=0;
    for(auto& p:net->parameters()){TORCH_CHECK(p.grad().defined()&&torch::isfinite(p.grad()).all().item<bool>(),"NONFINITE_THETA_GRADIENT");double v=p.grad().square().sum().item<double>();grad2+=v;layer2[ix/2]+=v;++ix;}
    auto before=a.detach();adam.step();T after;{torch::NoGradGuard guard;after=correction();}TORCH_CHECK(torch::isfinite(after).all().item<bool>(),"NONFINITE_OUTPUT");
    Json::Value z;z["kind"]="training";z["step"]=step;z["frame"]=frame;z["K"]=Json::Int64(K);z["sampled_N"]=Json::Int64(local.numel());z["expected_N"]=probability.detach().sum().item<double>();z["offset"]=offset;z["sampled_photo"]=loss.item<double>();z["sampled_psnr"]=psnr;z["theta_grad_norm"]=std::sqrt(grad2);for(int j=0;j<3;++j)z["layer_grad_norm"].append(std::sqrt(layer2[j]));z["correction_mean"]=after.mean().item<double>();z["correction_std"]=after.std(false).item<double>();z["correction_max_abs"]=after.abs().max().item<double>();
    z["score_grad_sum"]=a.grad().sum().item<double>();z["score_grad_abs_sum"]=a.grad().abs().sum().item<double>();
    if(step%250==0){
      auto selected=torch::zeros({n},rows.options().dtype(torch::kBool));selected.index_fill_(0,local,true);auto difference=(after-before).abs();auto changed=difference>1e-12;auto sampledN=local.numel();auto selchanged=changed.index_select(0,local).sum().item<int64_t>();int64_t allchanged=changed.sum().item<int64_t>();
      z["output_changed_fraction"]=double(allchanged)/n;z["selected_changed_fraction"]=double(selchanged)/sampledN;z["unselected_changed_fraction"]=double(allchanged-selchanged)/(n-sampledN);z["mean_abs_output_update"]=difference.mean().item<double>();
      torch::serialize::OutputArchive out;for(auto kv:std::vector<std::pair<std::string,T>>{{"correction_before",before},{"correction_after",after},{"selected_mask",selected}})out.write(kv.first,kv.second.cpu(),true);out.save_to(std::string(argv[3])+"/output_update_"+std::to_string(step)+".pt");
    }
    emit(z);
    if(step%100==0||step==250||step==steps){scene.verify();TORCH_CHECK(normalized.equal(original_normalized),"NORMALIZED_XYZ_CHANGED");hardBank(step);}
    if(step==250||step==1000||step==steps)snapshot(std::string(argv[3])+"/step_"+std::to_string(step),net,normalized,after,direct_count,uniform);
  }
  scene.verify();Json::Value end;end["kind"]="completed";end["completed"]=true;end["frozen_exact"]=true;end["only_theta_optimized"]=true;end["steps"]=steps;end["selection_gradient"]="implicit_expected_K_VJP";end["correction_gauge"]="full_population_mean";emit(end);
}
