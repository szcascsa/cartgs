#include <torch/torch.h>
#include <ATen/Context.h>
#include "include/gaussian_renderer.h"
#include "include/transform_field.h"
#include "include/loss_utils.h"
#include "include/selector_handoff.h"
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
void snapshot(std::string prefix,T a,T ema,T maximum,T updates,T hits,T uniform){
  torch::serialize::OutputArchive out;
  for(auto kv:std::vector<std::pair<std::string,T>>{{"coefficient",a},{"gradient_ema",ema},{"max_abs_gradient",maximum},{"cumulative_abs_update",updates},{"gradient_hit_steps",hits},{"cached_uniform",uniform}})out.write(kv.first,kv.second.detach().cpu(),true);
  out.save_to(prefix+".pt");
}
int main(int argc,char** argv){
  TORCH_CHECK(argc==8,"train_worker manifest input.bin outdir ratio seed steps lr");
  torch::set_num_threads(2);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
  int seed=std::stoi(argv[5]),steps=std::stoi(argv[6]);double ratio=std::stod(argv[4]),lr=std::stod(argv[7]);
  torch::manual_seed(seed);FrozenScene scene(readJson(argv[1]));scene.verify();
  std::ifstream f(argv[2],std::ios::binary);int64_t n;f.read((char*)&n,8);std::vector<int64_t> ids(n);std::vector<double> base(n);f.read((char*)ids.data(),n*8);f.read((char*)base.data(),n*8);TORCH_CHECK(f.good(),"Input truncated");
  auto rows=torch::tensor(ids,torch::kInt64).cuda(),b=torch::tensor(base,torch::kFloat64).cuda();
  auto a=torch::zeros_like(b).set_requires_grad(true),ema=torch::zeros_like(b),maximum=torch::zeros_like(b),updates=torch::zeros_like(b),hits=torch::zeros_like(b);
  auto uniform=torch::rand_like(b);torch::optim::Adam adam({a},torch::optim::AdamOptions(lr));
  int64_t K=int64_t(ratio*n+.5);std::mt19937 generator(seed);auto frames=scene.bank();size_t cursor=frames.size();std::vector<int64_t> prev;
  int64_t min_sample=n,max_sample=0;double max_expected_error=0;std::vector<int> visited;
  snapshot(std::string(argv[3])+"/step_0",a,ema,maximum,updates,hits,uniform);
  for(int step=1;step<=steps;++step){
    if(cursor==frames.size()){std::shuffle(frames.begin(),frames.end(),generator);cursor=0;}int frame=frames[cursor++];
    if(step==1){torch::NoGradGuard guard;auto o=ranking(b,rows);o.resize(K);auto chosen=rows.index_select(0,torch::tensor(o,torch::kInt64).cuda());chosen=std::get<0>(chosen.sort());double psnr;auto l=scene.photo(chosen,torch::ones({K},b.options()),frame,float(ratio),&psnr);Json::Value v;v["initial_hard_photo"]=l.item<double>();v["initial_hard_psnr"]=psnr;v["frame"]=frame;emit(v);}
    T score=b+a,probability;double offset;
    {torch::NoGradGuard guard;auto z=score.detach();double low=z.min().item<double>()-40,high=z.max().item<double>()+40;
      for(int it=0;it<42;++it){double mid=.5*(low+high);if((z-mid).sigmoid().sum().item<double>()>K)low=mid;else high=mid;}offset=.5*(low+high);
    }
    probability=(score-offset).sigmoid();auto mask=sampleSelectorST(probability,uniform);auto local=torch::nonzero(mask.detach()>0).flatten();TORCH_CHECK(local.numel()>0,"Empty Bernoulli draw");
    min_sample=std::min(min_sample,local.numel());max_sample=std::max(max_sample,local.numel());visited.push_back(frame);
    max_expected_error=std::max(max_expected_error,std::abs(probability.detach().sum().item<double>()-K));
    double psnr=0;auto loss=scene.photo(rows.index_select(0,local),mask.index_select(0,local),frame,float(ratio),&psnr);
    TORCH_CHECK(torch::isfinite(loss).item<bool>(),"Nonfinite loss");adam.zero_grad();loss.backward();auto gradient=a.grad();TORCH_CHECK(gradient.defined()&&torch::isfinite(gradient).all().item<bool>(),"Bad gradient");
    double lossvalue=loss.item<double>(),gmean=gradient.abs().mean().item<double>(),gmax=gradient.abs().max().item<double>();
    {torch::NoGradGuard guard;auto abs=gradient.abs();ema.mul_(.99).add_(abs,.01);maximum=torch::maximum(maximum,abs);hits.add_((abs>0).to(hits.scalar_type()));}
    auto before=a.detach().clone();adam.step();{torch::NoGradGuard guard;updates.add_((a-before).abs());}TORCH_CHECK(torch::isfinite(a).all().item<bool>(),"Nonfinite coefficient");
    if(step==1||step%50==0||step==steps){
      scene.verify();auto o=ranking(b+a,rows);std::vector<int64_t> selected(o.begin(),o.begin()+K);std::sort(selected.begin(),selected.end());int64_t intersection=0;if(!prev.empty()){std::vector<int64_t> v;std::set_intersection(selected.begin(),selected.end(),prev.begin(),prev.end(),std::back_inserter(v));intersection=v.size();}
      Json::Value r;r["step"]=step;r["frame"]=frame;r["train_loss"]=lossvalue;r["train_psnr"]=psnr;r["sampled_N"]=Json::Int64(local.numel());r["K"]=Json::Int64(K);r["expected_N"]=probability.detach().sum().item<double>();r["offset"]=offset;r["coefficient_norm"]=a.detach().norm().item<double>();r["mean_abs_gradient"]=gmean;r["max_abs_gradient"]=gmax;r["mean_abs_update"]=updates.mean().item<double>();r["changed_fraction"]=prev.empty()?0.:1.-double(intersection)/K;r["frozen_exact"]=true;emit(r);prev=selected;
    }
    if(step==250||step==1000||step==steps)snapshot(std::string(argv[3])+"/step_"+std::to_string(step),a,ema,maximum,updates,hits,uniform);
  }
  scene.verify();Json::Value r;r["completed"]=true;r["frozen_exact"]=true;r["only_coefficient_optimized"]=true;r["steps"]=steps;r["min_sampled_N"]=Json::Int64(min_sample);r["max_sampled_N"]=Json::Int64(max_sample);r["max_expected_K_error"]=max_expected_error;for(auto frame:visited)r["visited_Bank_frames"].append(frame);emit(r);
}
