#include <torch/torch.h>
#include <ATen/Context.h>
#include "include/gaussian_renderer.h"
#include "include/transform_field.h"
#include "include/loss_utils.h"
#include "pl_selection.h"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <chrono>
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
  T photo(T rows,int frame,float ratio, double* psnr){
    torch::NoGradGuard photo_guard;
    TORCH_CHECK(std::find(bank_.begin(),bank_.end(),frame)!=bank_.end(),"NON_BANK_TRAINING_FORBIDDEN");
    GaussianRenderInput x;
    {torch::NoGradGuard guard;x=raw_.indexSelect(rows);
      auto local=torch::nonzero(tf_->candidateMask(x.xyz,maturity_.index_select(0,rows))).flatten();
      if(local.numel()){
        auto y=tf_->applyResidual(x.xyz.index_select(0,local),x.scaling.index_select(0,local),x.rotation.index_select(0,local),ratio,torch::ones({local.numel()},x.xyz.options()));
        x.xyz=x.xyz.index_copy(0,local,y.xyz);x.scaling=x.scaling.index_copy(0,local,y.scaling);x.rotation=x.rotation.index_copy(0,local,y.rotation);
      }
    }
    auto cam=std::make_shared<GaussianKeyframe>(frame);cam->world_view_transform_=world_[frame];cam->full_proj_transform_=proj_[frame];cam->camera_center_=center_[frame];cam->FoVx_=params_[frame][2].item<float>();cam->FoVy_=params_[frame][3].item<float>();
    auto image=GaussianRenderer::render(cam,params_[frame][0].item<int>(),params_[frame][1].item<int>(),x,pipe_,bg_,color_,1.f,false,false,RasterizeMode::Inference).image;
    if(psnr)*psnr=(-10*torch::log10((image.detach().clamp(0,1)-gt_[frame]).square().mean().clamp_min(1e-30))).item<double>();
    auto truth=gt_[frame];return .8f*loss_utils::l1_loss(image,truth,1.f)+.2f*(1-loss_utils::fast_ssim(image,truth));
  }
};
std::vector<int64_t> ranking(T score,T rows){
  auto s=score.detach().cpu().contiguous(),r=rows.cpu().contiguous();auto* sp=s.data_ptr<double>();auto* rp=r.data_ptr<int64_t>();std::vector<int64_t> o(score.numel());std::iota(o.begin(),o.end(),0);
  std::sort(o.begin(),o.end(),[&](auto i,auto j){return sp[i]==sp[j]?rp[i]<rp[j]:sp[i]>sp[j];});return o;
}
using Clock=std::chrono::steady_clock;
void sync(){torch::cuda::synchronize();}
double elapsed(Clock::time_point p){return std::chrono::duration<double,std::milli>(Clock::now()-p).count();}
void snapshot(std::string prefix,T a,T ema,T maximum,T updates,T hits,T baseline){
  torch::serialize::OutputArchive out;
  for(auto kv:std::vector<std::pair<std::string,T>>{{"coefficient",a},{"gradient_ema",ema},{"max_abs_gradient",maximum},{"cumulative_abs_update",updates},{"gradient_hit_steps",hits},{"frame_baseline",baseline},{"cuda_rng_state",at::cuda::detail::getDefaultCUDAGenerator().get_state()}})out.write(kv.first,kv.second.detach().cpu(),true);
  out.save_to(prefix+".pt");
}
int main(int argc,char** argv){
  TORCH_CHECK(argc==8,"train_worker manifest input.bin outdir ratio seed steps lr");torch::set_num_threads(2);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
  int seed=std::stoi(argv[5]),steps=std::stoi(argv[6]);double ratio=std::stod(argv[4]),lr=std::stod(argv[7]);
  torch::manual_seed(seed);FrozenScene scene(readJson(argv[1]));scene.verify();torch::manual_seed(seed);
  std::ifstream f(argv[2],std::ios::binary);int64_t n;f.read((char*)&n,8);std::vector<int64_t> ids(n);std::vector<double> base(n);f.read((char*)ids.data(),n*8);f.read((char*)base.data(),n*8);TORCH_CHECK(f.good(),"Input truncated");
  auto rows=torch::tensor(ids,torch::kInt64).cuda(),b=torch::tensor(base,torch::kFloat64).cuda();auto a=torch::zeros_like(b).set_requires_grad(true),ema=torch::zeros_like(b),maximum=torch::zeros_like(b),updates=torch::zeros_like(b),hits=torch::zeros_like(b);
  torch::optim::Adam adam({a},torch::optim::AdamOptions(lr));int64_t K=int64_t(ratio*n+.5);std::mt19937 generator(seed);auto frames=scene.bank();size_t cursor=frames.size();
  int maxframe=*std::max_element(frames.begin(),frames.end());std::vector<double> baseline(maxframe+1,0);std::vector<int64_t> previous;auto raw_order=ranking(b,rows);std::vector<double> rawpct(n);for(int i=0;i<n;++i)rawpct[raw_order[i]]=double(i)/(n-1);auto rankpct=torch::tensor(rawpct,torch::kFloat64).cuda();
  auto hardBank=[&](int step){
    auto order=ranking(b+a,rows);order.resize(K);auto selected=std::get<0>(rows.index_select(0,torch::tensor(order,torch::kInt64).cuda()).sort());double ploss=0,psnr=0;
    for(auto frame:scene.bank()){double p;auto loss=scene.photo(selected,frame,float(ratio),&p);double l=loss.item<double>();ploss+=l;psnr+=p;if(step==0)baseline[frame]=l;Json::Value z;z["kind"]="hard_view";z["step"]=step;z["frame"]=frame;z["photo"]=l;z["psnr"]=p;emit(z);}
    std::sort(order.begin(),order.end());int64_t intersection=0;if(!previous.empty()){std::vector<int64_t> v;std::set_intersection(order.begin(),order.end(),previous.begin(),previous.end(),std::back_inserter(v));intersection=v.size();}
    Json::Value z;z["kind"]="hard_summary";z["step"]=step;z["photo"]=ploss/scene.bank().size();z["psnr"]=psnr/scene.bank().size();z["changed_fraction"]=previous.empty()?0.:1.-double(intersection)/K;for(auto index:order)z["selected_source_rows"].append(Json::Int64(ids[index]));emit(z);previous=order;
  };
  hardBank(0);auto baset=torch::tensor(baseline,torch::kFloat64);snapshot(std::string(argv[3])+"/step_0",a,ema,maximum,updates,hits,baset);T previous_signature;
  for(int step=1;step<=steps;++step){
    sync();auto start=Clock::now();if(cursor==frames.size()){std::shuffle(frames.begin(),frames.end(),generator);cursor=0;}int frame=frames[cursor++];
    auto score=b+a;auto z=score;sync();auto tick=Clock::now();auto[order,g]=sampleFreshGumbelTopK(z,K);sync();double sampling_ms=elapsed(tick);
    auto signature=g.slice(0,0,8).cpu();bool fresh=!previous_signature.defined()||!torch::equal(signature,previous_signature);TORCH_CHECK(fresh,"REUSED_GUMBEL_NOISE");previous_signature=signature;
    tick=Clock::now();auto probability=plLogProbability(z,order);sync();double log_prob_ms=elapsed(tick);
    auto selected_rows=std::get<0>(rows.index_select(0,order).sort());TORCH_CHECK(selected_rows.numel()==K,"SELECTED_COUNT_NOT_K");TORCH_CHECK((selected_rows.slice(0,1)>selected_rows.slice(0,0,K-1)).all().item<bool>(),"DUPLICATE_SELECTED_ROW");
    tick=Clock::now();double train_psnr;auto photo=scene.photo(selected_rows,frame,float(ratio),&train_psnr);sync();double render_ms=elapsed(tick);
    TORCH_CHECK(!photo.requires_grad()&&torch::isfinite(photo).item<bool>(),"RENDERER_MUST_BE_DETACHED_FINITE");double loss=photo.item<double>(),oldbaseline=baseline[frame],advantage=loss-oldbaseline;
    auto surrogate=probability.log_prob*torch::tensor(advantage,z.options()).detach();TORCH_CHECK(torch::isfinite(surrogate).item<bool>(),"NONFINITE_SURROGATE");
    tick=Clock::now();adam.zero_grad();surrogate.backward();auto gradient=a.grad();TORCH_CHECK(gradient.defined()&&torch::isfinite(gradient).all().item<bool>(),"NONFINITE_GRADIENT");auto before=a.detach().clone();adam.step();sync();double backward_ms=elapsed(tick);
    // Current sample updates the baseline only after optimizer.step().
    baseline[frame]=.99*oldbaseline+.01*loss;
    auto abs=gradient.abs();T delta;
    {torch::NoGradGuard guard;ema.mul_(.99).add_(abs,.01);maximum=torch::maximum(maximum,abs);hits.add_((abs>0).to(hits.scalar_type()));delta=(a-before).detach();updates.add_(delta.abs());}
    TORCH_CHECK(torch::isfinite(a).all().item<bool>(),"NONFINITE_COEFFICIENT");sync();double core_ms=elapsed(start);
    auto nonzero=gradient.ne(0);auto selected_nonzero=nonzero.index_select(0,order).sum().item<int64_t>();auto nnonzero=nonzero.sum().item<int64_t>();
    auto q=torch::quantile(abs,torch::tensor({.01,.1,.5,.9,.99},z.options())).cpu();auto selectedpct=rankpct.index_select(0,order);double sumabs=abs.sum().item<double>(),sumgrad=gradient.sum().item<double>();
    auto normalized=probability.weights.detach()/probability.weights.detach().sum();double ess=1/normalized.square().sum().item<double>();
    Json::Value r;r["kind"]="training";r["step"]=step;r["frame"]=frame;r["K"]=Json::Int64(K);r["sampled_count"]=Json::Int64(K);r["fresh_gumbel"]=fresh;r["train_loss"]=loss;r["train_psnr"]=train_psnr;r["baseline_before"]=oldbaseline;r["baseline_after"]=baseline[frame];r["advantage"]=advantage;r["log_prob"]=probability.log_prob.item<double>();r["reinforce_loss"]=surrogate.item<double>();r["nonzero_grad_N"]=Json::Int64(nnonzero);r["selected_nonzero_grad_N"]=Json::Int64(selected_nonzero);r["unselected_nonzero_grad_N"]=Json::Int64(nnonzero-selected_nonzero);r["gradient_coverage"]=double(nnonzero)/n;r["ever_nonzero_gradient"]=hits.gt(0).sum().item<int64_t>();r["grad_norm"]=gradient.norm().item<double>();r["grad_max"]=abs.max().item<double>();r["sum_gradient"]=sumgrad;r["abs_sum_gradient"]=std::abs(sumgrad);r["sum_abs_gradient"]=sumabs;r["gradient_shift_residual"]=std::abs(sumgrad)/(sumabs+1e-12);r["adam_update_norm"]=delta.norm().item<double>();r["coefficient_norm"]=a.detach().norm().item<double>();r["coefficient_max_abs"]=a.detach().abs().max().item<double>();r["score_spread"]=z.detach().max().item<double>()-z.detach().min().item<double>();r["z_spread"]=r["score_spread"];r["weight_ESS"]=ess;r["weight_max_fraction"]=normalized.max().item<double>();r["minimum_denominator"]=probability.denominator.detach().min().item<double>();r["minimum_shifted_weight"]=probability.weights.detach().min().item<double>();r["sample_Raw_percentile_mean"]=selectedpct.mean().item<double>();r["sample_Raw_percentile_max"]=selectedpct.max().item<double>();r["first_sample_source_row"]=rows[order[0]].item<int64_t>();r["last_sample_source_row"]=rows[order[K-1]].item<int64_t>();r["sampling_ms"]=sampling_ms;r["log_prob_ms"]=log_prob_ms;r["render_ms"]=render_ms;r["backward_Adam_ms"]=backward_ms;r["core_ms"]=core_ms;
    for(int j=0;j<5;++j)r["gradient_abs_quantiles"].append(q[j].item<double>());for(int j=0;j<8;++j)r["noise_signature"].append(signature[j].item<double>());
    if(step==1||step%100==0)scene.verify();
    if(step%100==0||step==250||step==steps)hardBank(step);
    if(step==250||step==1000||step==steps)snapshot(std::string(argv[3])+"/step_"+std::to_string(step),a,ema,maximum,updates,hits,torch::tensor(baseline,torch::kFloat64));
    sync();r["total_step_ms"]=elapsed(start);emit(r);
  }
  scene.verify();Json::Value end;end["kind"]="completed";end["completed"]=true;end["steps"]=steps;end["frozen_exact"]=true;end["pure_score_function"]=true;end["no_probability_clamp"]=true;emit(end);
}
