#pragma once
#include <json/json.h>
#include <torch/torch.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

// Observational only: abort before Adam; never clamp outputs or resample points.
struct LodHealthGuard {
  bool enabled=true;
  int zero_patience=8, saturation_patience=8;
  double probability_floor=1e-6, scale_limit=100., xyz_limit=256., tail_fraction=.01;
  std::map<size_t,int> zero_streak, saturation_streak;
  int global_zero_streak=0;
  Json::Value sample;

  std::string selection(size_t ratio_index,int64_t step,double ratio,int64_t population,
      int64_t selected,const torch::Tensor& probability,const torch::Tensor& eligible) {
    torch::NoGradGuard guard;
    sample=Json::Value();sample["step"]=Json::Int64(step);sample["ratio"]=ratio;
    sample["population"]=Json::Int64(population);sample["selected"]=Json::Int64(selected);
    auto p=probability.detach().masked_select(eligible);
    const auto count=p.numel();sample["eligible_candidates"]=Json::Int64(count);
    if(!count) { zero_streak[ratio_index]=0;saturation_streak[ratio_index]=0;global_zero_streak=0;return {}; }
    auto packed=torch::stack({p.sum(),p.max(),p.min(),(p*(1-p)).sum(),torch::isfinite(p).all().to(p.scalar_type())}).cpu();
    const auto a=packed.accessor<float,1>();const double target=std::min(ratio*population,double(count));
    sample["feasible_target"]=target;sample["expected_keep"]=a[0];sample["probability_max"]=a[1];
    sample["probability_min"]=a[2];sample["softmax_derivative_mass"]=a[3];
    zero_streak[ratio_index]=selected?0:zero_streak[ratio_index]+1;
    global_zero_streak=selected?0:global_zero_streak+1;
    const bool cold=a[1]<probability_floor;
    const bool hot=a[2]>1-probability_floor && target<.9*count;
    saturation_streak[ratio_index]=(cold||hot)?saturation_streak[ratio_index]+1:0;
    sample["zero_streak_ratio"]=zero_streak[ratio_index];sample["zero_streak_global"]=global_zero_streak;
    sample["saturation_streak_ratio"]=saturation_streak[ratio_index];
    if(!a[4])return "nonfinite_selector_probability";
    if(global_zero_streak>=zero_patience||zero_streak[ratio_index]>=zero_patience)return "consecutive_zero_selection_with_candidates";
    if(saturation_streak[ratio_index]>=saturation_patience)return cold?"selector_all_drop_saturation":"selector_all_keep_saturation";
    return {};
  }

  std::string compensation(const torch::Tensor& xyz,const torch::Tensor& scale,
      const torch::Tensor& opacity,const torch::Tensor& master_xyz,const torch::Tensor& master_scale,
      const torch::Tensor& reference) {
    torch::NoGradGuard guard;
    if(!xyz.size(0))return {};
    auto factor=std::get<0>((scale.detach()/master_scale.detach()).max(1));
    auto movement=(xyz.detach()-master_xyz.detach()).norm(2,1)/reference.detach().reshape({-1});
    auto packed=torch::stack({(factor>scale_limit).to(torch::kFloat32).sum(),factor.max(),
        (movement>xyz_limit).to(torch::kFloat32).sum(),movement.max(),
        (opacity.detach()>1-1e-6).to(torch::kFloat32).mean(),
        (torch::isfinite(scale).all()&torch::isfinite(xyz).all()&torch::isfinite(opacity).all()&
         torch::isfinite(factor).all()&torch::isfinite(movement).all()).to(torch::kFloat32)}).cpu();
    const auto a=packed.accessor<float,1>();sample["scale_tail_fraction"]=double(a[0])/xyz.size(0);sample["scale_factor_max"]=a[1];
    sample["xyz_tail_fraction"]=double(a[2])/xyz.size(0);sample["xyz_anchor_units_max"]=a[3];sample["opaque_fraction"]=a[4];
    if(!a[5])return "nonfinite_compensation";
    const auto limit=std::max(1.,std::ceil(tail_fraction*xyz.size(0)-1e-7));
    if(a[0]>=limit)return "compensation_scale_tail";
    if(a[2]>=limit)return "compensation_xyz_tail";
    return {};
  }

  void record(const std::filesystem::path& directory,const std::string& failure={}) {
    sample["before_optimizer_commit"]=true;
    if(!failure.empty()) {
      sample["error"]=failure;sample["zero_patience"]=zero_patience;sample["saturation_patience"]=saturation_patience;
      sample["probability_floor"]=probability_floor;sample["scale_limit"]=scale_limit;
      sample["xyz_limit"]=xyz_limit;sample["tail_fraction_limit"]=tail_fraction;
      std::ofstream out(directory/"lod_health_failure.json");out<<sample;out.flush();
    }
    Json::StreamWriterBuilder writer;writer["indentation"]="";
    std::ofstream stream(directory/"lod_health.jsonl",std::ios::app);stream<<Json::writeString(writer,sample)<<'\n';
  }
};
