#include "include/maturity_dynamics.h"
#include <json/json.h>
#include <fstream>
#include <limits>

namespace {
torch::Tensor column(const torch::Tensor& state, int c) { return state.select(1,c); }
torch::Tensor blank(int64_t n,const torch::TensorOptions& options,int64_t iteration) {
  auto state=torch::full({n,MaturityDynamics::Columns},-1,options.dtype(torch::kInt32).requires_grad(false));
  for(int c:{MaturityDynamics::Observations,MaturityDynamics::HighThenLow,MaturityDynamics::RecoveryEvents,
      MaturityDynamics::Drop10,MaturityDynamics::Drop20,MaturityDynamics::Inherited}) column(state,c).zero_();
  column(state,MaturityDynamics::Birth).fill_(iteration);
  return state;
}
torch::Tensor quantiles(const torch::Tensor& value,const std::vector<float>& qs,bool nan=false) {
  // GPU rank/gather avoids quantile's GPU-q validation readback. NaNs sort
  // last; the finite count remains a device scalar throughout the reduction.
  const auto q=torch::tensor(qs,value.options());
  const auto sorted=std::get<0>(value.sort());
  const auto count=nan?torch::isfinite(value).sum():torch::full({},value.numel(),value.options().dtype(torch::kInt64));
  const auto rank=q*(count-1).clamp_min(0);
  const auto lo=rank.floor().to(torch::kInt64),hi=rank.ceil().to(torch::kInt64);
  const auto result=torch::lerp(sorted.index_select(0,lo),sorted.index_select(0,hi),rank-lo);
  return torch::where(count>0,result,torch::full_like(result,std::numeric_limits<float>::quiet_NaN()));
}
Json::Value distribution(const torch::Tensor& value) {
  Json::Value result;
  result["valid_count"]=Json::Int64(value.numel());
  if(!value.numel()) {
    for(const auto* key:{"mean","median","p10","p25","p75","p90","p95"}) result[key]=Json::nullValue;
    return result;
  }
  const auto host=torch::cat({value.to(torch::kFloat32).mean().reshape({1}),
      quantiles(value.to(torch::kFloat32),{.5,.1,.25,.75,.9,.95})}).to(torch::kCPU);
  int i=0;for(const auto* key:{"mean","median","p10","p25","p75","p90","p95"}) result[key]=host[i++].item<double>();
  return result;
}
Json::Value histogramDistribution(const torch::Tensor& histogram) {
  const auto host=histogram.to(torch::kCPU).contiguous();
  const auto* counts=host.data_ptr<int64_t>();
  int64_t n=0;double sum=0;
  for(int64_t i=0;i<host.numel();++i){n+=counts[i];sum+=i*double(counts[i]);}
  Json::Value result;result["valid_count"]=Json::Int64(n);
  if(!n){for(const auto* k:{"mean","median","p10","p25","p75","p90","p95"})result[k]=Json::nullValue;return result;}
  result["mean"]=sum/n;
  const auto at=[&](int64_t rank){int64_t cumulative=0;for(int64_t i=0;i<host.numel();++i){cumulative+=counts[i];if(cumulative>rank)return double(i);}return 0.;};
  const std::vector<std::pair<std::string,double>> qs={{"median",.5},{"p10",.1},{"p25",.25},{"p75",.75},{"p90",.9},{"p95",.95}};
  for(const auto& q:qs){double rank=q.second*(n-1);auto lo=int64_t(std::floor(rank)),hi=int64_t(std::ceil(rank));result[q.first]=at(lo)+(rank-lo)*(at(hi)-at(lo));}
  return result;
}
const std::vector<std::string>& curveKeys() {
  static const std::vector<std::string> keys={"mapping_iteration","full_cycle_index","total_gaussians",
    "snapshot_valid_count","touched_count","drift_initialized_count","valid_evidence_count",
    "drift_mean","drift_median","drift_p10","drift_p90","drift_p95",
    "stability_mean","stability_median","stability_p10","stability_p50","stability_p90",
    "maturity_mean","maturity_median","maturity_p10","maturity_p50","maturity_p90","maturity_p95",
    "fraction_m_ge_0.1","fraction_m_ge_0.3","fraction_m_ge_0.5","fraction_m_ge_0.8","fraction_m_ge_0.95",
    "drop_gt_0.1_count","drop_gt_0.2_count","m_ge_0.3_count",
    "drift_ema_mean","drift_ema_median","drift_ema_p10","drift_ema_p90","drift_ema_p95"};
  return keys;
}
}

void MaturityDynamics::reset(int64_t count,const torch::TensorOptions& options,int64_t iteration) {
  torch::NoGradGuard guard;
  full_cycle_index=0;retired_state_.clear();retired_values_.clear();curve_.clear();
  state=enabled?blank(count,options,iteration):torch::Tensor{};
  values=enabled?torch::zeros({count,3},options.dtype(torch::kFloat32).requires_grad(false)):torch::Tensor{};
  recovery_histogram_=enabled?torch::zeros({2,256},options.dtype(torch::kInt64).requires_grad(false)):torch::Tensor{};
}
void MaturityDynamics::append(int64_t count,int64_t iteration,const torch::Tensor& m,bool inherited) {
  if(!enabled)return;
  torch::NoGradGuard guard;
  auto added=blank(count,state.options(),iteration);column(added,Inherited).fill_(inherited);
  // A child can be born above a threshold through existing trust inheritance.
  // Record that real first crossing at birth, with zero personal observations.
  for(int i=0;i<4;++i) {
    const float threshold=std::vector<float>{.3,.5,.8,.95}[i];
    const auto reached=m>=threshold;
    column(added,First03+i).copy_(torch::where(reached,full_cycle_index,column(added,First03+i)));
    column(added,Obs03+i).copy_(torch::where(reached,0,column(added,Obs03+i)));
  }
  auto floats=torch::zeros({count,3},values.options());floats.select(1,0).copy_(m);
  state=torch::cat({state,added});values=torch::cat({values,floats});
}
void MaturityDynamics::prune(const torch::Tensor& survivors,const torch::Tensor& m) {
  if(!enabled)return;
  torch::NoGradGuard guard;
  values.select(1,1).copy_(m);
  retired_state_.push_back(state.index({~survivors}));
  retired_values_.push_back(values.index({~survivors}));
  state=state.index({survivors});values=values.index({survivors});
}
void MaturityDynamics::observe(int64_t iteration,const torch::Tensor& valid,const torch::Tensor& touched,
    const torch::Tensor& snapshot,const torch::Tensor& initialized,const torch::Tensor& drift,
    const torch::Tensor& drift_ema,const torch::Tensor& stability,const torch::Tensor& previous,const torch::Tensor& m) {
  ++full_cycle_index;
  if(!enabled)return;
  torch::NoGradGuard guard;
  auto first=column(state,FirstValid),observations=column(state,Observations);
  first.copy_(torch::where(valid&(first<0),full_cycle_index,first));
  observations.add_(valid.to(torch::kInt32));
  auto snap=column(state,Snapshot);snap.copy_(torch::where(snapshot&(snap<0),iteration,snap));
  const float thresholds[]={.3,.5,.8,.95};
  for(int i=0;i<4;++i) {
    auto crossing=column(state,First03+i),obs=column(state,Obs03+i);
    auto hit=valid&(m>=thresholds[i])&(crossing<0);
    crossing.copy_(torch::where(hit,full_cycle_index,crossing));
    obs.copy_(torch::where(hit,observations,obs));
  }
  const auto drop=previous-m;
  const auto significant=valid&(drop>.1);
  column(state,Drop10).add_(significant.to(torch::kInt32));
  column(state,Drop20).add_((valid&(drop>.2)).to(torch::kInt32));
  column(state,HighThenLow).copy_(torch::maximum(column(state,HighThenLow),
      (((column(state,First08)>=0)|(values.select(1,0)>=.8))&valid&(m<.5)).to(torch::kInt32)));
  // One pending recovery episode per Gaussian. Further drops do not lower
  // the original target or restart its clock; pruning right-censors it.
  auto start=column(state,RecoveryStart),start_obs=column(state,RecoveryObservation);
  auto target=values.select(1,2);
  auto begin=significant&(start<0);
  start.copy_(torch::where(begin,full_cycle_index,start));
  start_obs.copy_(torch::where(begin,observations,start_obs));
  target.copy_(torch::where(begin,previous,target));
  column(state,RecoveryEvents).add_(begin.to(torch::kInt32));
  auto done=valid&(start>=0)&(m>=target);
  if(full_cycle_index>=recovery_histogram_.size(1)) {
    auto expanded=torch::zeros({2,recovery_histogram_.size(1)*2},recovery_histogram_.options());
    expanded.slice(1,0,recovery_histogram_.size(1)).copy_(recovery_histogram_);recovery_histogram_=expanded;
  }
  const auto durations=torch::stack({torch::full_like(start,full_cycle_index)-start,observations-start_obs});
  for(int i=0;i<2;++i)recovery_histogram_[i].scatter_add_(0,
      durations[i].clamp(0,recovery_histogram_.size(1)-1).to(torch::kInt64),done.to(torch::kInt64));
  start.copy_(torch::where(done,-1,start));start_obs.copy_(torch::where(done,-1,start_obs));

  const auto zero=torch::zeros({},m.options());
  const auto nan=torch::full_like(drift,std::numeric_limits<float>::quiet_NaN());
  const auto evidence_drift=torch::where(valid,drift,nan);
  const auto evidence_ema=torch::where(initialized,drift_ema,nan);
  const auto evidence_stability=torch::where(initialized,stability,nan);
  const auto dq=quantiles(evidence_drift,{.5,.1,.9,.95},true);
  const auto sq=quantiles(evidence_stability,{.5,.1,.5,.9},true);
  const auto mq=quantiles(m,{.5,.1,.5,.9,.95});
  std::vector<torch::Tensor> row={zero+iteration,zero+full_cycle_index,zero+m.numel(),snapshot.sum().to(m.dtype()),
    touched.sum().to(m.dtype()),initialized.sum().to(m.dtype()),valid.sum().to(m.dtype()),
    torch::nanmean(evidence_drift)};
  for(int i=0;i<4;++i)row.push_back(dq[i]);
  row.push_back(torch::nanmean(evidence_stability));for(int i=0;i<4;++i)row.push_back(sq[i]);
  row.push_back(m.mean());for(int i=0;i<5;++i)row.push_back(mq[i]);
  for(float threshold:{.1f,.3f,.5f,.8f,.95f})row.push_back((m>=threshold).to(m.dtype()).mean());
  row.push_back(significant.sum().to(m.dtype()));row.push_back((valid&(drop>.2)).sum().to(m.dtype()));
  row.push_back((m>=.3).sum().to(m.dtype()));
  row.push_back(torch::nanmean(evidence_ema));
  const auto eq=quantiles(evidence_ema,{.5,.1,.9,.95},true);
  for(int i=0;i<4;++i)row.push_back(eq[i]);
  curve_.push_back(torch::stack(row));
}

void MaturityDynamics::save(const std::filesystem::path& folder,const torch::Tensor& maturity,float rise,float fall) const {
  if(!enabled)return;
  torch::NoGradGuard guard;
  std::filesystem::create_directories(folder);
  Json::Value curve(Json::arrayValue);
  std::ofstream csv(folder/"maturity_dynamics.csv");
  const auto& keys=curveKeys();for(size_t i=0;i<keys.size();++i)csv<<(i?",":"")<<keys[i];csv<<'\n';
  if(!curve_.empty()) {
    const auto host=torch::stack(curve_).to(torch::kCPU).to(torch::kDouble).contiguous();
    auto data=host.accessor<double,2>();
    csv.precision(10);
    for(int64_t i=0;i<host.size(0);++i) {
      Json::Value row;
      for(size_t j=0;j<keys.size();++j) {
        if(std::isfinite(data[i][j])) {row[keys[j]]=data[i][j];csv<<(j?",":"")<<data[i][j];}
        else {row[keys[j]]=Json::nullValue;csv<<(j?",":"");}
      }
      csv<<'\n';curve.append(row);
    }
  }
  std::ofstream(folder/"maturity_dynamics.json")<<curve;
  auto active_values=values.clone();active_values.select(1,1).copy_(maturity);
  auto all_states=retired_state_;all_states.push_back(state);
  auto all_values=retired_values_;all_values.push_back(active_values);
  const auto all=torch::cat(all_states),floats=torch::cat(all_values);
  Json::Value summary;
  summary["rise_rate"]=rise;summary["fall_rate"]=fall;
  summary["full_cycle_count"]=Json::Int64(full_cycle_index);
  summary["all_births_including_pruned"]=Json::Int64(all.size(0));
  summary["surviving_count"]=Json::Int64(state.size(0));
  summary["retired_count"]=Json::Int64(all.size(0)-state.size(0));
  summary["time_definition"]="cycles_to = first_crossing_full_cycle - first_valid_full_cycle + 1; inherited birth already above threshold is 0 cycles/0 personal observations";
  summary["censoring"]="never_reached includes pruned and end-of-run censored rows; retired maturity is last value at pruning, not final-map maturity";
  const auto speed=[&](const torch::Tensor& s) {
    Json::Value result;result["gaussian_count"]=Json::Int64(s.size(0));
    result["without_valid_evidence_count"]=(column(s,FirstValid)<0).sum().item<int64_t>();
    for(int i=0;i<4;++i) {
      const auto reached=column(s,First03+i)>=0;
      const std::string threshold=std::vector<std::string>{"0.3","0.5","0.8","0.95"}[i];
      const auto elapsed=torch::where(column(s,Obs03+i)==0,0,column(s,First03+i)-column(s,FirstValid)+1);
      result["cycles_to_"+threshold]=distribution(elapsed.index({reached}));
      result["effective_observations_to_"+threshold]=distribution(column(s,Obs03+i).index({reached}));
      result["never_reached_"+threshold]=(~reached).sum().item<int64_t>();
    }
    return result;
  };
  summary["all_births"]=speed(all);summary["survivors"]=speed(state);
  summary["fresh_births"]=speed(all.index({column(all,Inherited)==0}));
  summary["inherited_births"]=speed(all.index({column(all,Inherited)!=0}));
  summary["inherited_initial_maturity"]=distribution(floats.select(1,0).index({column(all,Inherited)!=0}));
  summary["drop_gt_0.1_count"]=column(all,Drop10).sum().item<int64_t>();
  summary["drop_gt_0.2_count"]=column(all,Drop20).sum().item<int64_t>();
  summary["gaussians_with_drop_gt_0.1"]=(column(all,Drop10)>0).sum().item<int64_t>();
  summary["gaussians_with_drop_gt_0.2"]=(column(all,Drop20)>0).sum().item<int64_t>();
  summary["ever_0.8_then_below_0.5_count"]=column(all,HighThenLow).sum().item<int64_t>();
  summary["recovery_episodes"]=column(all,RecoveryEvents).sum().item<int64_t>();
  summary["unrecovered_or_pruned_episodes"]=(column(all,RecoveryStart)>=0).sum().item<int64_t>();
  summary["recovery_cycles"]=histogramDistribution(recovery_histogram_[0]);
  summary["recovery_effective_observations"]=histogramDistribution(recovery_histogram_[1]);
  const int bounds[]={0,500,1000,1500,2000,3000,std::numeric_limits<int>::max()};
  const auto birth=column(all,Birth);
  for(int i=0;i<6;++i) {
    auto mask=(birth>=bounds[i])&(birth<bounds[i+1]);
    auto cohort=speed(all.index({mask}));cohort["mapping_begin"]=bounds[i];
    cohort["mapping_end_exclusive"]=i==5?Json::Value(Json::nullValue):Json::Value(bounds[i+1]);
    cohort["terminal_maturity"]=distribution(floats.select(1,1).index({mask}));
    const auto live_birth=column(state,Birth),live=(live_birth>=bounds[i])&(live_birth<bounds[i+1]);
    cohort["surviving_count"]=live.sum().item<int64_t>();
    cohort["final_survivor_maturity"]=distribution(maturity.index({live}));
    summary["birth_cohorts"].append(cohort);
  }
  summary["diagnostics_device_memory_mb"]=memoryMB();
  std::ofstream(folder/"maturity_speed.json")<<summary;
}
double MaturityDynamics::memoryMB() const {
  int64_t bytes=0;
  for(const auto& t:{state,values,recovery_histogram_})if(t.defined())bytes+=t.numel()*t.element_size();
  for(const auto* list:{&retired_state_,&retired_values_,&curve_})
    for(const auto& t:*list)bytes+=t.numel()*t.element_size();
  return bytes/1048576.;
}
