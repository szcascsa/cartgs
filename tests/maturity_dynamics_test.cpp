#include "include/gaussian_protection.h"
#include <json/json.h>
#include <fstream>
#include <iostream>
#include <stdexcept>

void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void near(float a,float b,const char* message){check(std::abs(a-b)<2e-5,message);}
int main(int argc,char**) {
  if(argc>1&&!torch::cuda::is_available())return 77;
  const auto opts=torch::TensorOptions().dtype(torch::kFloat32).device(argc>1?torch::kCUDA:torch::kCPU);
  auto m=torch::zeros({1},opts),s=torch::full({1},.95,opts);
  for(int i=1;i<=10;++i) {
    auto next=temporalMaturity(m,s,.05,.5);
    check(next.item<float>()>m.item<float>()&&(next-m).item<float>()<.05,"sustained stability jumped or failed to rise");
    m=next;
    if(i==1)near(m.item<float>(),.0475,"first evidence must rise slowly");
    if(i==2)near(m.item<float>(),.092625,"second trust update");
    if(i==10)near(m.item<float>(),.95*(1-std::pow(.95,10)),"cycle ten trust update");
  }
  auto falling=temporalMaturity(torch::full({1},.8,opts),torch::full({1},.2,opts),.05,.5);
  near(falling.item<float>(),.5,"fast withdrawal must yield 0.5");
  near(temporalMaturity(falling,s,.05,.5).item<float>(),.5225,"slow recovery used wrong rate");
  check(temporalMaturity(m,s,1,1).equal(s),"legacy disable must be bitwise stability");
  GaussianProtection quiet,loud;quiet.config.maturity_diagnostics=false;
  quiet.config.performance_logging=loud.config.performance_logging=false;
  torch::manual_seed(19);quiet.reset(1,opts);torch::manual_seed(19);loud.reset(1,opts);
  auto probe=torch::zeros({1,3},opts),probe_q=torch::tensor({{1.f,0.f,0.f,0.f}},opts);
  for(int i=0;i<12;++i) {
    probe.add_(.001);quiet.mapping_iteration=loud.mapping_iteration=5*(i+1);
    if(i%3){quiet.markTouched(torch::zeros({1},opts.dtype(torch::kInt64)));loud.markTouched(torch::zeros({1},opts.dtype(torch::kInt64)));}
    quiet.observeFull(probe,torch::zeros_like(probe),probe_q);loud.observeFull(probe,torch::zeros_like(probe),probe_q);
    check(quiet.maturity.equal(loud.maturity)&&quiet.drift_ema.equal(loud.drift_ema)&&
        quiet.cached_uniform.equal(loud.cached_uniform),"diagnostics changed maturity or admission RNG");
  }
  GaussianProtection p;p.config.performance_logging=false;p.reset(2,opts);
  auto xyz=torch::zeros({2,3},opts),scale=torch::zeros_like(xyz),q=torch::zeros({2,4},opts);
  q.select(1,0).fill_(1);auto rows=torch::arange(2,opts.dtype(torch::kInt64));
  p.mapping_iteration=5;p.markTouched(rows);p.observeFull(xyz,scale,q);
  check(p.maturity.eq(0).all().item<bool>()&&p.dynamics.state.select(1,MaturityDynamics::FirstValid).eq(-1).all().item<bool>(),"first snapshot updated trust or first-valid diagnostics");
  const float drift=-.1*std::log(.95);
  p.mapping_iteration=10;xyz.select(1,0).add_(drift);p.markTouched(rows);p.observeFull(xyz,scale,q);
  near(p.drift_ema[0].item<float>(),drift,"first drift EMA must initialize directly");
  near(p.maturity[0].item<float>(),.0475,"first actual drift jumped to stability");
  auto frozen_m=p.maturity.clone(),frozen_ema=p.drift_ema.clone();
  for(int i=0;i<3;++i){p.mapping_iteration+=5;p.observeFull(xyz,scale,q);}
  check(p.maturity.equal(frozen_m)&&p.drift_ema.equal(frozen_ema),"untouched state changed");
  for(int i=0;i<120;++i){p.mapping_iteration+=5;xyz.select(1,0).add_(drift);p.markTouched(rows);p.observeFull(xyz,scale,q);}
  check(p.maturity.min().item<float>()>.94,"long stable history did not earn trust");
  auto before=p.maturity.clone();xyz.select(1,0).add_(1);p.mapping_iteration+=5;p.markTouched(rows);p.observeFull(xyz,scale,q);
  check((before-p.maturity).min().item<float>()>.4&&p.maturity.max().item<float>()<.5,"trust was not quickly reversible");
  auto low=p.maturity.clone();
  for(int i=0;i<240;++i){p.mapping_iteration+=5;xyz.select(1,0).add_(drift);p.markTouched(rows);p.observeFull(xyz,scale,q);}
  check((p.maturity>low).all().item<bool>()&&p.maturity.min().item<float>()>.94,"trust failed to recover slowly");
  // The row's slow-clock crossings and effective observation crossings are distinct.
  check(p.dynamics.state[0][MaturityDynamics::Obs03].item<int>()==8,"0.3 effective evidence crossing");
  check(p.dynamics.state[0][MaturityDynamics::First03].item<int>()-p.dynamics.state[0][MaturityDynamics::FirstValid].item<int>()+1==11,"untouched Full cycles lost in elapsed crossing");
  const auto original=p.dynamics.state.clone();
  p.append(1,rows.slice(0,0,1));p.append(2,torch::tensor({1,1},rows.options()));p.append(1);p.assertAligned(6);
  near(p.maturity[2].item<float>(),p.maturity[0].item<float>(),"clone maturity inheritance changed");
  near(p.drift_ema[3].item<float>(),p.drift_ema[1].item<float>(),"split drift inheritance changed");
  check(p.dynamics.state[2][MaturityDynamics::Inherited].item<int>()==1&&
      p.dynamics.state[2][MaturityDynamics::FirstValid].item<int>()==-1&&
      p.maturity[5].item<float>()==0,"child diagnostic identity not fresh");
  check(p.dynamics.state[2][MaturityDynamics::First03].item<int>()==p.dynamics.full_cycle_index&&
      p.dynamics.state[2][MaturityDynamics::Obs03].item<int>()==0,"inherited birth threshold must record zero personal observations");
  auto keep=torch::tensor({false,true,true,false,true,true},opts.dtype(torch::kBool));
  const auto all=p.dynamics.state.clone();p.prune(keep);p.assertAligned(4);
  check(p.dynamics.state.equal(all.index({keep})),"diagnostic prune index alignment changed");
  auto folder=std::filesystem::temp_directory_path()/("trust-dynamics-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  p.dynamics.save(folder,p.maturity,.05,.5);
  Json::Value summary;std::ifstream input(folder/"maturity_speed.json");input>>summary;
  check(summary["all_births_including_pruned"].asInt()==6&&summary["retired_count"].asInt()==2,"retired cohort was lost");
  check(summary["drop_gt_0.1_count"].asInt()>=2&&summary["ever_0.8_then_below_0.5_count"].asInt()==2,"drop/reversal diagnostics missing");
  check(summary["fresh_births"]["effective_observations_to_0.3"]["median"].asFloat()==8,"fresh-birth maturity distribution");
  check(summary["inherited_births"]["cycles_to_0.3"]["median"].asFloat()==0&&
      summary["inherited_births"]["never_reached_0.3"].asInt()==0,"inherited at-birth crossing falsely censored");
  check(summary["recovery_cycles"]["valid_count"].asInt()>=2,"recovery episodes were not recorded");
  for(auto rates:std::vector<std::pair<float,float>>{{0,.5},{.6,.5},{.05,1.1}}) {
    ProtectionConfig config;config.maturity_rise_rate=rates.first;config.maturity_fall_rate=rates.second;
    bool rejected=false;try{config.validate();}catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"invalid trust rates accepted");
  }
  std::cout<<"PASS first observation / sustained stability / fast fall / slow recovery / untouched / reversibility / topology / cohorts\n";
}
