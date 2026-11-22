#include "pl_selection.h"
#include <json/json.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
using T=torch::Tensor;
void save(std::string p,Json::Value v){Json::StreamWriterBuilder w;w["indentation"]="  ";std::ofstream(p)<<Json::writeString(w,v)<<"\n";}
int main(int argc,char** argv){
  TORCH_CHECK(argc==2,"pl_tests output_directory");torch::set_num_threads(2);torch::manual_seed(41020);
  auto opts=torch::TensorOptions().dtype(torch::kFloat64);auto z=torch::tensor({-.7,.2,.9,-.3,.55,1.1},opts).set_requires_grad(true);
  std::vector<T> ps,gradients;std::vector<double> losses,manual;std::vector<std::pair<int,int>> pairs;
  auto weights=z.detach().exp();double total=weights.sum().item<double>();
  for(int i=0;i<6;++i)for(int j=0;j<6;++j)if(i!=j){
    auto order=torch::tensor({i,j},torch::kInt64);auto l=plLogProbability(z,order).log_prob;
    double p=weights[i].item<double>()/total*weights[j].item<double>()/(total-weights[i].item<double>());
    TORCH_CHECK(std::abs(l.exp().item<double>()-p)<1e-12,"MANUAL_PROBABILITY_MISMATCH");
    ps.push_back(l.exp());manual.push_back(p);gradients.push_back(torch::autograd::grad({l},{z},{},true)[0]);pairs.emplace_back(i,j);
    int a=std::min(i,j),b=std::max(i,j);losses.push_back(.2+std::sin(1.31*(a+1)+.77*(b+1))+.11*a*b);
  }
  auto probabilities=torch::stack(ps);auto costs=torch::tensor(losses,opts);auto objective=(probabilities*costs).sum();auto exact=torch::autograd::grad({objective},{z})[0].detach();
  TORCH_CHECK(std::abs(probabilities.sum().item<double>()-1)<1e-12,"PROBABILITY_NOT_NORMALIZED");
  int64_t M=200000;auto counts=torch::zeros({6,6},torch::kInt64);
  for(int batch=0;batch<20;++batch){torch::NoGradGuard guard;auto u=torch::rand({M/20,6},opts);TORCH_CHECK((u>0).all().item<bool>()&&(u<1).all().item<bool>(),"BAD_UNIFORM");auto g=-(-u.log()).log();auto o=std::get<1>((z.detach().unsqueeze(0)+g).topk(2,1,true,true));auto index=(o.select(1,0)*6+o.select(1,1)).contiguous();counts+=torch::bincount(index,{},36).reshape({6,6});}
  double maxfrequencyZ=0,maxfrequencyError=0;Json::Value prob;
  std::vector<double> empirical;
  for(size_t n=0;n<pairs.size();++n){auto [i,j]=pairs[n];double p=manual[n],f=double(counts[i][j].item<int64_t>())/M,se=std::sqrt(p*(1-p)/M);TORCH_CHECK(std::abs(f-p)<=6*se+1e-12,"GUMBEL_EMPIRICAL_MISMATCH");empirical.push_back(f);maxfrequencyZ=std::max(maxfrequencyZ,std::abs(f-p)/se);maxfrequencyError=std::max(maxfrequencyError,std::abs(f-p));Json::Value r;r["first"]=i;r["second"]=j;r["analytical"]=p;r["empirical"]=f;prob["ordered_pairs"].append(r);}
  double baseline=objective.item<double>();auto grads=torch::stack(gradients);auto h=(costs-baseline).unsqueeze(1)*grads;
  auto mc=(torch::tensor(empirical,opts).unsqueeze(1)*h).sum(0);auto analytic=(probabilities.detach().unsqueeze(1)*h).sum(0);
  auto second=(probabilities.detach().unsqueeze(1)*h.square()).sum(0);auto se=((second-analytic.square())/M).sqrt();
  TORCH_CHECK((analytic-exact).abs().max().item<double>()<1e-12,"EXACT_SCORE_FUNCTION_IDENTITY_FAILED");
  TORCH_CHECK(((mc-exact).abs()<=6*se+1e-12).all().item<bool>(),"MONTE_CARLO_GRADIENT_MISMATCH");
  TORCH_CHECK((mc*exact).sum().item<double>()>0,"GRADIENT_SIGN_REVERSED");
  auto expectedManual=[&](T logits){auto w=logits.exp();double sum=w.sum().item<double>(),J=0;for(size_t k=0;k<pairs.size();++k){auto[i,j]=pairs[k];J+=w[i].item<double>()/sum*w[j].item<double>()/(sum-w[i].item<double>())*losses[k];}return J;};
  double fd_error=0;for(int i=0;i<6;++i){auto plus=z.detach().clone(),minus=plus.clone();plus[i]+=.000001;minus[i]-=.000001;double fd=(expectedManual(plus)-expectedManual(minus))/.000002;fd_error=std::max(fd_error,std::abs(fd-exact[i].item<double>()));}TORCH_CHECK(fd_error<1e-8,"FINITE_DIFFERENCE_MISMATCH");
  double after=expectedManual(z.detach()-.01*exact);TORCH_CHECK(after<baseline,"DESCENT_SIGN_INVALID");
  auto z8=torch::tensor({-.8,.3,1.2,.15,-.2,.7,.9,-1.1},opts).set_requires_grad(true);auto[o8,g8]=sampleFreshGumbelTopK(z8,3);auto log8=plLogProbability(z8,o8).log_prob;auto gradient=torch::autograd::grad({2.3*log8},{z8})[0];auto selected=torch::zeros({8},torch::kBool);selected.index_fill_(0,o8,true);
  TORCH_CHECK(torch::isfinite(gradient).all().item<bool>()&&(gradient!=0).all().item<bool>(),"SELECTED_OR_UNSELECTED_ZERO_GRADIENT");
  auto invariant=plLogProbability(z8.detach()+17,o8).log_prob;TORCH_CHECK(std::abs(invariant.item<double>()-log8.item<double>())<1e-12,"SHIFT_INVARIANCE_FAILED");
  prob["all_passed"]=true;prob["sum_probability"]=probabilities.sum().item<double>();prob["samples"]=Json::Int64(M);prob["max_frequency_abs_error"]=maxfrequencyError;prob["max_frequency_error_SE"]=maxfrequencyZ;prob["no_probability_clamp"]=true;
  Json::Value test;test["all_passed"]=true;test["baseline"]=baseline;test["after_exact_gradient_descent"]=after;test["finite_difference_max_abs_error"]=fd_error;test["selected_gradient_nonzero"]=true;test["unselected_gradient_nonzero"]=true;test["gradient_sum_abs"]=std::abs(gradient.sum().item<double>());test["MC_gradient_cosine"]=(mc*exact).sum().item<double>()/mc.norm().item<double>()/exact.norm().item<double>();
  for(int i=0;i<6;++i){Json::Value r;r["exact"]=exact[i].item<double>();r["MC"]=mc[i].item<double>();r["MC_SE"]=se[i].item<double>();test["gradient_components"].append(r);}
  for(int i=0;i<8;++i){Json::Value r;r["i"]=i;r["selected"]=selected[i].item<bool>();r["gradient"]=gradient[i].item<double>();test["N8_K3"].append(r);}
  save(std::string(argv[1])+"/pl_probability_tests.json",prob);save(std::string(argv[1])+"/reinforce_gradient_tests.json",test);std::cout<<"PL_TOY_GATES_PASS"<<std::endl;
}
