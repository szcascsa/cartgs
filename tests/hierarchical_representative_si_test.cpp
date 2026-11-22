#include "include/hierarchical_representative_si.h"
#include "include/global_gi.h"
#include "include/gi_teacher.h"
#include "include/selector_handoff.h"
#include "improvements/selector/gumbel_network.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace h = hierarchical_si;
namespace {
torch::Device device(torch::kCPU);
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
torch::TensorOptions options() { return torch::TensorOptions().dtype(torch::kFloat64).device(device); }
h::CellSelection cell(torch::Tensor x, torch::Tensor radius, torch::Tensor weights) {
  return h::selectCell(x.to(options()), radius.to(options()), weights.to(options()),
      torch::arange(weights.numel(), options().dtype(torch::kLong)));
}
h::Result anchored(const torch::Tensor& centers, const torch::Tensor& weights, double radius = 1e-4) {
  auto x = torch::cat({centers.to(options()), torch::tensor({{0.,0.,0.},{1600.,1600.,1600.}}, options())});
  auto w = torch::cat({weights.to(options()), torch::zeros({2}, options())});
  auto valid = torch::ones({w.numel()}, options().dtype(torch::kBool));
  auto observed = valid.clone(); observed.slice(0, weights.numel()).fill_(false);
  return h::build(x, torch::full_like(x, radius), w, valid, observed, h::RootBackend::LegacyAabb2048);
}
int leaves(const h::Result& result, int level) {
  int count = 0; for (const auto& node : result.nodes) count += node.leaf && node.level == level;
  return count;
}
void conserved(const h::Result& r) {
  check(r.conservation_relative <= 1e-6, "Global SI mass conservation failed");
  for (const auto& n : r.nodes) if (n.leaf) check(n.conservation_relative <= 1e-6, "Leaf mass conservation failed");
}
void identical(const h::Result& a, const h::Result& b) {
  check(a.dimensions == b.dimensions && a.lower == b.lower && a.upper == b.upper && a.nodes.size() == b.nodes.size(), "Tree nondeterministic");
  check(a.importance.equal(b.importance) && a.leaf_assignment.equal(b.leaf_assignment), "SI or assignment nondeterministic");
  for (size_t i = 0; i < a.nodes.size(); ++i) {
    const auto& x = a.nodes[i]; const auto& y = b.nodes[i];
    check(x.level == y.level && x.root == y.root && x.path == y.path && x.leaf == y.leaf &&
        x.quality == y.quality && x.representatives == y.representatives && x.gains == y.gains,
        "Representative sequence or Q nondeterministic");
  }
}
void runTests() {
  auto dup = cell(torch::zeros({3,3}), torch::ones({3}), torch::tensor({10.,9.,8.}));
  check(dup.representatives[0] == 0 && dup.gains[0] == 27 && dup.gains[1] == 0 && dup.gains[2] == 0 && dup.quality > .999, "Duplicate 3-rep cluster failed");
  auto independent = cell(torch::tensor({{0.,0.,0.},{1.,0.,0.},{2.,0.,0.}}), torch::full({3},1e-4), torch::tensor({10.,9.,8.}));
  check(independent.gains == std::vector<double>({10,9,8}) && independent.quality > .999, "Independent cluster failed");
  auto four = cell(torch::tensor({{0.,0.,0.},{1.,0.,0.},{2.,0.,0.},{3.,0.,0.}}), torch::full({4},1e-4), torch::ones({4}));
  check(std::abs(four.quality - .75) < 1e-10 && four.quality < .8, "4 independent threshold must be .75");
  auto center=cell(torch::tensor({{0.,0.,0.},{.1,0.,0.},{.2,0.,0.}}),torch::full({3},.1),torch::tensor({10.,1.,10.}));
  check(center.representatives[0]==1,"First representative must maximize weighted coverage, not Raw SI");
  // Independent full-matrix brute force checks all 128-row block boundaries.
  auto bx=torch::rand({257,3},options()),br=torch::rand({257},options())*.2+.01,bw=torch::rand({257},options());
  auto sequence=cell(bx,br,bw);auto cover=torch::zeros_like(bw);
  auto distances=(bx.unsqueeze(0)-bx.unsqueeze(1)).square().sum(2);
  auto kernel=torch::exp(-distances/(.5*(br.unsqueeze(0)+br.unsqueeze(1)).square()+h::kEpsilon));
  auto selected=torch::zeros({257},options().dtype(torch::kBool));
  for(size_t k=0;k<sequence.representatives.size();++k){
    auto gains=((kernel-cover).clamp_min(0)*bw).sum(1);gains.masked_fill_(selected,-INFINITY);
    auto best=gains.argmax().item<int64_t>();
    check(best==sequence.representatives[k] && std::abs(gains[best].item<double>()-sequence.gains[k])<1e-10,"Blocked production != dense reference greedy");
    selected[best]=true;cover=torch::maximum(cover,kernel[best]);
  }
  auto large = cell(torch::zeros({100,3}), torch::ones({100}), torch::ones({100}));
  check(large.gains[0] == 100 && large.quality > .8, "High occupancy duplicates should not split");
  auto two_x = torch::cat({torch::zeros({20,3}), torch::ones({20,3}) * 10});
  auto two = cell(two_x, torch::full({40},.01), torch::ones({40}));
  check(two.gains[0] == 20 && two.gains[1] == 20 && two.quality > .8, "Two duplicate clusters failed");
  auto four_x = torch::tensor({{25.,25.,25.},{175.,25.,25.},{25.,75.,25.},{175.,75.,25.}}, options());
  auto clusters = anchored(four_x.repeat_interleave(5,0), torch::ones({20}));
  check(clusters.nodes.front().quality < .8 && !clusters.nodes.front().leaf && leaves(clusters,2) == 4, "Four clusters must split into L2");
  auto duplicate_tree = anchored(torch::ones({100,3}) * 20, torch::ones({100}), 1.);
  check(duplicate_tree.nodes.size() == 1 && leaves(duplicate_tree,1) == 1, "Duplicate tree split by occupancy");
  auto one = anchored(four_x, torch::ones({4}));
  auto two_levels_x = torch::tensor({{10.,10.,10.},{90.,10.,10.},{10.,40.,10.},{90.,40.,10.}}, options());
  auto two_levels = anchored(two_levels_x, torch::ones({4}));
  check(leaves(two_levels,3) == 4, "Two splits did not reach L3");
  auto unresolved_x = torch::tensor({{5.,5.,5.},{6.,5.,5.},{5.,6.,5.},{6.,6.,5.}}, options());
  auto unresolved = anchored(unresolved_x, torch::ones({4}));
  check(unresolved.nodes.size() == 3 && unresolved.nodes.back().level == 3 &&
      unresolved.nodes.back().leaf && unresolved.nodes.back().quality < .8, "Max depth must stop at unresolved L3");
  for (const auto* r : {&duplicate_tree,&one,&two_levels,&unresolved,&clusters}) conserved(*r);
  for (const auto& fixture : std::vector<torch::Tensor>{torch::rand({80,3},options()) * 80, torch::rand({80,3},options()) * 5}) {
    auto weights = torch::rand({80},options()) + .01;
    auto a = anchored(fixture, weights, .1), b = anchored(fixture, weights, .1);
    conserved(a); identical(a,b);
  }
  auto reconstructed = torch::zeros_like(clusters.importance);
  // Identical within-cluster rows have zero residual. Only final leaf deltas may appear.
  for (const auto& n : clusters.nodes) if (n.leaf)
    for (size_t k=0;k<n.representatives.size();++k) reconstructed[n.representatives[k]]=n.gains[k];
  check(reconstructed.equal(clusters.importance), "Parent provisional score leaked into final SI");
  auto empty = h::build(torch::empty({0,3},options()),torch::empty({0,3},options()),
      torch::empty({0},options()),torch::empty({0},options().dtype(torch::kBool)),torch::empty({0},options().dtype(torch::kBool)));
  check(empty.nodes.empty() && empty.candidates==0, "Empty map failed");
  auto zero = anchored(torch::zeros({5,3}),torch::zeros({5}));
  check(zero.nodes.size()==3 && zero.nodes.back().level==3 && zero.nodes.back().quality==0 && zero.importance.sum().item<double>()==0, "Zero mass must follow Q=0 and max-depth rule");
  auto fx=torch::rand({32,3},options()).to(torch::kFloat32), fw=torch::rand({32},fx.options());
  auto fm=torch::ones({32},fx.options().dtype(torch::kBool));
  conserved(h::build(fx,torch::full_like(fx,.05),fw,fm,fm));
  auto rows = torch::tensor({9L,3L,7L},options().dtype(torch::kLong));
  auto tie = h::selectCell(torch::zeros({3,3},options()),torch::ones({3},options()),torch::ones({3},options()),rows);
  check(tie.representatives==std::vector<int64_t>({1,2,0}), "Stable source row tie break failed");
  // Stateless lifecycle: append/prune/reorder and geometry or activated-scale edits rebuild.
  auto x=torch::rand({24,3},options()), s=torch::full_like(x,.03), w=torch::rand({24},options());
  auto valid=torch::ones({24},options().dtype(torch::kBool));
  auto before=h::build(x,s,w,valid,valid);
  x=torch::cat({x,torch::ones({1,3},options())});s=torch::cat({s,torch::full({1,3},.03,options())});
  w=torch::cat({w,torch::ones({1},options())});valid=torch::cat({valid,torch::zeros({1},valid.options())});
  auto appended=h::build(x,s,w,valid,valid);
  check(appended.population==25 && appended.candidates==24 && appended.leaf_assignment[24].item<int64_t>()==-1, "Append resurrected unknown GI row");
  auto keep=torch::arange(25,options().dtype(torch::kLong))!=2;
  x=x.index({keep}).flip({0});s=s.index({keep}).flip({0});w=w.index({keep}).flip({0});valid=valid.index({keep}).flip({0});
  x[2][0]=10;s[3][1]=.2;
  auto edited=h::build(x,s,w,valid,valid);auto fresh=h::build(x.clone(),s.clone(),w.clone(),valid.clone(),valid.clone());
  identical(edited,fresh);conserved(edited);
  check(edited.candidates==23 && edited.population==24,"Topology rebuild reused old rows");
  std::cout << "SYNTHETIC_PASS duplicate3 independent3 independent4 duplicate100 two_clusters four_clusters mass_no_one_two_split determinism max_depth parent_discard zero_empty tie topology_geometry_scale\n";
}
void legacyRegression() {
  GlobalGIConfig config;config.mode="streaming";config.view_ratio=1;config.performance_logging=false;
  check(!config.hierarchical_representative_enabled && !config.hierarchical_representative_diagnostics,"Experimental feature default must be off");
  GlobalGIManager gi(device.is_cuda()?torch::kCUDA:torch::kCPU,config);gi.resetForGaussianCount(8);
  auto a=torch::tensor({1.f,2.f,3.f,4.f,0.f,6.f,7.f,8.f}).to(device),b=2*a+.5f;
  gi.updateBank({{1,{1}},{2,{2}}},[&](uint64_t id){return id==1?a:b;});
  check(gi.importance().equal((a+b)/2),"Raw Bank mean changed");
  auto raw=gi.importance().clone(),valid=gi.validMask().clone(),observed=gi.observedMask();
  auto score=h::score(false,{}, {},gi.importance(),{},{});
  check(score.is_same(gi.importance()) && score.equal(raw) && valid.equal(gi.validMask()),"Disabled path must return original Raw handle");
  auto maturity=torch::linspace(0,1,8,torch::TensorOptions().device(device));
  torch::manual_seed(20261019);improvements::selector::GumbelNetwork selector;selector->to(device);
  auto x=torch::rand({8,3},a.options()),rotation=torch::rand({8,4},a.options()),scale=torch::rand({8,3},a.options());
  auto probability=selector->forwardProbability(x,rotation,scale,.05f);auto u=torch::rand_like(probability);
  for(float ratio:{.01f,.05f,.1f,.15f})for(float entry:{0.f,.3f}){
    auto old=GITeacherBuilder().build(raw,valid,observed,maturity,ratio,gi.version(),gi.topologyVersion(),entry);
    auto now=GITeacherBuilder().build(score,gi.validMask(),gi.observedMask(),maturity,ratio,gi.version(),gi.topologyVersion(),entry);
    if(ratio>.1f && entry==0)check(now.label.sum().item<float>()>0 && now.label.sum().item<float>()<8,"Legacy regression must exercise both label classes");
    check(old.label.equal(now.label)&&old.weight.equal(now.weight)&&old.supervised_mask.equal(now.supervised_mask)&&old.target_k==now.target_k,"Disabled teacher parity failed");
    auto p=probability.detach().clone().requires_grad_(true),q=probability.detach().clone().requires_grad_(true);
    auto old_loss=old.loss(p,false)+selectorRatioLoss(p,ratio);auto new_loss=now.loss(q,false)+selectorRatioLoss(q,ratio);
    old_loss.backward();new_loss.backward();check(old_loss.equal(new_loss)&&p.grad().equal(q.grad()),"Disabled loss/gradient parity failed");
    check(sampleSelectorST(p,u).equal(sampleSelectorST(q,u)),"Disabled selector ST parity failed");
  }
  check(probability.equal(selector->forwardProbability(x,rotation,scale,.05f)),"Selector forward changed");
  std::cout << "LEGACY_PARITY_PASS raw valid observed teacher_labels weights loss_gradients selector_forward ST ratio_loss\n";
}
void smoke(const std::string& input,const std::filesystem::path& out){
  std::ifstream stream(input,std::ios::binary);int64_t n;stream.read(reinterpret_cast<char*>(&n),8);check(n>=0&&n<10000000,"Smoke size invalid");
  std::vector<double>x(n*3),s(n*3),w(n);std::vector<uint8_t>valid(n),observed(n);
  for(int64_t i=0;i<n;++i){stream.read(reinterpret_cast<char*>(x.data()+i*3),24);stream.read(reinterpret_cast<char*>(s.data()+i*3),24);stream.read(reinterpret_cast<char*>(w.data()+i),8);stream.read(reinterpret_cast<char*>(valid.data()+i),1);stream.read(reinterpret_cast<char*>(observed.data()+i),1);}
  check(bool(stream),"Smoke input incomplete");auto cpu=torch::TensorOptions().dtype(torch::kFloat64);
  auto X=torch::from_blob(x.data(),{n,3},cpu).clone().to(device),S=torch::from_blob(s.data(),{n,3},cpu).clone().to(device);
  auto W=torch::from_blob(w.data(),{n},cpu).clone().to(device);
  auto V=torch::from_blob(valid.data(),{n},cpu.dtype(torch::kUInt8)).to(device,torch::kBool);
  auto O=torch::from_blob(observed.data(),{n},cpu.dtype(torch::kUInt8)).to(device,torch::kBool);
  auto result=h::build(X,S,W,V,O,h::RootBackend::LegacyAabb2048);conserved(result);h::writeDiagnostics(result,out,"fr1_room");
  auto importance=result.importance.to(torch::kCPU,torch::kFloat64).contiguous();auto leaf=result.leaf_assignment.contiguous();
  std::ofstream output(out/"fr1_room_output.bin",std::ios::binary);output.write(reinterpret_cast<char*>(&n),8);
  output.write(reinterpret_cast<char*>(importance.data_ptr<double>()),n*8);output.write(reinterpret_cast<char*>(leaf.data_ptr<int64_t>()),n*8);
  check(bool(output),"Smoke output incomplete");std::cout<<"FROZEN_SMOKE_PASS N="<<n<<" candidates="<<result.candidates<<" mass_relative="<<result.conservation_relative<<" ms="<<result.build_ms<<'\n';
}
}  // namespace
int main(int argc,char**argv){try{
  torch::set_num_threads(2);torch::manual_seed(20261019);
  if(argc>1&&std::string(argv[1])=="cuda")device=torch::Device(torch::kCUDA);
  if(argc>2){check(argc==4,"smoke requires device/input/output");smoke(argv[2],argv[3]);}
  else {runTests();legacyRegression();}
  std::cout<<"IMPLEMENTATION_TEST_PASS device="<<device<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
