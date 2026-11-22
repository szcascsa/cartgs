#include "include/hierarchical_representative_si.h"
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <set>

namespace h = hierarchical_si;
namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void sameGrid(const h::SparseRootGrid& a, const h::SparseRootGrid& b) {
  check(a.h == b.h && a.roots == b.roots && a.samples.size() == b.samples.size(), "Sparse search/membership nondeterministic");
  for (size_t i=0;i<a.samples.size();++i) check(a.samples[i].h==b.samples[i].h && a.samples[i].occupied==b.samples[i].occupied,"Search trace nondeterministic");
}
void partition(const h::Result& r) {
  int64_t leaf_count=0;
  for(const auto& n:r.nodes){
    check(n.level>=1&&n.level<=3,"Sparse depth exceeds L3");
    if(n.leaf){leaf_count+=n.occupancy;continue;}
    int64_t children=0;
    for(const auto& c:r.nodes) if(c.root_key==n.root_key && c.path.size()==n.path.size()+1 &&
        c.path.compare(0,n.path.size(),n.path)==0) children+=c.occupancy;
    check(children==n.occupancy,"Child occupancy lost/duplicated rows");
  }
  check(leaf_count==r.candidates,"Final leaves do not partition candidates");
  check(r.conservation_relative<1e-6,"Sparse hierarchy mass not conserved");
}
void synthetic(){
  auto empty=torch::empty({0},torch::kFloat64);
  auto empty_mask=torch::empty({0},torch::kBool);
  auto empty_result=h::build(torch::empty({0,3},torch::kFloat64),torch::empty({0,3},torch::kFloat64),empty,empty_mask,empty_mask);
  check(empty_result.nodes.empty()&&empty_result.dimensions==std::array<int,3>{0,0,0},"Empty sparse grid has dense dimensions");
  check(h::sparseRootKey({-.1,0.,-1.},1.)==h::RootKey({-1,0,-1}),"Negative floor/truncation bug");
  check(h::sparseRootKey({1e6,-1e6,0.},1e-6)[0]>int64_t(2147483647),"Hash key must use int64");
  bool overflow=false;try{h::sparseRootKey({1e300,0,0},1e-12);}catch(const std::exception&){overflow=true;}
  check(overflow,"Unrepresentable key must fail explicitly");
  std::vector<h::Position> two{{.01,.01,.01},{.02,.02,.02},{1e6,1e6,1e6},{1e6+.01,1e6,1e6}};
  auto sparse=h::makeSparseRootGrid(two,std::vector<double>(4,.01),{0,1,2,3});
  check(sparse.roots.size()<=4,"Empty space created root cells");
  size_t entries=0;for(auto& item:sparse.roots) entries+=item.second.size();check(entries==4,"Sparse membership failed");
  std::mt19937_64 rng(20261019);std::uniform_real_distribution<double> random(-1,1);
  std::vector<h::Position> body;std::vector<double>radius;std::vector<int64_t>rows;
  for(int64_t i=0;i<6000;++i){body.push_back({random(rng),random(rng),random(rng)});radius.push_back(.01);rows.push_back(i);}
  auto first=h::makeSparseRootGrid(body,radius,rows);auto repeated=h::makeSparseRootGrid(body,radius,rows);sameGrid(first,repeated);
  body.push_back({1e6,-1e6,1e6});radius.push_back(.01);rows.push_back(6000);
  auto outlier=h::makeSparseRootGrid(body,radius,rows);double change=outlier.h/first.h;
  check(change>.1 && change<10,"One outlier changed body cell scale by an order of magnitude");
  check(outlier.roots.size()<=body.size(),"Sparse memory depends on key span");
  std::cout<<"SPARSE_TEST_PASS A_empty_space C_negative_key D_determinism E_huge_key_range B_outlier_h_ratio="<<change
           <<" body_h="<<first.h<<" outlier_h="<<outlier.h<<" roots="<<outlier.roots.size()<<'\n';

  auto options=torch::TensorOptions().dtype(torch::kFloat64);torch::set_num_threads(2);
  std::vector<double>points;
  for(int x=0;x<16;++x)for(int y=0;y<16;++y)for(int z=0;z<16;++z)
    {points.push_back(.1*x-.8);points.push_back(.1*y-.8);points.push_back(.1*z-.8);}
  auto X=torch::from_blob(points.data(),{4096,3},options).clone(),S=torch::full_like(X,1e-4),W=torch::ones({4096},options);
  auto mask=torch::ones({4096},options.dtype(torch::kBool));
  auto result=h::build(X,S,W,mask,mask);partition(result);
  int split=0;for(auto& n:result.nodes)split+=!n.leaf;check(split>0,"Octree partition fixture must exercise split");
  auto second=h::build(X,S,W,mask,mask);
  check(result.importance.equal(second.importance)&&result.leaf_assignment.equal(second.leaf_assignment),"Sparse tree/output nondeterministic");
  auto changed_mass=h::build(X,S,W*7,mask,mask);
  check(result.h_root==changed_mass.h_root&&result.search_samples.size()==changed_mass.search_samples.size(),
      "Raw SI changed root search");
  for(size_t i=0;i<result.search_samples.size();++i)
    check(result.search_samples[i].h==changed_mass.search_samples[i].h&&
        result.search_samples[i].occupied==changed_mass.search_samples[i].occupied,"Raw SI changed search trace");
  auto noncandidate=mask.clone();noncandidate[4095]=false;
  auto excluded=h::build(X,S,W,noncandidate,mask);
  auto changed_x=X.clone();changed_x[4095]=torch::tensor({1e200,-1e200,1e200},options);
  auto changed_scale=S.clone();changed_scale[4095]=torch::tensor({1e100,1e100,1e100},options);
  auto excluded_repeat=h::build(changed_x,changed_scale,W,noncandidate,mask);
  check(excluded.h_root==excluded_repeat.h_root&&excluded.leaf_assignment.equal(excluded_repeat.leaf_assignment)&&
      excluded.importance.equal(excluded_repeat.importance),"Invalid row changed sparse grid/search/tree");
  auto unobserved=h::build(changed_x,changed_scale,W,mask,noncandidate);
  check(excluded.importance.equal(unobserved.importance),"Unobserved row changed hierarchy population");
  // One completely identical leaf has identical geometry/rows under both backends.
  auto duplicate=torch::full({5,3},.123,options),weights=torch::tensor({10.,9.,8.,7.,6.},options);
  auto five=torch::ones({5},options.dtype(torch::kBool));
  auto legacy=h::build(duplicate,torch::ones_like(duplicate),weights,five,five,h::RootBackend::LegacyAabb2048);
  auto modern=h::build(duplicate,torch::ones_like(duplicate),weights,five,five);
  check(legacy.nodes.size()==1&&modern.nodes.size()==1,"Parity fixture must use same leaf");
  check(legacy.nodes[0].representatives==modern.nodes[0].representatives&&legacy.nodes[0].gains==modern.nodes[0].gains&&
      legacy.nodes[0].quality==modern.nodes[0].quality&&legacy.importance.equal(modern.importance),"Same leaf representative backend parity failed");
  std::cout<<"SPARSE_TEST_PASS F_octree_partition G_same_leaf_representative_parity masks_and_no_mass_in_search\n";
}
void smoke(const std::string& file,const std::filesystem::path& output){
  std::ifstream stream(file,std::ios::binary);int64_t n;stream.read(reinterpret_cast<char*>(&n),8);check(n>0&&n<10000000,"Bad fixture N");
  auto options=torch::TensorOptions().dtype(torch::kFloat64);
  auto x=torch::empty({n,3},options),s=torch::empty({n,3},options),w=torch::empty({n},options);
  auto valid=torch::empty({n},options.dtype(torch::kUInt8)),observed=torch::empty_like(valid);
  for(int64_t i=0;i<n;++i){
    stream.read(reinterpret_cast<char*>(x.data_ptr<double>()+3*i),24);stream.read(reinterpret_cast<char*>(s.data_ptr<double>()+3*i),24);
    stream.read(reinterpret_cast<char*>(w.data_ptr<double>()+i),8);stream.read(reinterpret_cast<char*>(valid.data_ptr<uint8_t>()+i),1);
    stream.read(reinterpret_cast<char*>(observed.data_ptr<uint8_t>()+i),1);
  }
  check(bool(stream),"Incomplete fixture");
  check(torch::cuda::is_available(),"Fixture validation requires existing CUDA device");
  x=x.cuda();s=s.cuda();w=w.cuda();valid=valid.to(torch::kCUDA,torch::kBool);observed=observed.to(torch::kCUDA,torch::kBool);
  for(auto backend:{h::RootBackend::LegacyAabb2048,h::RootBackend::SparseHashTarget2048}){
    auto result=h::build(x,s,w,valid,observed,backend);partition(result);
    std::string stem=backend==h::RootBackend::LegacyAabb2048?"legacy_aabb_2048":"sparse_hash_target_2048";
    h::writeDiagnostics(result,output,stem);
    auto out=result.importance.to(torch::kCPU).contiguous();auto leaf=result.leaf_assignment.contiguous();
    std::ofstream binary(output/(stem+".bin"),std::ios::binary);binary.write(reinterpret_cast<char*>(&n),8);
    binary.write(reinterpret_cast<char*>(out.data_ptr<double>()),8*n);binary.write(reinterpret_cast<char*>(leaf.data_ptr<int64_t>()),8*n);
    check(bool(binary),"Smoke export failed");
    if(backend==h::RootBackend::SparseHashTarget2048){
      auto repeat=h::build(x,s,w,valid,observed,backend);
      check(repeat.h_root==result.h_root&&repeat.nodes.size()==result.nodes.size()&&repeat.leaf_assignment.equal(result.leaf_assignment)&&
          repeat.importance.equal(result.importance),"Real sparse repeat not deterministic");
      for(size_t i=0;i<result.nodes.size();++i){const auto&a=result.nodes[i];const auto&b=repeat.nodes[i];
        check(a.root_key==b.root_key&&a.path==b.path&&a.representatives==b.representatives&&a.quality==b.quality&&a.gains==b.gains,"Real tree/greedy repeat not deterministic");}
    }
    std::cout<<"STRUCTURAL_FIXTURE_PASS "<<stem<<" candidates="<<result.candidates<<" h="<<result.h_root<<" mass_relative="<<result.conservation_relative<<" ms="<<result.build_ms<<'\n';
  }
}
}  // namespace
int main(int argc,char**argv){try{
  torch::set_num_threads(2);torch::manual_seed(20261019);
  if(argc==1)synthetic();else{check(argc==3,"Expected input/output arguments");smoke(argv[1],argv[2]);}
  std::cout<<"SPARSE_ROOT_TEST_PASS\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
