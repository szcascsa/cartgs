#include <torch/torch.h>
#include "include/hierarchical_representative_si.h"
#include <fstream>
#include <iostream>
#include <set>
#include <cmath>
namespace h=hierarchical_si;
void same(const h::Result&a,const h::Result&b){
  TORCH_CHECK(a.h_root==b.h_root&&a.d50==b.d50&&a.nodes.size()==b.nodes.size()&&a.importance.equal(b.importance)&&a.leaf_assignment.equal(b.leaf_assignment),"repeat changed");
  for(size_t i=0;i<a.nodes.size();++i){auto&x=a.nodes[i];auto&y=b.nodes[i];TORCH_CHECK(x.root_key==y.root_key&&x.path==y.path&&x.representatives==y.representatives&&x.gains==y.gains&&x.quality==y.quality,"tree changed");}
}
void partition(const h::Result&r){
  for(size_t i=0;i<r.nodes.size();++i){auto&n=r.nodes[i];TORCH_CHECK(n.level<=3,"L4");
    if(n.leaf)continue;int64_t sum=0;
    for(auto&c:r.nodes)if(c.root==n.root&&c.level==n.level+1&&c.path.size()==n.path.size()+1&&c.path.substr(0,n.path.size())==n.path)sum+=c.occupancy;
    TORCH_CHECK(sum==n.occupancy,"child occupancy lost");
  }
  TORCH_CHECK((r.leaf_assignment>=0).sum().item<int64_t>()==r.candidates,"lost rows");
}
void synthetic(){
  std::vector<h::Position>x{{-.1,0,0},{.1,0,0},{1e6,-1e6,1e6}};std::vector<double>r(3,1./64);std::vector<int64_t>ids{0,1,2};
  auto g=h::makeD50NormalizedRootGrid(x,r,ids),repeat=h::makeD50NormalizedRootGrid(x,r,ids);
  TORCH_CHECK(g.h==1&&g.d50==1./32&&g.h/g.d50==32&&g.roots==repeat.roots,"A/B deterministic d50");
  TORCH_CHECK(g.roots.count({-1,0,0})&&g.roots.size()==3&&g.samples.empty()&&g.search_ms==0,"C/D floor sparse no search");
  std::vector<h::Position>body;std::vector<double>rad;std::vector<int64_t>rows;
  for(int i=0;i<101;++i){body.push_back({i*.001,0,0});rad.push_back(.01);rows.push_back(i);}
  auto before=h::makeD50NormalizedRootGrid(body,rad,rows);body.push_back({1e6,1e6,1e6});rad.push_back(1e4);rows.push_back(101);
  auto after=h::makeD50NormalizedRootGrid(body,rad,rows);TORCH_CHECK(before.h==after.h&&after.roots.size()<=body.size(),"E outlier changed scale");
  auto o=torch::TensorOptions().dtype(torch::kFloat64);std::vector<double>points;
  for(int a=0;a<8;++a)for(int b=0;b<8;++b)for(int c=0;c<8;++c){points.push_back(.0001+.0008*a);points.push_back(.0001+.0008*b);points.push_back(.0001+.0008*c);}
  auto X=torch::from_blob(points.data(),{512,3},o).clone();auto S=torch::full_like(X,1e-4);auto W=torch::ones({512},o);auto mask=torch::ones({512},o.dtype(torch::kBool));
  auto a=h::build(X,S,W,mask,mask,h::RootBackend::SparseHashD50x32),b=h::build(X,S,W,mask,mask,h::RootBackend::SparseHashD50x32);same(a,b);partition(a);
  int split=0;for(auto&n:a.nodes)split+=!n.leaf;TORCH_CHECK(split>1&&a.h_root==32*a.d50,"G/H/I split fixture");
  auto XX=torch::cat({X,torch::tensor({{1e6,1e6,1e6},{-1e6,-1e6,-1e6}},o)});auto SS=torch::cat({S,torch::ones({2,3},o)});auto WW=torch::ones({514},o);
  auto valid=torch::ones({514},o.dtype(torch::kBool)),obs=valid.clone();valid[512]=false;obs[513]=false;
  auto excluded=h::build(XX,SS,WW,valid,obs,h::RootBackend::SparseHashD50x32);
  XX[512]=torch::tensor({1e12,-1e12,1e12},o);XX[513]=torch::tensor({-1e12,1e12,-1e12},o);SS[512]=1e8;SS[513]=1e9;
  auto excluded2=h::build(XX,SS,WW,valid,obs,h::RootBackend::SparseHashD50x32);same(excluded,excluded2);
  TORCH_CHECK(a.h_root==excluded.h_root&&excluded.importance.slice(0,0,512).equal(a.importance),"F excluded row changed scale/tree");
  auto bad=S.clone();bad[0][0]=0;bool threw=false;try{h::build(X,bad,W,mask,mask,h::RootBackend::SparseHashD50x32);}catch(const std::exception&){threw=true;}TORCH_CHECK(threw,"invalid positive candidate scale not rejected");
  std::cout<<"SYNTHETIC_PASS A_d50_repeat B_32_d50 C_negative_floor D_sparse_only E_outlier_h_ratio="<<after.h/before.h<<" F_masks G_child_partition H_unique I_C_repeat mass_error="<<a.conservation_relative<<std::endl;
}
int main(int argc,char**argv){try{
  torch::set_num_threads(2);torch::NoGradGuard guard;if(argc==2&&std::string(argv[1])=="--synthetic"){synthetic();return 0;}
  TORCH_CHECK(argc==4,"input output backend(0|32)");std::ifstream f(argv[1],std::ios::binary);int64_t n;f.read((char*)&n,8);TORCH_CHECK(n>0,"N");auto o=torch::TensorOptions().dtype(torch::kFloat64);
  auto x=torch::empty({n,3},o),s=torch::empty({n,3},o),w=torch::empty({n},o);auto v=torch::empty({n},o.dtype(torch::kUInt8)),b=torch::empty_like(v);
  for(int64_t i=0;i<n;++i){f.read((char*)(x.data_ptr<double>()+3*i),24);f.read((char*)(s.data_ptr<double>()+3*i),24);f.read((char*)(w.data_ptr<double>()+i),8);f.read((char*)(v.data_ptr<uint8_t>()+i),1);f.read((char*)(b.data_ptr<uint8_t>()+i),1);}TORCH_CHECK(bool(f),"input incomplete");
  x=x.cuda();s=s.cuda();w=w.cuda();v=v.to(torch::kCUDA,torch::kBool);b=b.to(torch::kCUDA,torch::kBool);
  TORCH_CHECK(std::string(argv[3])=="0"||std::string(argv[3])=="32","only old or fixed32");auto backend=std::string(argv[3])=="32"?h::RootBackend::SparseHashD50x32:h::RootBackend::SparseHashTarget2048;
  auto a=h::build(x,s,w,v,b,backend),repeat=h::build(x,s,w,v,b,backend);same(a,repeat);partition(a);h::writeDiagnostics(a,argv[2],"hierarchy");
  auto score=a.importance.cpu().contiguous(),leaf=a.leaf_assignment.cpu().contiguous();std::ofstream out(std::string(argv[2])+"/importance.bin",std::ios::binary);out.write((char*)&n,8);out.write((char*)score.data_ptr<double>(),8*n);out.write((char*)leaf.data_ptr<int64_t>(),8*n);TORCH_CHECK(bool(out),"output incomplete");
  std::cout<<"FIXTURE_PASS backend="<<argv[3]<<" d50="<<a.d50<<" h="<<a.h_root<<" mass_relative="<<a.conservation_relative<<" build_ms="<<a.build_ms<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
