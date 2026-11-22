#include <torch/torch.h>
#include "include/hierarchical_representative_si.h"
#include <fstream>
#include <iostream>
int main(int argc,char**argv){try{
  TORCH_CHECK(argc==3,"input/output");torch::set_num_threads(2);torch::NoGradGuard guard;
  std::ifstream f(argv[1],std::ios::binary);int64_t n;f.read((char*)&n,8);TORCH_CHECK(n>0,"N");
  auto o=torch::TensorOptions().dtype(torch::kFloat64);
  auto x=torch::empty({n,3},o),s=torch::empty({n,3},o),w=torch::empty({n},o);
  auto v=torch::empty({n},o.dtype(torch::kUInt8)),b=torch::empty_like(v);
  for(int64_t i=0;i<n;++i){f.read((char*)(x.data_ptr<double>()+3*i),24);f.read((char*)(s.data_ptr<double>()+3*i),24);f.read((char*)(w.data_ptr<double>()+i),8);f.read((char*)(v.data_ptr<uint8_t>()+i),1);f.read((char*)(b.data_ptr<uint8_t>()+i),1);}
  TORCH_CHECK(bool(f),"input incomplete");x=x.cuda();s=s.cuda();w=w.cuda();v=v.to(torch::kCUDA,torch::kBool);b=b.to(torch::kCUDA,torch::kBool);
  auto a=hierarchical_si::build(x,s,w,v,b);auto r=hierarchical_si::build(x,s,w,v,b);
  TORCH_CHECK(a.importance.equal(r.importance)&&a.leaf_assignment.equal(r.leaf_assignment)&&a.h_root==r.h_root&&a.nodes.size()==r.nodes.size(),"repeat changed");
  for(size_t i=0;i<a.nodes.size();++i){auto&u=a.nodes[i];auto&z=r.nodes[i];TORCH_CHECK(u.root_key==z.root_key&&u.path==z.path&&u.representatives==z.representatives&&u.gains==z.gains&&u.quality==z.quality,"tree repeat changed");}
  hierarchical_si::writeDiagnostics(a,argv[2],"sparse_hash_target_2048");
  auto score=a.importance.cpu().contiguous();auto leaf=a.leaf_assignment.cpu().contiguous();
  std::ofstream out(std::string(argv[2])+"/sparse_hash_target_2048.bin",std::ios::binary);out.write((char*)&n,8);out.write((char*)score.data_ptr<double>(),8*n);out.write((char*)leaf.data_ptr<int64_t>(),8*n);TORCH_CHECK(bool(out),"output");
  std::cout<<"C_REPEAT_EXACT candidates="<<a.candidates<<" h="<<a.h_root<<" mass_error="<<a.conservation_relative<<std::endl;
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
