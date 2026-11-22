// 诊断导出：链接冻结hierarchy库，仅复用selectCell；不改变结构或数学。
#include <torch/torch.h>
#include "include/hierarchical_representative_si.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
int main(int argc,char**argv){try{
  TORCH_CHECK(argc==4,"input.bin leaf.bin output-dir");torch::set_num_threads(2);torch::NoGradGuard guard;
  std::ifstream f(argv[1],std::ios::binary);int64_t n;f.read((char*)&n,8);TORCH_CHECK(n>0,"N");
  auto o=torch::TensorOptions().dtype(torch::kFloat64);
  auto x=torch::empty({n,3},o),s=torch::empty({n,3},o),w=torch::empty({n},o);
  auto valid=torch::empty({n},o.dtype(torch::kUInt8)),observed=torch::empty_like(valid);
  for(int64_t i=0;i<n;++i){f.read((char*)(x.data_ptr<double>()+3*i),24);f.read((char*)(s.data_ptr<double>()+3*i),24);f.read((char*)(w.data_ptr<double>()+i),8);f.read((char*)(valid.data_ptr<uint8_t>()+i),1);f.read((char*)(observed.data_ptr<uint8_t>()+i),1);}
  TORCH_CHECK(bool(f),"input incomplete");
  std::ifstream a(argv[2],std::ios::binary);int64_t nb;a.read((char*)&nb,8);TORCH_CHECK(nb==n,"population changed");
  auto expected=torch::empty({n},o);auto leaf=torch::empty({n},o.dtype(torch::kInt64));
  a.read((char*)expected.data_ptr<double>(),8*n);a.read((char*)leaf.data_ptr<int64_t>(),8*n);TORCH_CHECK(bool(a),"leaf input incomplete");
  std::map<int64_t,std::vector<int64_t>> leaves;
  for(int64_t i=0;i<n;++i){bool candidate=valid[i].item<uint8_t>()&&observed[i].item<uint8_t>();TORCH_CHECK(candidate==(leaf[i].item<int64_t>()>=0),"candidate mismatch");if(candidate)leaves[leaf[i].item<int64_t>()].push_back(i);}
  x=x.cuda();s=s.cuda();w=w.cuda();auto radius=torch::exp(s.log().mean(1));auto cover=torch::zeros_like(w);auto output=torch::zeros_like(w);
  std::ofstream reps(std::string(argv[3])+"/leaf_reps.csv");reps<<std::setprecision(17)<<"leaf,position,source_row,Delta\n";
  for(const auto& item:leaves){auto rows=torch::tensor(item.second,torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));auto weight=w.index_select(0,rows);
    auto selection=hierarchical_si::selectCell(x.index_select(0,rows),radius.index_select(0,rows),weight,rows);
    auto local=weight*(1-selection.cover);
    for(size_t k=0;k<selection.representatives.size();++k){auto j=selection.representatives[k];local[j]=selection.gains[k];reps<<item.first<<','<<k+1<<','<<item.second[j]<<','<<selection.gains[k]<<'\n';}
    cover.index_copy_(0,rows,selection.cover);output.index_copy_(0,rows,local);
  }
  TORCH_CHECK(output.cpu().equal(expected),"STOP reconstructed final score not EXACT");
  auto residual=w*(1-cover);auto gap=1-cover;
  std::ofstream out(std::string(argv[3])+"/coverage.bin",std::ios::binary);out.write((char*)&n,8);
  for(const auto& t:{radius,cover,gap,residual}){auto cpu=t.cpu().contiguous();out.write((char*)cpu.data_ptr<double>(),8*n);}
  TORCH_CHECK(bool(out)&&bool(reps),"output failure");std::cout<<"COVERAGE_EXACT_PASS N="<<n<<" leaves="<<leaves.size()<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
