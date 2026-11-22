#include "probe_common.h"
#include <unordered_set>
int main(int argc,char** argv) {
  torch::set_num_threads(2);at::globalContext().setAllowTF32CuBLAS(false);const std::string root=argv[1];
  std::ofstream csv(root+"/scatter_replay.csv");csv<<"ratio_index,K,deterministic,layout,rep,bits,total,different,max_abs,relative_l2\n";
  for(int j=0;j<11;++j) {
    torch::serialize::InputArchive archive;archive.load_from(root+"/snapshots/d0_r"+std::to_string(j)+"_fused.pt",torch::Device(torch::kCUDA));
    torch::Tensor owners,g,previous,strides;archive.read("owners",owners);archive.read("incoming_a_0",g);archive.read("scatter_a_0",previous);archive.read("strides_0",strides);
    const auto a=previous.size(0);strides=strides.cpu();std::vector<int64_t> gs{strides[0].item<int64_t>(),strides[1].item<int64_t>()};
    auto cpu=owners.cpu().contiguous();std::vector<int64_t> unique;std::unordered_set<int64_t> seen;
    for(int64_t i=0;i<cpu.numel();++i)if(seen.insert(cpu.data_ptr<int64_t>()[i]).second)unique.push_back(i);
    for(int deterministic:{0,1})for(int layout=0;layout<3;++layout) {
      at::globalContext().setDeterministicAlgorithms(deterministic,false);
      auto incoming=layout==0?torch::empty_strided(g.sizes(),gs,g.options()):torch::empty_like(g);incoming.copy_(g);
      auto ids=owners;
      if(layout==2){auto subset=torch::tensor(unique,owners.options());ids=owners.index_select(0,subset);incoming=incoming.index_select(0,subset);}
      torch::Tensor reference,min,max;
      for(int rep=0;rep<24;++rep) {
        auto table=torch::zeros({a,48},g.options()).requires_grad_(true);auto selected=table.index_select(0,ids);
        std::string label="ScatterReplay.d"+std::to_string(deterministic)+".layout"+std::to_string(layout)+".r"+std::to_string(j);nvtxRangePushA(label.c_str());
        selected.backward(incoming);nvtxRangePop();auto result=table.grad();
        if(!rep){reference=result.detach().clone();min=reference.clone();max=reference.clone();}
        auto d=difference(reference,result);csv<<j<<','<<ids.numel()<<','<<deterministic<<','<<layout<<','<<rep<<','<<d.equal<<','<<d.total<<','<<d.different<<','<<d.absolute<<','<<d.relative_l2<<'\n';
        min=torch::minimum(min,result);max=torch::maximum(max,result);
      }
      torch::serialize::OutputArchive out;out.write("owners",ids.cpu());out.write("incoming",incoming.cpu());out.write("first",reference.cpu());out.write("minimum",min.cpu());out.write("maximum",max.cpu());
      out.save_to(root+"/snapshots/scatter_d"+std::to_string(deterministic)+"_l"+std::to_string(layout)+"_r"+std::to_string(j)+".pt");
    }
    csv.flush();std::cout<<"SCATTER_ISOLATION_COMPLETE r="<<j<<" K="<<g.size(0)<<std::endl;
  }
}
