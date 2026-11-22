#include "probe_common.h"
int main(int argc,char** argv) {
  torch::set_num_threads(2);at::globalContext().setAllowTF32CuBLAS(false);const std::string root=argv[1];Scene scene(root,argv[2]);
  std::filesystem::create_directory(root+"/snapshots");
  std::ofstream csv(root+"/nodes.csv"),layout(root+"/gradient_layout.csv");csv<<"deterministic,seed,ratio,comparison,rep,node,bits,total,different,max_abs,relative_l2\n";
  layout<<"deterministic,seed,ratio,fused,table,gather_stride0,gather_stride1,scatter_stride0,scatter_stride1\n";
  const std::vector<double> ratios{.01,.05,.1,.2,.3,.4,.025,.075,.15,.25,.35};
  for(int deterministic:{0,1})for(int seed=0;seed<3;++seed)for(size_t j=0;j<ratios.size();++j) {
    at::globalContext().setDeterministicAlgorithms(deterministic,false);torch::manual_seed(2245+seed);const auto ratio=ratios[j];auto ids=scene.ids(ratio,j);
    auto up=torch::randn({ids.numel(),14},scene.xyz.options());auto reference=scene.capture(ids,ratio,up,false,true);
    for(int rep=0;rep<3;++rep) {
      const bool fused=rep!=0;auto result=scene.capture(ids,ratio,up,fused,true);
      compare_capture(csv,key(deterministic,seed,ratio,fused?"baseline_vs_fused":"baseline_repeat",rep),reference,result);
      for(size_t k=0;k<result.gather_vjp.size();++k)layout<<deterministic<<','<<seed<<','<<ratio<<','<<fused<<','<<k<<','<<result.gather_vjp[k].stride0<<','<<result.gather_vjp[k].stride1<<','<<result.scatter_output[k].stride0<<','<<result.scatter_output[k].stride1<<'\n';
      if(false) {
        torch::serialize::OutputArchive archive;archive.write("owners",scene.net->anchors()->rows().index_select(0,ids).cpu());
        for(size_t k=0;k<reference.gather_vjp.size();++k) {
          archive.write("incoming_a_"+std::to_string(k),reference.gather_vjp[k].value.cpu());archive.write("incoming_b_"+std::to_string(k),result.gather_vjp[k].value.cpu());
          archive.write("scatter_a_"+std::to_string(k),reference.scatter_output[k].value.cpu());archive.write("scatter_b_"+std::to_string(k),result.scatter_output[k].value.cpu());
          archive.write("leaf_a_"+std::to_string(k),reference.leaf_gradients[k].cpu());archive.write("leaf_b_"+std::to_string(k),result.leaf_gradients[k].cpu());
          archive.write("strides_"+std::to_string(k),torch::tensor({reference.gather_vjp[k].stride0,reference.gather_vjp[k].stride1},torch::kInt64));
        }
        archive.save_to(root+"/snapshots/d"+std::to_string(deterministic)+"_r"+std::to_string(j)+"_"+(fused?"fused":"repeat")+".pt");
      }
    }
    if(seed==0&&j<6) {
      auto a=scene.capture(ids,ratio,up,false,true,false),b=scene.capture(ids,ratio,up,false,true,false),c=scene.capture(ids,ratio,up,true,true,false);
      compare_capture(csv,key(deterministic,seed,ratio,"unhooked_baseline_repeat",0),a,b,false);compare_capture(csv,key(deterministic,seed,ratio,"unhooked_baseline_vs_fused",0),a,c,false);
      if(deterministic) {
        auto original=scene.capture(ids,ratio,up,false,true,false,true);
        compare_capture(csv,key(deterministic,seed,ratio,"original_API_vs_traced_baseline",0),a,original,false);
      }
    }
    csv.flush();layout.flush();std::cout<<"NODES_COMPLETE det="<<deterministic<<" seed="<<seed<<" ratio="<<ratio<<" K="<<ids.numel()<<std::endl;
  }
  std::cout<<"NODE_LOCALIZATION_COMPLETE"<<std::endl;
}
