// 研究用cached推理：直接复用训练的校准器和ST sampler，forward不改变opacity。
#include <torch/torch.h>
#include "calibrated_selection.h"
#include "include/selector_handoff.h"
#include <json/json.h>
#include <fstream>
#include <sstream>
#include <algorithm>

int main(int argc,char** argv){
  TORCH_CHECK(argc==2,"cached_inference jobs.txt");
  torch::set_num_threads(2);torch::NoGradGuard guard;
  std::ifstream jobs(argv[1]);std::string line;
  while(std::getline(jobs,line)){
    std::istringstream fields(line);std::string statepath,inputpath,outpath;int64_t k;
    fields>>statepath>>inputpath>>k>>outpath;TORCH_CHECK(!outpath.empty(),"Invalid job");
    torch::serialize::InputArchive state;state.load_from(statepath,torch::kCUDA);
    torch::Tensor score,uniform;state.read("score",score,true);state.read("cached_uniform",uniform,true);
    TORCH_CHECK(score.scalar_type()==torch::kFloat64&&score.sizes()==uniform.sizes(),"Invalid state");
    std::ifstream input(inputpath,std::ios::binary);int64_t n;input.read((char*)&n,8);
    std::vector<int64_t> ids(n);input.read((char*)ids.data(),8*n);TORCH_CHECK(input.good()&&n==score.numel(),"Row mismatch");
    auto probability=calibratedSelectionProbability(score,k).first;
    auto mask=sampleSelectorST(probability,uniform);
    TORCH_CHECK((mask.eq(0)|mask.eq(1)).all().item<bool>(),"NONBINARY_FORWARD");
    auto local=torch::nonzero(mask>0).flatten();TORCH_CHECK(local.numel()>0,"Empty sample");
    auto repeated=sampleSelectorST(calibratedSelectionProbability(score,k).first,uniform);
    TORCH_CHECK(mask.equal(repeated),"NONDETERMINISTIC_CACHED_FORWARD");
    auto indices=local.cpu().contiguous();auto ptr=indices.data_ptr<int64_t>();
    Json::Value z;z["N"]=Json::Int64(n);z["K"]=Json::Int64(k);z["actual_count"]=Json::Int64(local.numel());
    z["expected_count"]=probability.sum().item<double>();z["min_sampling_margin"]=(probability-uniform).abs().min().item<double>();
    z["binary_forward_exact"]=true;z["repeated_exact"]=true;
    for(int64_t j=0;j<indices.numel();++j)z["selected_source_rows"].append(Json::Int64(ids[ptr[j]]));
    Json::StreamWriterBuilder builder;builder["indentation"]="";
    std::ofstream out(outpath);out<<Json::writeString(builder,z)<<std::endl;
  }
}
