#define main fixed_training_main
#include MLP_FIXED_WORKER
#undef main

int main(int argc,char** argv){
  TORCH_CHECK(argc==7,"probe manifest input weights ratio seed output");
  torch::set_num_threads(2);torch::manual_seed(std::stoi(argv[5]));
  at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
  FrozenScene scene(readJson(argv[1]));
  std::ifstream input(argv[2],std::ios::binary);int64_t n;input.read((char*)&n,8);
  std::vector<int64_t> ids(n);std::vector<double> base(n);
  input.read((char*)ids.data(),n*8);input.read((char*)base.data(),n*8);TORCH_CHECK(input.good(),"Input truncated");
  auto rows=torch::tensor(ids,torch::kInt64).cuda(),b=torch::tensor(base,torch::kFloat64).cuda();
  auto uniform=torch::rand_like(b),xyz=scene.normalized(rows);
  // 此探针保留bias以验证其gradient为0；正式修复worker的head没有bias。
  torch::nn::Sequential net(torch::nn::Linear(3,32),torch::nn::SiLU(),torch::nn::Linear(32,32),torch::nn::SiLU(),torch::nn::Linear(32,1));
  net->to(torch::kCUDA,torch::kFloat64);torch::serialize::InputArchive archive;archive.load_from(argv[3],torch::kCUDA);net->load(archive);
  auto r=net->forward(xyz).flatten();r.retain_grad();auto a=r-r.mean();a.retain_grad();
  double ratio=std::stod(argv[4]);int64_t k=int64_t(ratio*n+.5);
  auto calibrated=calibratedSelectionProbability(b+a,k);auto probability=calibrated.first;
  auto oldprob=(b+a-calibrated.second).sigmoid();
  TORCH_CHECK(probability.detach().equal(oldprob.detach()),"FORWARD_CHANGED");
  auto mask=sampleSelectorST(probability,uniform),local=torch::nonzero(mask.detach()>0).flatten();
  auto loss=scene.photo(rows.index_select(0,local),mask.index_select(0,local),scene.bank().front(),float(ratio),nullptr);
  loss.backward();auto g=a.grad(),rawgrad=r.grad();
  auto relative=g.sum().abs()/(g.abs().sum()+1e-30);
  auto bias=net->at<torch::nn::LinearImpl>(4).bias.grad().item<double>();
  TORCH_CHECK(relative.item<double>()<1e-12&&std::abs(bias)<1e-12,"GAUGE_GRADIENT_LEAK");
  TORCH_CHECK(torch::isfinite(g).all().item<bool>(),"NONFINITE_GRADIENT");scene.verify();
  Json::Value result;result["all_passed"]=true;result["forward_probability_exact"]=true;
  result["score_grad_sum"]=g.sum().item<double>();result["raw_output_grad_sum"]=rawgrad.sum().item<double>();
  result["normalized_sum_gradient"]=relative.item<double>();result["output_bias_gradient"]=bias;
  result["frozen_exact"]=true;result["optimizer_steps"]=0;result["sampled_N"]=Json::Int64(local.numel());
  Json::StreamWriterBuilder writer;std::ofstream out(argv[6]);out<<Json::writeString(writer,result)<<std::endl;
}
