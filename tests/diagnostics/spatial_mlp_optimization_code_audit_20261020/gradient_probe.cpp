// 复用已审计 worker 的 FrozenScene；只在保存的 checkpoint 上 backward，不做 optimizer step。
#define main legacy_training_main
#include MLP_LEGACY_WORKER
#undef main

int main(int argc, char** argv) {
  TORCH_CHECK(argc == 7, "probe manifest input.bin weights.pt ratio seed output.json");
  torch::set_num_threads(2);
  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setAllowTF32CuDNN(false);
  const double ratio = std::stod(argv[4]);
  torch::manual_seed(std::stoi(argv[5]));
  FrozenScene scene(readJson(argv[1]));
  std::ifstream input(argv[2], std::ios::binary);
  int64_t n;
  input.read((char*)&n, 8);
  std::vector<int64_t> ids(n);
  std::vector<double> base(n);
  input.read((char*)ids.data(), n * 8);
  input.read((char*)base.data(), n * 8);
  TORCH_CHECK(input.good(), "Input truncated");
  auto rows = torch::tensor(ids, torch::kInt64).cuda();
  auto b = torch::tensor(base, torch::kFloat64).cuda();
  auto uniform = torch::rand_like(b);
  auto normalized = scene.normalized(rows);
  torch::nn::Sequential net(torch::nn::Linear(3,32),torch::nn::SiLU(),
      torch::nn::Linear(32,32),torch::nn::SiLU(),torch::nn::Linear(32,1));
  net->to(torch::kCUDA, torch::kFloat64);
  torch::serialize::InputArchive archive;
  archive.load_from(argv[3], torch::kCUDA);
  net->load(archive);
  const int64_t k = int64_t(ratio * n + .5);
  auto calibrate = [&](T score) {
    torch::NoGradGuard guard;
    double low=score.min().item<double>()-40, high=score.max().item<double>()+40;
    for(int i=0;i<42;++i) {
      double mid=.5*(low+high);
      if((score-mid).sigmoid().sum().item<double>()>k) low=mid; else high=mid;
    }
    return .5*(low+high);
  };
  auto a=net->forward(normalized).flatten();
  a.retain_grad();
  auto score=b+a;
  auto p=(score-calibrate(score)).sigmoid();
  p.retain_grad();
  auto mask=sampleSelectorST(p,uniform);
  auto local=torch::nonzero(mask.detach()>0).flatten();
  const int frame=scene.bank().front();
  auto loss=scene.photo(rows.index_select(0,local),mask.index_select(0,local),frame,float(ratio),nullptr);
  loss.backward();
  auto g=a.grad().detach();
  auto h=p.grad().detach();
  TORCH_CHECK(torch::isfinite(g).all().item<bool>(),"Nonfinite gradient");
  auto d=p.detach()*(1-p.detach());
  auto centered=g-g.mean();
  auto calibrated=g-d*g.sum()/d.sum();
  torch::NoGradGuard guard;
  auto hidden=net->at<torch::nn::LinearImpl>(0).forward(normalized);
  hidden=torch::silu(hidden);
  hidden=torch::silu(net->at<torch::nn::LinearImpl>(2).forward(hidden));
  auto common_w=hidden.mean(0)*g.sum();
  auto relative_w=((hidden-hidden.mean(0))*g.unsqueeze(1)).sum(0);
  auto full_w=net->at<torch::nn::LinearImpl>(4).weight.grad().flatten();
  auto shifted=(score.detach()+100-calibrate(score.detach()+100)).sigmoid();
  auto m=torch::zeros({n},rows.options().dtype(torch::kBool));m.index_fill_(0,local,true);
  scene.verify();
  Json::Value z;
  z["N"]=Json::Int64(n);z["K"]=Json::Int64(k);z["sampled_N"]=Json::Int64(local.numel());
  z["Bank_frame"]=frame;z["photo"]=loss.item<double>();
  z["checkpoint"]=argv[3];z["ratio"]=ratio;z["seed"]=std::stoi(argv[5]);
  z["sum_output_grad"]=g.sum().item<double>();
  z["output_bias_grad"]=net->at<torch::nn::LinearImpl>(4).bias.grad().item<double>();
  z["output_grad_norm"]=g.norm().item<double>();
  z["unselected_direct_nonzero_grad"]=Json::Int64(g.masked_select(~m).ne(0).sum().item<int64_t>());
  z["shift100_probability_max_error"]=(shifted-p.detach()).abs().max().item<double>();
  z["shift100_sample_exact"]=torch::equal(shifted>uniform,p.detach()>uniform);
  z["sigmoid_chain_rule_max_error"]=(h*d-g).abs().max().item<double>();
  z["centered_gradient_sum"]=centered.sum().item<double>();
  z["calibrated_gradient_sum"]=calibrated.sum().item<double>();
  z["centered_calibrated_relative_gap"]=(centered-calibrated).norm().item<double>()/calibrated.norm().item<double>();
  z["last_weight_grad_norm"]=full_w.norm().item<double>();
  z["last_weight_common_norm"]=common_w.norm().item<double>();
  z["last_weight_relative_norm"]=relative_w.norm().item<double>();
  z["last_weight_decomposition_error"]=(full_w-common_w-relative_w).abs().max().item<double>();
  z["common_weight_cosine"]=(full_w*common_w).sum().item<double>()/(full_w.norm().item<double>()*common_w.norm().item<double>());
  z["frozen_exact"]=true;z["optimizer_steps"]=0;
  Json::StreamWriterBuilder writer;writer["indentation"]="  ";
  std::ofstream out(argv[6]);out<<Json::writeString(writer,z)<<std::endl;
  std::cout<<Json::writeString(writer,z)<<std::endl;
}
