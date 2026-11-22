// 仅复用冻结scene、ranking和已修复校准器；原main不执行。
#define main per_gaussian_training_main
#include FIXED_MLP_WORKER
#undef main

int main(int argc,char** argv){
  TORCH_CHECK(argc==10,"worker manifest input.bin grid.bin outdir ratio seed steps lr arm");
  torch::set_num_threads(2);at::globalContext().setAllowTF32CuBLAS(false);at::globalContext().setAllowTF32CuDNN(false);
  const double ratio=std::stod(argv[5]),lr=std::stod(argv[8]);const int seed=std::stoi(argv[6]),steps=std::stoi(argv[7]);
  const bool independent=std::string(argv[9])=="independent";TORCH_CHECK(independent||std::string(argv[9])=="mlp","Invalid arm");
  torch::manual_seed(seed);FrozenScene scene(readJson(argv[1]));scene.verify();
  std::ifstream input(argv[2],std::ios::binary);int64_t n;input.read((char*)&n,8);
  std::vector<int64_t> ids(n);std::vector<double> base(n);input.read((char*)ids.data(),8*n);input.read((char*)base.data(),8*n);TORCH_CHECK(input.good(),"Input truncated");
  auto rows=torch::tensor(ids,torch::kInt64).cuda(),b=torch::tensor(base,torch::kFloat64).cuda();auto uniform=torch::rand_like(b);
  std::ifstream grid(argv[3],std::ios::binary);int64_t gn,m;grid.read((char*)&gn,8);grid.read((char*)&m,8);TORCH_CHECK(gn==n&&m<n,"Grid N/M mismatch");
  std::vector<int64_t> ci(n),oc(m);std::vector<double> centers(m*3);grid.read((char*)ci.data(),8*n);grid.read((char*)centers.data(),8*m*3);grid.read((char*)oc.data(),8*m);TORCH_CHECK(grid.good(),"Grid truncated");
  auto cell=torch::tensor(ci,torch::kInt64).cuda(),occupancy=torch::tensor(oc,torch::kInt64).cuda();
  auto normalized=torch::tensor(centers,torch::kFloat64).reshape({m,3}).cuda();auto weights=occupancy.to(torch::kFloat64)/n;
  TORCH_CHECK(occupancy.sum().item<int64_t>()==n&&occupancy.min().item<int64_t>()>0&&cell.min().item<int64_t>()==0&&cell.max().item<int64_t>()==m-1,"Invalid mapping");
  auto mapcopy=cell.clone(),normcopy=normalized.clone(),occopy=occupancy.clone();
  torch::nn::Sequential net(torch::nn::Linear(3,32),torch::nn::SiLU(),torch::nn::Linear(32,32),torch::nn::SiLU(),torch::nn::Linear(torch::nn::LinearOptions(32,1).bias(false)));
  net->to(torch::kCUDA,torch::kFloat64);{torch::NoGradGuard guard;net->at<torch::nn::LinearImpl>(4).weight.zero_();}
  auto scalar=torch::zeros({m},b.options().requires_grad(true));
  auto params=independent?std::vector<T>{scalar}:net->parameters();torch::optim::Adam optimizer(params,torch::optim::AdamOptions(lr));
  auto rawOutput=[&](){auto r=independent?scalar:net->forward(normalized).flatten();TORCH_CHECK(r.numel()==m,"FORWARD_NOT_CELLS");return r;};
  auto field=[&](T r){return r-(r*weights).sum();};
  auto raw=ranking(b,rows);const int64_t K=int64_t(ratio*n+.5);auto rawlocal=torch::tensor(std::vector<int64_t>(raw.begin(),raw.begin()+K),torch::kInt64).cuda();
  auto kraw=torch::zeros({m},cell.options());kraw.index_add_(0,cell.index_select(0,rawlocal),torch::ones({K},cell.options()));
  std::vector<int64_t> grouped(n);std::iota(grouped.begin(),grouped.end(),0);
  std::sort(grouped.begin(),grouped.end(),[&](auto i,auto j){return ci[i]!=ci[j]?ci[i]<ci[j]:base[i]!=base[j]?base[i]>base[j]:ids[i]<ids[j];});
  std::vector<int64_t> ls,rs;for(int64_t i=1;i<n;++i)if(ci[grouped[i-1]]==ci[grouped[i]]){ls.push_back(grouped[i-1]);rs.push_back(grouped[i]);}
  auto left=torch::tensor(ls,torch::kInt64).cuda(),right=torch::tensor(rs,torch::kInt64).cuda();
  auto validate=[&](T a,T score){
    TORCH_CHECK(torch::isfinite(a).all().item<bool>()&&torch::isfinite(a.exp()).all().item<bool>(),"NONFINITE_CELL_COEFFICIENT");
    auto ga=a.index_select(0,cell);TORCH_CHECK(ga.index_select(0,left).equal(ga.index_select(0,right)),"CELL_NOT_SHARED_EXACT");
    const double err=(a*weights).sum().abs().item<double>();TORCH_CHECK(err<1e-12*std::max(1.,a.abs().max().item<double>()),"WEIGHTED_GAUGE");
    auto bl=b.index_select(0,left),br=b.index_select(0,right),sl=score.index_select(0,left),sr=score.index_select(0,right);
    TORCH_CHECK((torch::where(bl>br,sl>sr,sl==sr)).all().item<bool>(),"INTRA_CELL_RANKING_VIOLATION");
  };
  auto initial=field(rawOutput());TORCH_CHECK(initial.eq(0).all().item<bool>()&&ranking(b+initial.index_select(0,cell),rows)==raw,"INITIAL_NOT_RAW");validate(initial,b);
  auto direct=torch::zeros({n},cell.options()),gradema=torch::zeros({m},b.options()),lastgrad=torch::zeros({m},b.options());
  std::vector<int64_t> previous;std::mt19937 schedule(seed);auto frames=scene.bank();size_t cursor=frames.size();
  auto saveState=[&](int step){
    torch::NoGradGuard guard;auto r=rawOutput(),a=field(r),ga=a.index_select(0,cell);
    torch::serialize::OutputArchive out;for(auto kv:std::vector<std::pair<std::string,T>>{{"cell_raw_output",r},{"cell_coefficient",a},{"gaussian_correction",ga},{"score",b+ga},{"cell_id",cell},{"occupancy",occupancy},{"normalized_cell_centers",normalized},{"gradient_ema",gradema},{"last_raw_output_gradient",lastgrad},{"direct_selected_count",direct},{"cached_uniform",uniform}})out.write(kv.first,kv.second.detach().cpu(),true);
    out.save_to(std::string(argv[4])+"/step_"+std::to_string(step)+".pt");
    if(!independent){torch::serialize::OutputArchive w;net->save(w);w.save_to(std::string(argv[4])+"/step_"+std::to_string(step)+"_weights.pt");}
  };
  auto hardBank=[&](int step){
    torch::NoGradGuard guard;auto a=field(rawOutput()),score=b+a.index_select(0,cell);validate(a,score);auto order=ranking(score,rows);order.resize(K);
    auto local=torch::tensor(order,torch::kInt64).cuda(),selected=std::get<0>(rows.index_select(0,local).sort());double photo=0,psnr=0;
    for(auto frame:scene.bank()){double ps;auto loss=scene.photo(selected,torch::ones({K},b.options()),frame,float(ratio),&ps);photo+=loss.item<double>();psnr+=ps;Json::Value z;z["kind"]="hard_view";z["step"]=step;z["frame"]=frame;z["photo"]=loss.item<double>();z["psnr"]=ps;emit(z);}
    auto kc=torch::zeros({m},cell.options());kc.index_add_(0,cell.index_select(0,local),torch::ones({K},cell.options()));auto delta=kc-kraw;
    std::sort(order.begin(),order.end());std::vector<int64_t> rawsorted(raw.begin(),raw.begin()+K),intersection,prev;std::sort(rawsorted.begin(),rawsorted.end());
    std::set_intersection(order.begin(),order.end(),rawsorted.begin(),rawsorted.end(),std::back_inserter(intersection));if(!previous.empty())std::set_intersection(order.begin(),order.end(),previous.begin(),previous.end(),std::back_inserter(prev));
    Json::Value z;z["kind"]="hard_summary";z["step"]=step;z["photo"]=photo/scene.bank().size();z["psnr"]=psnr/scene.bank().size();z["Jaccard_Raw"]=double(intersection.size())/(2*K-intersection.size());z["Jaccard_previous"]=previous.empty()?1.:double(prev.size())/(2*K-prev.size());z["added"]=Json::Int64(K-intersection.size());z["removed"]=Json::Int64(K-intersection.size());
    z["changed_cells"]=delta.ne(0).sum().item<int>();z["increased_cells"]=delta.gt(0).sum().item<int>();z["decreased_cells"]=delta.lt(0).sum().item<int>();z["budget_transferred"]=Json::Int64(delta.clamp_min(0).sum().item<int64_t>());z["negative_delta_sum"]=Json::Int64(delta.clamp_max(0).sum().item<int64_t>());z["max_abs_delta"]=Json::Int64(delta.abs().max().item<int64_t>());z["violated_cells"]=0;
    for(auto id:order)z["selected_source_rows"].append(Json::Int64(ids[id]));emit(z);previous=order;
    torch::serialize::OutputArchive out;for(auto kv:std::vector<std::pair<std::string,T>>{{"raw_count",kraw},{"selected_count",kc},{"delta",delta},{"gradient_ema",gradema},{"raw_output_gradient",lastgrad},{"cell_coefficient",a}})out.write(kv.first,kv.second.cpu(),true);out.save_to(std::string(argv[4])+"/diagnostic_"+std::to_string(step)+".pt");
  };
  hardBank(0);saveState(0);
  for(int step=1;step<=steps;++step){
    if(cursor==frames.size()){std::shuffle(frames.begin(),frames.end(),schedule);cursor=0;}int frame=frames[cursor++];
    auto r=rawOutput();r.retain_grad();auto a=field(r),ga=a.index_select(0,cell),score=b+ga;score.retain_grad();
    auto calibrated=calibratedSelectionProbability(score,K);auto p=calibrated.first,mask=sampleSelectorST(p,uniform),local=torch::nonzero(mask.detach()>0).flatten();TORCH_CHECK(local.numel()>0,"EMPTY_SAMPLE");
    direct.index_add_(0,local,torch::ones({local.numel()},direct.options()));double psnr;auto loss=scene.photo(rows.index_select(0,local),mask.index_select(0,local),frame,float(ratio),&psnr);
    TORCH_CHECK(torch::isfinite(loss).item<bool>(),"NONFINITE_LOSS");optimizer.zero_grad();loss.backward();double gn=0;
    for(auto& param:params){TORCH_CHECK(param.grad().defined()&&torch::isfinite(param.grad()).all().item<bool>(),"NONFINITE_PARAM_GRADIENT");gn+=param.grad().square().sum().item<double>();}
    lastgrad=r.grad().detach().clone();TORCH_CHECK(torch::isfinite(lastgrad).all().item<bool>(),"NONFINITE_CELL_GRADIENT");gradema=.95*gradema+.05*lastgrad.abs();
    const double normalizedsum=score.grad().sum().abs().item<double>()/(score.grad().abs().sum().item<double>()+1e-30);TORCH_CHECK(normalizedsum<1e-12,"CALIBRATION_GRADIENT_SUM");
    optimizer.step();T after;{torch::NoGradGuard guard;after=field(rawOutput());validate(after,b+after.index_select(0,cell));}
    Json::Value z;z["kind"]="training";z["step"]=step;z["frame"]=frame;z["N"]=Json::Int64(n);z["M"]=Json::Int64(m);z["MLP_forward_count"]=Json::Int64(independent?0:m);z["K"]=Json::Int64(K);z["sampled_N"]=Json::Int64(local.numel());z["expected_N"]=p.detach().sum().item<double>();z["sampled_photo"]=loss.item<double>();z["sampled_psnr"]=psnr;z["parameter_grad_norm"]=std::sqrt(gn);z["weighted_mean"]=after.mul(weights).sum().item<double>();z["cell_std"]=after.std(false).item<double>();z["max_abs_coefficient"]=after.abs().max().item<double>();z["normalized_gradient_sum"]=normalizedsum;z["cell_gradient_sum"]=lastgrad.sum().item<double>();z["cell_gradient_abs_mean"]=lastgrad.abs().mean().item<double>();emit(z);
    if(step%100==0||step==250||step==steps){scene.verify();TORCH_CHECK(cell.equal(mapcopy)&&normalized.equal(normcopy)&&occupancy.equal(occopy),"FROZEN_GRID_CHANGED");hardBank(step);}
    if(step==250||step==1000||step==steps)saveState(step);
  }
  scene.verify();Json::Value done;done["kind"]="completed";done["completed"]=true;done["frozen_exact"]=true;done["grid_frozen_exact"]=true;done["intra_cell_violations"]=0;done["steps"]=steps;emit(done);return 0;
}
