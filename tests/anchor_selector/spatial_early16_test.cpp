#include "improvements/selector/gumbel_network.h"
#include <iostream>
#include <map>

using improvements::selector::GumbelNetwork;
int main(int argc,char** argv) {
  torch::manual_seed(2026);
  const torch::Device device=argc>1&&std::string(argv[1])=="cpu"?torch::kCPU:torch::kCUDA;
  const bool direct_ratio=argc>3&&std::string(argv[3])=="direct_ratio";
  const int64_t bands=argc>4?std::stoll(argv[4]):6;
  GumbelNetwork net(.1,true,false,true,std::vector<double>{},0);
  net->to(device);net->anchors()->enableVariableMembers();net->enableSpatialCompensation(true,direct_ratio,bands);
  torch::optim::Adam optimizer(net->parameters(),torch::optim::AdamOptions(.001));
  auto bank=net->anchors();auto options=torch::TensorOptions().device(device);
  auto input=torch::tensor({{.01f,.02f,.03f},{.02f,.03f,.04f},{.21f,.02f,.03f}},options);
  auto seeds=bank->observeBlocks(input,torch::full_like(input,.4),{&optimizer});
  auto n=seeds.xyz.size(0),m=bank->size();
  auto xyz=seeds.xyz,scale=seeds.sigma.log(),rotation=torch::zeros({n,4},options);
  rotation.select(1,0).fill_(1);auto dc=seeds.color.unsqueeze(1),opacity=torch::zeros({n,1},options);
  auto owner=bank->rows(),center=bank->centers().index_select(0,owner),reference=bank->offsetScales().index_select(0,owner);
  auto spatial=net->spatialCompensation();spatial->setFused(false);
  TORCH_CHECK(spatial->directRatioInput()==direct_ratio&&spatial->encode->weight.size(1)==(direct_ratio?56:55),"Ratio input layout mismatch");
  TORCH_CHECK(static_cast<bool>(spatial->ratio_net)!=direct_ratio,"FiLM module layout mismatch");
  spatial->initialize(xyz,scale,rotation,dc,opacity,center,reference,true);
  auto packed=spatial->input55(xyz,scale,rotation,dc,opacity,center,reference,true);
  for(int64_t axis=0;axis<3;++axis)for(int64_t k=bands;k<6;++k)
    TORCH_CHECK(packed.select(1,19+axis*6+k).count_nonzero().item<int64_t>()==0&&packed.select(1,37+axis*6+k).count_nonzero().item<int64_t>()==0,"Disabled frequency is nonzero");
  {torch::NoGradGuard guard;for(auto& p:spatial->heads->parameters())p.fill_(.01);}
  auto run=[&](const torch::Tensor& feature) {return spatial->forward(xyz,scale,rotation,dc,opacity,center,reference,.05,true,feature);};
  TORCH_CHECK(bank->features().size(1)==0&&bank->spatialFeaturesTable().sizes()==torch::IntArrayRef({m,16}),"Feature leaked into Selector layout");
  auto feature=bank->spatialFeaturesTable().index_select(0,owner);
  auto output=run(feature);
  // Same original weights, explicit zero input: function-preserving insertion.
  auto zero=run(torch::zeros_like(feature));TORCH_CHECK(torch::equal(output,zero),"Initial output changed");
  auto selection=net->forwardProbability(xyz,{},seeds.sigma,.05,1,{},dc,opacity.sigmoid()).detach();
  (output-.03).square().mean().backward();
  TORCH_CHECK(bank->spatialFeaturesTable().grad().defined()&&bank->spatialFeaturesTable().grad().abs().sum().item<float>()>0,"No first Feature gradient");
  {torch::NoGradGuard guard;bank->spatialFeaturesTable().add_(.1);}
  auto unchanged=net->forwardProbability(xyz,{},seeds.sigma,.05,1,{},dc,opacity.sigmoid()).detach();
  TORCH_CHECK(torch::equal(selection,unchanged),"Feature affects Selector");
  optimizer.step();auto before=bank->spatialFeaturesTable().detach().clone();auto uids=bank->blockUIDs().cpu().clone();
  auto& state=static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(bank->spatialFeaturesTable().unsafeGetTensorImpl()));
  auto moment=state.exp_avg().clone();
  bank->observeBlocks(torch::tensor({{.61f,.02f,.03f}},options),torch::full({1,3},.5,options),{&optimizer});
  TORCH_CHECK(torch::equal(bank->spatialFeaturesTable().slice(0,0,m),before)&&bank->spatialFeaturesTable().slice(0,m).count_nonzero().item<int64_t>()==0,"Append state incorrect");
  auto keep=torch::nonzero(bank->rows()!=0).reshape({-1});bank->pruneVariableMembers(keep,{&optimizer});
  auto alive=bank->blockUIDs().cpu();
  auto& migrated=static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(bank->spatialFeaturesTable().unsafeGetTensorImpl()));
  for(int64_t row=0;row<alive.numel();++row)for(int64_t old=0;old<uids.numel();++old)if(alive[row].item<int64_t>()==uids[old].item<int64_t>()) {
    TORCH_CHECK(torch::equal(bank->spatialFeaturesTable()[row],before[old])&&torch::equal(migrated.exp_avg()[row],moment[old]),"UID/Adam mismatch");
  }
  optimizer.zero_grad(true);bank->spatialFeaturesTable().square().sum().backward();optimizer.step();
  auto frozen=net->cloneFrozen();TORCH_CHECK(torch::equal(frozen->anchors()->spatialFeaturesTable(),bank->spatialFeaturesTable()),"Teacher copy mismatch");
  const std::string path=argc>2?argv[2]:"early16_native.pt";torch::save(net,path);
  GumbelNetwork loaded(.1,true,false,true,std::vector<double>{},0);loaded->to(device);
  torch::serialize::InputArchive archive;archive.load_from(path,device);loaded->load(archive);
  TORCH_CHECK(loaded->anchors()->spatialFeatures()&&torch::equal(loaded->anchors()->spatialFeaturesTable(),bank->spatialFeaturesTable()),"Checkpoint mismatch");
  TORCH_CHECK(loaded->spatialCompensation()->directRatioInput()==direct_ratio,"Loaded ratio layout mismatch");
  TORCH_CHECK(loaded->spatialCompensation()->fourierBands()==bands&&frozen->spatialCompensation()->fourierBands()==bands,"Fourier checkpoint/teacher mismatch");
  std::cout<<"EARLY16_PASS selected_gradient selector_isolation append prune adam_next_step teacher checkpoint\n";
}
