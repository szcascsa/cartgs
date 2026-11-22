#include "improvements/selector/gumbel_network.h"
#ifdef CARTGS_FUSED_MAPPING_CUDA
#include "include/mapping_cuda_optimizer.h"
#endif
#include <iostream>

using namespace improvements::selector;
void check(bool condition,const char* message) { TORCH_CHECK(condition,message); }

int main(int argc,char** argv) {
 try {
  const auto device=torch::Device(argc>1?argv[1]:"cpu");
  auto denseStep=[&](torch::optim::Adam& optimizer) {
#ifdef CARTGS_FUSED_MAPPING_CUDA
    if(argc>2 && std::string(argv[2])=="fused") {
      TORCH_CHECK(mapping_cuda::tryFusedDenseAdam(optimizer),"Fused lifecycle layout unsupported");return;
    }
#endif
    optimizer.step();
  };
  auto opts=torch::TensorOptions().device(device).dtype(torch::kFloat);
  torch::manual_seed(2026);
  for(bool independent:{true,false}) {
    auto model=GumbelNetwork(.1,true,false,true);model->to(device);
    model->setFusedInputPack(argc>2);
    model->anchors()->enableVariableMembers();model->enableCompensation();
    model->enablePrivateCompensation(independent,{.01,.05,.1,.15});
    auto xyz=torch::tensor({{.01f,.01f,.01f},{.03f,.01f,.01f},{.21f,.01f,.01f}},opts);
    auto color=torch::full_like(xyz,.4);auto seeds=model->anchors()->observeBlocks(xyz,color,{});
    xyz=seeds.xyz;auto bank=model->privateCompensation();bank->append(xyz.size(0),{},{});
    const auto population=xyz.size(0);
    auto ids=torch::arange(xyz.size(0),opts.dtype(torch::kLong));
    auto scale=torch::full_like(xyz,.01),dc=torch::full_like(xyz,.2),alpha=torch::full({xyz.size(0),1},.4,opts);
    auto forward=model->forwardSelectionFeatures(xyz,.05,1,{},scale,dc,alpha);
    const auto original_probability=forward.probability.detach().clone();
    auto residual=model->compensationResidual(forward.shared,ids,.05);
    check(torch::equal(residual,torch::zeros_like(residual)),"Step0 must be exactly zero");
    torch::optim::Adam optimizer(model->parameters(),torch::optim::AdamOptions(.001));
    for(int step=0;step<3;++step) {
      optimizer.zero_grad(true);
      forward=model->forwardSelectionFeatures(xyz,.05,1,{},scale,dc,alpha);
      residual=model->compensationResidual(forward.shared,ids,.05);
      (residual-.1).square().mean().backward();
      check(!model->named_parameters()["local_emd.0.weight"].grad().defined(),"Private photo must not update Encoder");
      for(auto p:model->named_parameters())if(p.key().find("compensation.")==0)
        check(!p.value().requires_grad()&&!p.value().grad().defined(),"C must remain frozen");
      for(size_t i=0;i<bank->tables().size();++i)
        check(bank->tables()[i].grad().defined()==(!independent||i==1),"Inactive ratio table acquired gradient");
      denseStep(optimizer);
    }
    check(torch::equal(original_probability,model->forwardSelectionFeatures(xyz,.05,1,{},scale,dc,alpha).probability),
          "Private-only training changed Selector scores");
    auto table=bank->tables()[independent?1:0];
    {torch::NoGradGuard guard;table.add_(torch::arange(population,opts).unsqueeze(1)*.01);}
    auto old=table.detach().clone();auto uids=bank->uids().clone();
    auto moment=static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(table.unsafeGetTensorImpl())).exp_avg().clone();
    bank->append(1,ids.slice(0,0,1),{&optimizer});
    table=bank->tables()[independent?1:0];
    check(torch::equal(table[population],old[0]),"Clone/split must inherit residual");
    auto& state=static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(table.unsafeGetTensorImpl()));
    check(torch::equal(state.exp_avg().slice(0,0,population),moment)&&state.exp_avg()[population].abs().sum().item<float>()==0,"Append lost Adam state");
    check(bank->uids()[population].item<int64_t>()!=uids[0].item<int64_t>(),"Child must receive fresh Gaussian identity");
    auto tail=torch::arange(population-1,0,-1,ids.options());
    auto order=torch::cat({torch::full({1},population,ids.options()),tail});bank->select(order,{&optimizer});
    check(torch::equal(bank->tables()[independent?1:0],old.index_select(0,torch::cat({torch::zeros({1},ids.options()),tail}))),"Prune/reorder misaligned residual");
    check((bank->uids().slice(0,1)!=bank->uids()[0]).all().item<bool>(),"UIDs lost in compaction");
    optimizer.zero_grad(true);
    bank->forward(ids,.05).square().sum().backward();denseStep(optimizer);
    check(bank->tables()[independent?1:0].sizes()==torch::IntArrayRef({population,14}),"Resized backward leaf invalid");
    {
      torch::NoGradGuard guard;
      for(size_t j=0;j<bank->tables().size();++j)bank->tables()[j].fill_(double(j+1)*.01);
    }
    if(independent) {
      check(torch::allclose(bank->forward(ids,.03),torch::full({population,14},.015,opts)),"Linear interpolation mismatch");
      check(torch::equal(bank->forward(ids,1.),bank->forward(ids,.15)),"Outside range must use endpoint");
    }
    auto before=bank->forward(ids,.05).detach().clone();
    GaussianTransformResult correction;correction.affects_all=true;correction.scale=2;
    correction.transform_4x4=torch::eye(4,opts);correction.rotation_wxyz=torch::tensor({1.f,0.f,0.f,0.f},opts);
    correction.changed_attributes=toBits(GaussianAttrFlag::XYZ)|toBits(GaussianAttrFlag::ROTATION);
    bank->transport(correction,{&optimizer});
    check(torch::allclose(bank->forward(ids,.05).slice(1,0,3),before.slice(1,0,3)*2),"Geometry residual transport mismatch");
    before=bank->forward(ids,.05).detach().clone();
    correction.scale=1;
    correction.transform_4x4=torch::tensor({{0.f,1.f,0.f,0.f},{-1.f,0.f,0.f,0.f},{0.f,0.f,1.f,0.f},{0.f,0.f,0.f,1.f}},opts);
    correction.rotation_wxyz=torch::tensor({.70710678f,0.f,0.f,.70710678f},opts);
    bank->transport(correction,{&optimizer});
    auto transported=bank->forward(ids,.05);
    check(torch::allclose(transported.slice(1,0,3),torch::matmul(before.slice(1,0,3),correction.transform_4x4.slice(0,0,3).slice(1,0,3))),
          "Rotated XYZ compensation mismatch");
    auto identity=torch::zeros({population,4},opts);identity.select(1,0).fill_(1);
    auto world_q=correction.rotation_wxyz.reshape({1,4}).expand_as(identity);
    auto old_q=quaternion_utils::normalize(identity+before.slice(1,6,10));
    auto new_q=quaternion_utils::normalize(identity+transported.slice(1,6,10));
    check(torch::allclose(quaternion_utils::multiply(new_q,world_q),quaternion_utils::multiply(world_q,old_q),1e-5,1e-6),
          "Noncommuting quaternion transport changed compensated world rotation");
    auto teacher=GumbelNetwork(model->cloneFrozen());
    auto expected=model->compensationResidual(model->forwardSelectionFeatures(xyz,.075,1,{},scale,dc,alpha).shared,ids,.075);
    check(torch::equal(expected,teacher->compensationResidual(teacher->forwardSelectionFeatures(xyz,.075,1,{},scale,dc,alpha).shared,ids,.075)),"Teacher mismatch");
    const std::string prefix=independent?"private_r3":"private_r2";
    torch::serialize::OutputArchive output;model->save(output);output.save_to(prefix+".pt");
    auto loaded=GumbelNetwork(.1,true,false,true);loaded->to(device);
    torch::serialize::InputArchive input;input.load_from(prefix+".pt",device);loaded->load(input);
    auto actual=loaded->compensationResidual(loaded->forwardSelectionFeatures(xyz,.075,1,{},scale,dc,alpha).shared,ids,.075);
    check(torch::equal(expected,actual),"Checkpoint roundtrip mismatch");
    torch::serialize::OutputArchive fixture;
    fixture.write("xyz",xyz);fixture.write("scale",scale);fixture.write("dc",dc);fixture.write("alpha",alpha);
    fixture.write("expected",expected);fixture.write("probability",model->forwardSelectionFeatures(xyz,.075,1,{},scale,dc,alpha).probability);
    fixture.save_to(prefix+"_fixture.pt");
  }
  std::cout<<"PASS private R2/R3 zero init, ratio isolation, gradient routing, append/prune/reorder/Adam, transport, teacher, checkpoint\n";
 } catch(const std::exception& error) { std::cerr<<error.what()<<std::endl;return 1; }
}
