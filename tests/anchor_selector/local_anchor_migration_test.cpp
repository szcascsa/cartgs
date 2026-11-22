#include "improvements/selector/anchor_bank.h"
#include <iostream>

using improvements::selector::AnchorBank;
using Key = std::array<int64_t,4>;
int64_t feature_width=16;

struct Fixture {
  AnchorBank bank{.1,feature_width};
  torch::Tensor xyz;
  torch::optim::Adam first,second;
  Fixture(const std::vector<std::pair<Key,int64_t>>& cells,torch::Device device)
      : first(bank->parameters(),torch::optim::AdamOptions(.001)),
        second(bank->parameters(),torch::optim::AdamOptions(.002)) {
    bank->to(device);bank->enableExplicitBlocks();bank->enableVariableMembers();bank->enableSlotDeactivation();
    // Optimizer aliases must follow Feature replacements on either device.
    first.param_groups()[0].params()[0]=bank->features();second.param_groups()[0].params()[0]=bank->features();
    std::map<Key,std::vector<int64_t>> groups;std::vector<double> positions;
    for(const auto& entry:cells) {
      const auto& key=entry.first;const double h=.1/double(int64_t(1)<<key[0]);
      for(int64_t i=0;i<entry.second;++i) {
        groups[key].push_back(positions.size()/3);
        for(int axis=0;axis<3;++axis)positions.push_back((key[axis+1]+.2)*h+(axis==0?.001*i:0));
      }
    }
    auto observed=torch::tensor(positions,torch::kFloat64).reshape({-1,3}).to(bank->features().options().requires_grad(false));
    xyz=bank->insertVariableBlocks(groups,observed,torch::full_like(observed,.4),false,{&first,&second}).xyz;
    bank->features().sum().backward();first.step();second.step();first.zero_grad(true);second.zero_grad(true);
    torch::NoGradGuard guard;
    for(int64_t j=0;j<bank->size();++j)bank->features()[j].fill_(j==0?2:j==1?8:12);
  }
  torch::Tensor rows(int64_t owner) {return torch::nonzero(bank->rows()==owner).reshape({-1});}
  torch::Tensor moment(torch::optim::Adam& optimizer) {
    return static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(bank->features().unsafeGetTensorImpl())).exp_avg();
  }
  auto migrate(const torch::Tensor& indices) {
    auto result=bank->migrateLocalMembers(xyz,indices,{&first,&second});
    bank->pruneVariableMembers(~result.remove,{&first,&second});xyz=xyz.index({~result.remove});
    bank->assertBlocks(xyz.size(0));return result;
  }
};

int main(int argc,char** argv) {
  const torch::Device device=argc>1&&std::string(argv[1])=="cuda"?torch::kCUDA:torch::kCPU;
  feature_width=argc>2&&std::string(argv[2])=="feature32"?32:16;
  for(int target_count:{5,8,10}) {
    Fixture f({{{0,0,0,0},5},{{0,2,0,0},target_count},{{0,4,0,0},2}},device);
    const auto moving=f.rows(0);const auto untouched=f.bank->features()[2].clone();
    const auto first_moment=f.moment(f.first)[2].clone(),second_moment=f.moment(f.second)[2].clone();
    const auto second_order=static_cast<torch::optim::AdamParamState&>(
        *f.first.state().at(f.bank->features().unsafeGetTensorImpl())).exp_avg_sq()[2].clone();
    f.bank->inactiveSlots().fill_(true);
    const auto uids=f.bank->blockUIDs().clone();
    f.xyz.index_put_({moving,0},f.xyz.index({moving,0})+.2);
    auto result=f.migrate(moving.flip({0}));
    const int64_t accepted=std::min<int64_t>(5,10-target_count);
    TORCH_CHECK(result.moved==accepted&&result.created==0&&result.remove.sum().item<int64_t>()==5-accepted,
                "Destination capacity or deterministic rejection failed");
    for(int64_t i=0;i<5;++i)TORCH_CHECK(result.remove[moving[i]].item<bool>()==(i>=accepted),"Admission depends on correction order");
    const auto expected=(8.*target_count+2.*accepted)/(target_count+accepted);
    TORCH_CHECK(torch::allclose(f.bank->features()[0],torch::full_like(f.bank->features()[0],expected)),
                "Feature weights include rejected members or omit resident population");
    TORCH_CHECK(torch::equal(f.bank->blockUIDs(),uids.slice(0,1))&&torch::equal(f.bank->features()[1],untouched),
                "Empty source cleanup or unaffected Feature failed");
    TORCH_CHECK(torch::equal(f.moment(f.first)[1],first_moment)&&torch::equal(f.moment(f.second)[1],second_moment),
                "Unaffected Adam history changed");
    const auto& state=static_cast<torch::optim::AdamParamState&>(*f.first.state().at(f.bank->features().unsafeGetTensorImpl()));
    TORCH_CHECK(state.step()==1&&torch::equal(state.exp_avg_sq()[1],second_order),"Adam step or second moment changed during compaction");
    TORCH_CHECK(f.bank->inactiveSlots().index({f.bank->rows()==1}).all().item<bool>(),"Unchanged Anchor lost slot history");
    if(accepted) {
      TORCH_CHECK(f.moment(f.first)[0].count_nonzero().item<int64_t>()==0&&
                 f.moment(f.second)[0].count_nonzero().item<int64_t>()==0&&state.exp_avg_sq()[0].count_nonzero().item<int64_t>()==0,
                 "Mixed Feature retained stale Adam history");
      TORCH_CHECK(!f.bank->inactiveSlots().index({f.bank->rows()==0}).any().item<bool>(),"Mixed membership retained slot history with an old denominator");
    } else {
      TORCH_CHECK(f.moment(f.first)[0].abs().sum().item<float>()>0,"Full destination lost its untouched Adam history");
    }
  }
  {
    Fixture f({{{0,0,0,0},2},{{0,2,0,0},3},{{0,4,0,0},5}},device);
    {torch::NoGradGuard guard;f.bank->features()[1].fill_(10);f.bank->features()[2].fill_(8);}
    auto moving=torch::cat({f.rows(0),f.rows(1)});f.xyz.index_put_({moving,0},.41);
    auto result=f.migrate(moving);
    TORCH_CHECK(result.moved==5&&result.created==0&&f.bank->size()==1&&
                torch::allclose(f.bank->features(),torch::full_like(f.bank->features(),7.4)),
                "Existing destination did not combine all accepted sources against one frozen Feature snapshot");
  }
  {
    Fixture f({{{0,0,0,0},6},{{0,2,0,0},8}},device);
    {torch::NoGradGuard guard;f.bank->features()[1].fill_(12);}
    f.xyz.select(1,0).fill_(.61);
    auto result=f.migrate(torch::arange(f.xyz.size(0),f.bank->rows().options()));
    TORCH_CHECK(result.created==1&&result.moved==10&&result.remove.sum().item<int64_t>()==4&&f.bank->size()==1,
                "Missing destination created duplicate overflow Anchors");
    TORCH_CHECK(torch::allclose(f.bank->features(),torch::full_like(f.bank->features(),6)),
                "Multi-source new Feature must average only six 2s and four 12s");
    TORCH_CHECK(f.bank->blockUIDs()[0].item<int64_t>()==2&&f.bank->levels()[0].item<int64_t>()==0&&
                f.moment(f.first).count_nonzero().item<int64_t>()==0,"New UID/level/Adam initialization failed");
  }
  {
    Fixture f({{{0,0,0,0},10},{{0,2,0,0},10}},device);
    const auto original=f.bank->rows().clone(),uids=f.bank->blockUIDs().clone();
    f.xyz.index_put_({f.rows(0),0},f.xyz.index({f.rows(0),0})+.2);
    f.xyz.index_put_({f.rows(1),0},f.xyz.index({f.rows(1),0})-.2);
    auto result=f.migrate(torch::arange(f.xyz.size(0),original.options()));
    TORCH_CHECK(result.moved==20&&!result.remove.any().item<bool>()&&result.created==0&&
                torch::equal(f.bank->rows(),1-original)&&torch::equal(f.bank->blockUIDs(),uids),"Reciprocal full-cell moves falsely discarded points");
    TORCH_CHECK(torch::equal(f.bank->features()[0],torch::full_like(f.bank->features()[0],8))&&
                torch::equal(f.bank->features()[1],torch::full_like(f.bank->features()[1],2)),"Reciprocal Feature migration used updated sources");
  }
  {
    Fixture f({{{0,0,0,0},2},{{1,4,0,0},10}},device);
    const auto moving=f.rows(0);f.xyz.index_put_({moving,0},f.xyz.index({moving,0})+.2);
    auto result=f.migrate(moving);
    TORCH_CHECK(result.created==1&&result.moved==2&&!result.remove.any().item<bool>()&&
                (f.bank->levels()==0).sum().item<int64_t>()==1,"Other levels incorrectly consumed destination capacity");
  }
  {
    Fixture f({{{0,-1,0,0},3},{{0,0,0,0},2}},device);
    const auto moving=f.rows(1);f.xyz.index_put_({moving,0},-.005);
    auto result=f.migrate(torch::cat({moving,moving}));
    TORCH_CHECK(result.moved==2&&f.bank->size()==1&&f.bank->memberCounts()[0].item<int64_t>()==5,
                "Negative-coordinate floor or duplicate correction handling failed");
    const auto features=f.bank->features().clone(),moment=f.moment(f.first).clone();
    result=f.migrate(torch::arange(f.xyz.size(0),f.bank->rows().options()));
    TORCH_CHECK(result.moved==0&&result.created==0&&torch::equal(features,f.bank->features())&&
                torch::equal(moment,f.moment(f.first)),"Within-cell correction changed Feature or Adam");
  }
  std::cout<<"PASS: 5+5 weighted Feature, partial/full admission, multi-source overflow, reciprocal moves, level isolation, negative keys, stable UID, two Adam owners\n";
}
