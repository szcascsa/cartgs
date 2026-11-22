#include "improvements/selector/anchor_bank.h"
#include "include/anchor_stats_cuda.h"
#include <iostream>

using improvements::selector::AnchorBank;
using Key=std::array<int64_t,4>;

struct Fixture {
  AnchorBank bank{.1,16};
  torch::Tensor xyz;
  torch::optim::Adam optimizer;
  Fixture(int mode,torch::Device device,const std::vector<Key>& cells)
      : optimizer(bank->parameters(),torch::optim::AdamOptions(.001)) {
    bank->to(device);bank->enableExplicitBlocks();bank->enableVariableMembers();bank->enableFixedTenMembers(mode);
    optimizer.param_groups()[0].params()[0]=bank->features();
    std::map<Key,std::vector<int64_t>> groups;std::vector<double> points;
    for(const auto& key:cells) {
      groups[key].push_back(points.size()/3);
      const double h=.1/double(int64_t(1)<<key[0]);
      for(int a=0;a<3;++a)points.push_back((key[a+1]+.3)*h);
    }
    auto observed=torch::tensor(points,torch::kFloat64).reshape({-1,3}).to(bank->features().options().requires_grad(false));
    auto rng=at::globalContext().defaultGenerator(device).get_state().clone();
    auto seeds=bank->insertVariableBlocks(groups,observed,torch::full_like(observed,.4),false,{&optimizer});xyz=seeds.xyz;
    TORCH_CHECK(torch::equal(rng,at::globalContext().defaultGenerator(device).get_state()),"Padding consumed training RNG");
    TORCH_CHECK(xyz.size(0)==10*int64_t(cells.size())&&(bank->memberCounts()==10).all().item<bool>(),"Initial groups were not padded to ten");
    for(int64_t j=0;j<observed.size(0);++j)TORCH_CHECK(torch::equal(xyz[10*j],observed[j]),"Real observation was moved");
    TORCH_CHECK(torch::isfinite(xyz).all().item<bool>()&&torch::isfinite(seeds.sigma).all().item<bool>(),"Nonfinite padding");
    if(device.is_cuda()) {
      auto members=mapping_cuda::validateAnchors(bank->rows(),bank->memberCounts(),bank->levels(),bank->centers(),bank->offsetScales());
      auto options=bank->rows().options();auto alpha=torch::full({xyz.size(0),1},.1f,xyz.options());
      std::vector<torch::Tensor> stats={torch::zeros({bank->size()},options.dtype(torch::kFloat64)),torch::zeros({bank->size()},options),
          torch::zeros({xyz.size(0)},options.dtype(torch::kFloat64)),torch::zeros({xyz.size(0)},options),torch::zeros({xyz.size(0)},options)};
      mapping_cuda::anchorFullStats(bank->rows(),members,bank->memberCounts(),torch::arange(xyz.size(0),options),alpha,stats,
          bank->inactiveSlots(),bank->recoveredSlots());
      TORCH_CHECK((stats[1]==1).all().item<bool>()&&torch::equal(stats[0],torch::full_like(stats[0],double(.1f)*10)),
          "CUDA fixed-ten metadata/Full statistics do not match the ten stored members");
    }
    bank->features().sum().backward();optimizer.step();optimizer.zero_grad(true);
    torch::NoGradGuard guard;
    for(int64_t j=0;j<bank->size();++j)bank->features()[j].fill_(j+2);
  }
  torch::Tensor rows(int64_t owner) {return torch::nonzero(bank->rows()==owner).reshape({-1});}
  torch::Tensor moment() {
    return static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(bank->features().unsafeGetTensorImpl())).exp_avg();
  }
  void shift(int64_t owner,double x) {
    auto ids=rows(owner);xyz.index_put_({ids,0},xyz.index({ids,0})+x);
    GaussianTransformResult correction;correction.indices=ids;correction.population_count=xyz.size(0);
    correction.transform_4x4=torch::eye(4,xyz.options());correction.transform_4x4[3][0]=x;
    correction.changed_attributes=toBits(GaussianAttrFlag::XYZ);
    bank->transportBlocks(correction);
  }
  auto finish() {
    bank->finishBlockGeometry();auto result=bank->takeFixedMigration();
    auto keep=~result.remove;bank->pruneVariableMembers(keep,{&optimizer});xyz=xyz.index({keep});
    bank->assertBlocks(xyz.size(0));return result;
  }
};

int main(int argc,char** argv) {
  const torch::Device device=argc>1&&std::string(argv[1])=="cuda"?torch::kCUDA:torch::kCPU;
  for(int mode:{3,4}) {
    Fixture f(mode,device,{{0,0,0,0},{0,2,0,0}});
    TORCH_CHECK(f.bank->slotDeactivation()==(mode==4),"Mode lost slot policy");
    auto features=f.bank->features().detach().clone(),moments=f.moment().clone(),uids=f.bank->blockUIDs().clone();
    f.bank->beginBlockGeometry();f.shift(0,.2);auto result=f.finish();
    TORCH_CHECK(result.remove.sum().item<int64_t>()==10&&f.bank->size()==1&&f.bank->blockUIDs()[0].item<int64_t>()==uids[1].item<int64_t>(),
        "Occupied destination did not keep its stationary Anchor");
    TORCH_CHECK(torch::equal(f.bank->features()[0],features[1])&&torch::equal(f.moment()[0],moments[1]),"Collision changed surviving Feature/Adam");
    auto original=f.xyz.clone();auto mask=torch::ones({f.xyz.size(0)},f.bank->rows().options().dtype(torch::kBool));mask[0]=false;
    bool rejected=false;try{f.bank->pruneVariableMembers(mask,{&f.optimizer});}catch(const c10::Error&){rejected=true;}
    TORCH_CHECK(rejected&&f.bank->rows().numel()==original.size(0),"Partial deletion was accepted or mutated ownership");

    Fixture swap(mode,device,{{0,0,0,0},{0,2,0,0}});
    features=swap.bank->features().detach().clone();moments=swap.moment().clone();uids=swap.bank->blockUIDs().clone();
    swap.bank->beginBlockGeometry();swap.shift(1,-.2);swap.shift(0,.2);result=swap.finish();
    TORCH_CHECK(!result.remove.any().item<bool>()&&result.moved==20&&torch::equal(uids,swap.bank->blockUIDs()),"Swap incorrectly treated departed cells as occupied");
    TORCH_CHECK(torch::equal(features,swap.bank->features())&&torch::equal(moments,swap.moment()),"Swap reinitialized trained state");

    Fixture collision(mode,device,{{0,0,0,0},{0,2,0,0}});
    collision.bank->beginBlockGeometry();collision.shift(1,.2);collision.shift(0,.4);result=collision.finish();
    TORCH_CHECK(collision.bank->size()==1&&collision.bank->blockUIDs()[0].item<int64_t>()==0,"Arrival tie did not retain lowest UID");
    Fixture reverse(mode,device,{{0,0,0,0},{0,2,0,0}});
    reverse.bank->beginBlockGeometry();reverse.shift(0,.4);reverse.shift(1,.2);reverse.finish();
    TORCH_CHECK(torch::equal(collision.xyz,reverse.xyz),"Collision depends on correction order");

    Fixture levels(mode,device,{{0,0,0,0},{1,0,0,0}});
    levels.bank->beginBlockGeometry();levels.shift(0,.01);result=levels.finish();
    TORCH_CHECK(levels.bank->size()==2&&!result.remove.any().item<bool>(),"Cross-level Anchors collided");

    torch::serialize::OutputArchive output;levels.bank->save(output);
    const auto path=std::string("fixed_ten_bank_")+std::to_string(mode)+".pt";output.save_to(path);
    AnchorBank loaded(.1,16);loaded->to(device);torch::serialize::InputArchive input;input.load_from(path,device);loaded->load(input);
    TORCH_CHECK(loaded->fixedTenMode()==mode&&torch::equal(loaded->rows(),levels.bank->rows()),"Fixed-ten checkpoint lost mode/ownership");
    AnchorBank teacher(.1,16);teacher->to(device);teacher->copyLayoutFrom(*levels.bank);
    TORCH_CHECK(teacher->fixedTenMode()==mode&&(teacher->memberCounts()==10).all().item<bool>(),"Frozen layout lost fixed-ten invariant");
  }
  // Legacy checkpoints remain readable normally but cannot silently switch into fixed-ten training.
  AnchorBank old(.1,16);old->to(device);old->enableExplicitBlocks();old->enableVariableMembers();
  auto point=torch::tensor({{.01f,.01f,.01f}},old->features().options());old->observeBlocks(point,torch::full_like(point,.4));
  torch::serialize::OutputArchive legacy;old->save(legacy);legacy.save_to("fixed_ten_legacy.pt");
  AnchorBank fixed(.1,16);fixed->to(device);fixed->enableExplicitBlocks();fixed->enableVariableMembers();fixed->enableFixedTenMembers(3);
  torch::serialize::InputArchive input;input.load_from("fixed_ten_legacy.pt",device);
  bool rejected=false;try{fixed->load(input);}catch(const c10::Error&){rejected=true;}
  TORCH_CHECK(rejected,"Legacy partial checkpoint silently converted to fixed-ten");
  std::cout<<"PASS fixed-ten Anchors: deterministic padding, real seeds, resident/UID collision, swaps, levels, Feature/Adam, checkpoint, legacy rejection\n";
}
