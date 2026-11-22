#include "include/gaussian_model.h"
#include "improvements/selector/gumbel_network.h"
#include <iostream>

int main(int argc,char** argv) {
  const auto device=argc>1&&std::string(argv[1])=="cuda"?torch::kCUDA:torch::kCPU;
  const auto opts=torch::TensorOptions().device(device).dtype(torch::kFloat32);
  auto selector=improvements::selector::GumbelNetwork(.1,true,false,true);selector->to(device);
  auto bank=selector->anchors();bank->enableVariableMembers();bank->enableSlotDeactivation();
  torch::optim::Adam feature_optimizer(selector->parameters(),torch::optim::AdamOptions(.001));
  GaussianModelParams parameters;parameters.sh_degree_=1;parameters.data_device_=device==torch::kCUDA?"cuda":"cpu";
  GaussianModel map(parameters);map.enableExplicitAnchors(bank,&feature_optimizer);map.setAnchorTopologyPolicy(true,false);
  std::map<point3D_id_t,Point3D> observed;int index=0;
  for(int cell=0;cell<3;++cell)for(int k=0;k<std::vector<int>{1,3,12}[cell];++k) {
    Point3D p;p.xyz_=Eigen::Vector3d(.01+.2*cell+.001*k,.01,2.01);p.color_=Eigen::Vector3f(.4,.4,.4);observed[index++]=p;
  }
  map.createFromPcd(observed,1);GaussianOptimizationParams optimization;map.trainingSetup(optimization);
  bank->assertBlocks(map.xyz_.size(0));
  const auto n=map.xyz_.size(0);auto original_rows=bank->rows().clone(),uids=bank->blockUIDs().clone();
  // Both old opacity and screen-size pruning predicates fail, but no point may be quality-pruned.
  {torch::NoGradGuard guard;map.opacity_.fill_(-20);map.max_radii2D_.fill_(100);map.xyz_gradient_accum_.zero_();map.denom_.fill_(1);
    map.densifyAndPrune(1e6,.02,1,1,5);}
  TORCH_CHECK(map.xyz_.size(0)==n&&torch::equal(bank->rows(),original_rows)&&torch::equal(bank->blockUIDs(),uids)&&
      map.gaussianPrunedMembers()==0,"Anchor-only mode performed per-Gaussian opacity/size pruning");
  auto dead=torch::nonzero(bank->memberCounts()==10).reshape({-1})[0].item<int64_t>();
  auto dead_rows=torch::nonzero(bank->rows()==dead).reshape({-1});
  {torch::NoGradGuard guard;map.opacity_.fill_(std::log(.1/.9));map.opacity_.index_fill_(0,dead_rows,-20);}
  auto all=torch::arange(n,opts.dtype(torch::kLong));for(int i=0;i<5;++i)map.addAnchorPruningStats(all);
  {torch::NoGradGuard guard;map.densifyAndPrune(1e6,.02,1,0,5);}
  auto surviving=torch::arange(uids.numel(),uids.options())!=dead;
  TORCH_CHECK(map.xyz_.size(0)==n-10&&torch::equal(bank->blockUIDs(),uids.index({surviving}))&&
      map.anchorPrunedMembers()==10&&map.gaussianPrunedMembers()==0,"Original whole-Anchor window pruning failed");
  // Gaussian densification is retained, including split-parent replacement.
  auto before=map.xyz_.size(0);
  {torch::NoGradGuard guard;map.scaling_.fill_(-7);map.xyz_gradient_accum_.fill_(100);map.denom_.fill_(1);
    map.densifyAndPrune(1,.02,1,0,5);}
  TORCH_CHECK(map.xyz_.size(0)>before&&map.gaussianPrunedMembers()==0,"Anchor-only mode disabled clone filling");
  {torch::NoGradGuard guard;bank->topologyRound().fill_(4);map.scaling_.fill_(-2);map.xyz_gradient_accum_.fill_(100);map.denom_.fill_(1);}
  before=map.xyz_.size(0);auto anchor_ids=bank->blockUIDs().clone();
  {torch::NoGradGuard guard;map.densifyAndPrune(1,.02,1,0,5);}
  TORCH_CHECK(map.xyz_.size(0)>before&&torch::equal(anchor_ids,bank->blockUIDs()),"Split net growth or ownership changed");
  bank->assertBlocks(map.xyz_.size(0));

  for(bool require_full:{false,true}) {
    auto growth=improvements::selector::GumbelNetwork(.1,true,false,true);growth->to(device);
    auto b=growth->anchors();b->enableVariableMembers();
    auto initial=torch::tensor({{.01f,.01f,.01f}},opts);
    auto seeds=b->observeBlocks(initial,torch::full_like(initial,.4));auto xyz=seeds.xyz,color=seeds.color;
    auto grow=[&](double value) {
      auto grad=torch::zeros({xyz.size(0),1},opts);grad[0]=value;
      auto next=b->growBlocks(xyz,color,grad,torch::ones_like(grad),1,2,{},require_full);
      xyz=torch::cat({xyz,next.xyz});color=torch::cat({color,next.color});return next.xyz.size(0);
    };
    xyz[0][0]=.411;
    TORCH_CHECK(grow(5)==(require_full?0:1),"Underfull fine growing gate incorrect");
    xyz[0][0]=.611;auto anchor_count=b->size();
    TORCH_CHECK(grow(1.5)==1&&b->size()==anchor_count+1&&b->levels()[-1].item<int64_t>()==0,
                "Full gate must NOT restrict underfull same-level growing");
    auto parents=torch::zeros({9},b->rows().options());b->appendVariableMembers(parents);
    xyz=torch::cat({xyz,xyz.slice(0,0,1).repeat({9,1})});color=torch::cat({color,color.slice(0,0,1).repeat({9,1})});
    TORCH_CHECK(b->memberCounts()[0].item<int64_t>()==10,"Full source fixture failed");
    xyz[0][0]=.811;
    TORCH_CHECK(grow(5)==1&&b->levels()[-1].item<int64_t>()==1,"Full Anchor failed to grow into next level");
    b->assertBlocks(xyz.size(0));
  }
  std::cout<<"PASS: no Gaussian opacity/size prune; original whole-Anchor prune; clone/split retained; partial same-level growth retained; fine growth only blocked below ten; full source grows\n";
}
