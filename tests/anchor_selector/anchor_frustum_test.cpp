#include "include/anchor_frustum.h"
#include "include/reliability_maturity.h"
#include "include/selector_handoff.h"
#include <iostream>

int main(int argc,char** argv) {
  const bool cuda=argc>1 && std::string(argv[1])=="cuda";
  if(cuda && !torch::cuda::is_available())return 77;
  auto o=torch::TensorOptions().dtype(torch::kFloat32).device(cuda?torch::kCUDA:torch::kCPU);
  auto c=torch::tensor({{0.f,0.f,.5f},{1.4f,0.f,.5f},{1.6f,0.f,.5f},
      {0.f,0.f,-.6f},{0.f,0.f,1.6f},{0.f,0.f,-.1f},{1.5f,0.f,.5f}},o).set_requires_grad(true);
  auto h=torch::ones({7,1},o),p=torch::eye(4,o);
  auto visible=anchorNominalFrustumMask(c,h,p);
  TORCH_CHECK(visible.equal(torch::tensor({true,true,false,false,false,true,true},o.dtype(torch::kBool))),
      "Nominal cell intersection/boundary failure");
  TORCH_CHECK(!visible.requires_grad(),"Culling must detach geometry");
  TORCH_CHECK(anchorNominalFrustumMask(torch::empty({0,3},o),torch::empty({0,1},o),p).numel()==0,"Empty Anchors");
  // Compare the six-plane support test to all eight box corners, including a
  // large box enclosing the frustum and a camera with translation/perspective.
  torch::manual_seed(9);
  c=torch::randn({103,3},o);h=torch::rand({103,1},o)*2+.01;
  p=torch::tensor({{1.2f,0.f,0.f,0.f},{0.f,1.4f,0.f,0.f},
      {0.f,0.f,1.01f,1.f},{.3f,-.2f,-.1f,0.f}},o);
  auto maxima=torch::full({103,6},-INFINITY,o);
  for(float x:{-1.f,1.f})for(float y:{-1.f,1.f})for(float z:{-1.f,1.f}) {
    auto corner=c+(h*.5)*torch::tensor({x,y,z},o);
    auto clip=torch::cat({corner,torch::ones({103,1},o)},1).matmul(p);
    auto w=clip.select(1,3),cx=clip.select(1,0),cy=clip.select(1,1),cz=clip.select(1,2);
    maxima=torch::maximum(maxima,torch::stack({w+cx,w-cx,w+cy,w-cy,cz,w-cz},1));
  }
  TORCH_CHECK(anchorNominalFrustumMask(c,h,p).equal((maxima>=0).all(1)),"Six-plane support mismatch");
  // Membership, not each member's XYZ, defines the candidate pool. Reorder and
  // compact Anchor IDs exactly as prune/migration does, without a stale cache.
  auto centers=torch::tensor({{0.f,0.f,.5f},{3.f,0.f,.5f},{-.5f,0.f,.5f}},o);
  auto scales=torch::ones({3,1},o);
  auto owners=torch::tensor({2,0,1,2,1,0},o.dtype(torch::kLong));
  auto mask=anchorNominalFrustumMask(centers,scales,torch::eye(4,o)).index_select(0,owners);
  auto rows=torch::nonzero(mask).flatten();
  TORCH_CHECK(rows.equal(torch::tensor({0,1,3,5},owners.options())),"Global row ordering");
  auto order=torch::tensor({2,0,1},owners.options()),remap=torch::tensor({1,2,0},owners.options());
  auto moved=anchorNominalFrustumMask(centers.index_select(0,order),scales.index_select(0,order),torch::eye(4,o));
  TORCH_CHECK(moved.index_select(0,remap.index_select(0,owners)).equal(mask),"Anchor reorder identity");
  // Budget uses the restored N-row ST mask, never the compact C-row mean.
  for(float ratio:{.01f,.05f,.1f,.2f,.3f,.4f}) {
    auto weights=torch::full({rows.numel()},.2,o).set_requires_grad(true);
    auto hard=weights+(.0-weights.detach());
    auto full=torch::zeros({6},o).index_copy(0,rows,hard);
    auto loss=selectorGumbelRatioLoss(full,ratio,true);
    const double target=double(ratio)*6;
    TORCH_CHECK(std::abs(loss.item<double>()-target/std::max(1.,target))<1e-6,"Global-N budget");
    loss.backward();
    TORCH_CHECK(torch::allclose(weights.grad(),torch::full_like(weights,-1/std::max(1.,target))),"Budget VJP");
  }
  // Inactive/noncandidate rows cannot influence the relative maturity pool.
  auto m=torch::tensor({.9f,.7f,0.f,.8f,0.f,.6f},o).set_requires_grad(true);
  auto weights=torch::full({6},.2,o).set_requires_grad(true);
  auto mloss=reliabilitySelectionLoss(weights,m,mask,.05f);mloss.backward();
  TORCH_CHECK(!m.grad().defined() && weights.grad().index({~mask}).abs().sum().item<float>()==0,
      "Maturity detach/candidate gradient isolation");
  auto view=torch::randn({4,4},o).transpose(0,1),detached=view.detach();
  TORCH_CHECK(view.is_view() && !detached.is_view() && view.data_ptr()==detached.data_ptr() &&
      view.equal(detached),"Private VJP wrapper detach must preserve gradient storage/values");
  std::cout<<"PASS anchor nominal frustum: six planes, empty, membership/reorder, global-N budget/VJP, maturity isolation "
      <<(cuda?"CUDA":"CPU")<<'\n';
}
