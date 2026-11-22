#include "include/gaussian_mapper.h"
#include <stdexcept>
#include <filesystem>

void check(bool value,const char* message) { if(!value)throw std::runtime_error(message); }

class EntryMapper : public GaussianMapper {
 public:
  EntryMapper(torch::DeviceType device,const std::filesystem::path& output)
      : GaussianMapper(nullptr,std::filesystem::path(CARTGS_TEST_SOURCE_DIR)/"cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml",
          output,0,device,true,std::filesystem::path(CARTGS_TEST_SOURCE_DIR)/"cfg/selector/fast.yaml") {}
  void verify(torch::DeviceType device) {
    const auto opts=torch::TensorOptions().dtype(torch::kFloat32).device(device);
    transform_field_=nullptr; transform_optimizer_.reset();
    transform_field_config_.start_mode="maturity_population";
    transform_field_config_.elastic_entry_maturity=.3;
    transform_field_config_.ready_maturity=.3;
    transform_field_config_.min_ready_gaussians=500;
    transform_field_config_.hexplane.spatial_resolution=4;
    transform_field_config_.hexplane.ratio_resolution=4;
    transform_field_config_.hexplane.feature_dim=2;
    transform_field_config_.hexplane.multires={1}; transform_field_config_.hidden_dim=8;
    drift_stability_config_.enabled=false;
    gaussians_->xyz_=torch::zeros({501,3},opts);
    gaussians_->xyz_.select(1,0).slice(0,0,500).copy_(torch::linspace(-1,1,500,opts));
    gaussians_->xyz_[500].fill_(100000);
    gaussians_->protection_.reset(501,opts);
    gaussians_->protection_.maturity.fill_(.65);
    gaussians_->protection_.maturity[500].fill_(.2);
    mapping_iter_=5;
    initializeTransformFieldIfNeeded(); check(!transform_field_,"snapshot without true cycle initialized TF");
    gaussians_->protection_.first_valid_maturity_cycle_iteration=10; mapping_iter_=10;
    gaussians_->protection_.maturity[499].fill_(.2);
    prepareElasticEntryState();
    initializeTransformFieldIfNeeded(); check(!transform_field_,"499 ready rows initialized TF");
    gaussians_->protection_.maturity[499].fill_(.65);
    gaussians_->protection_.mapping_iteration=10;
    prepareElasticEntryState(); initializeTransformFieldIfNeeded();
    check(bool(transform_field_),"500 rows plus true cycle failed to initialize TF");
    check(continuous_tf_statistics_.first_ready_population_iteration==10 &&
        continuous_tf_statistics_.ready_count_when_initialized==500 &&
        continuous_tf_statistics_.initial_aabb_population==500,"initial population logging");
    check(transform_field_->aabbMax().max().item<float>()<2,"low maturity outlier affected initial AABB");
    const auto field=transform_field_.ptr();
    const auto lo=field->aabbMin().clone(),hi=field->aabbMax().clone();
    gaussians_->protection_.maturity.zero_();
    initializeTransformFieldIfNeeded(); evaluateDynamicAABB(true);
    check(transform_field_.ptr()==field && lo.equal(field->aabbMin()) && hi.equal(field->aabbMax()),
        "empty readiness population destroyed TF or rebuilt AABB");
    gaussians_->protection_.maturity[0].fill_(.65);
    evaluateDynamicAABB(true);
    check(lo.equal(field->aabbMin())&&hi.equal(field->aabbMax()),"one-row domain forced rebuild");
  }
};

int main(int argc,char**) {
  const auto device=argc>1?torch::kCUDA:torch::kCPU;
  const auto opts=torch::TensorOptions().dtype(torch::kFloat32).device(device);
  torch::manual_seed(0);
  auto m=torch::tensor({.2f,.3f,.65f,1.f},opts).requires_grad_(true);
  const auto indices=torch::nonzero(m.detach()>=.3).reshape({-1});
  auto scores=torch::tensor({.7f,.8f,.8f,.8f},opts).requires_grad_(true);
  int64_t forwarded=0;
  const auto p=elasticSelectorProbability(m,m,.3,.1,indices,[&](const torch::Tensor& rows) {
    forwarded=rows.numel();check(rows.equal(torch::tensor({1,2,3},opts.dtype(torch::kLong))),"Selector forwarded low row");
    return scores.index_select(0,rows);
  });
  check(forwarded==3 && torch::allclose(p,torch::tensor({.1f,.1f,.45f,.8f},opts)),"Selector entry endpoints");
  check(torch::allclose(elasticMaturity(m,.3),torch::tensor({0.f,0.f,.5f,1.f},opts)),"elastic remapping");
  p.sum().backward();
  check(!m.grad().defined() && torch::allclose(scores.grad(),torch::tensor({0.f,0.f,.5f,1.f},opts)),"Selector low-row gradient leak");
  const auto near=torch::tensor({.3f-1e-6f,.3f,.3f+1e-6f},opts);
  const auto near_rows=torch::nonzero(near>=.3).reshape({-1});
  const auto continuous=elasticSelectorProbability(near,near,.3,.1,near_rows,[&](const torch::Tensor& rows) {
    return torch::full({rows.numel()},.8,opts);
  });
  check((continuous-.1).abs().max().item<float>()<2e-6,"entry probability discontinuity");
  bool called=false;
  auto none=elasticSelectorProbability(torch::full({4},.2,opts),m,.3,.1,torch::empty({0},opts.dtype(torch::kLong)),
      [&](const torch::Tensor& rows){called=true;return torch::zeros({rows.numel()},opts);});
  check(!called && !none.requires_grad(),"empty entry population executed Selector");
  auto master=torch::ones({4,3},opts).requires_grad_(true);
  const auto mask=sampleSelectorST(p.detach(),torch::zeros({4},opts));
  (master*mask.unsqueeze(1)).sum().backward();
  check(master.grad()[0].abs().sum().item<float>()>0,"neutral low row lost Master LoD gradient");
  // Disabled feature preserves probability and gradients exactly, including legacy epsilon.
  auto old_scores=scores.detach().clone().requires_grad_(true);
  auto new_scores=scores.detach().clone().requires_grad_(true);
  auto old_probability=mixSelectorProbability(old_scores,m,.1);
  auto new_probability=elasticSelectorProbability(m,m,0,.1,torch::Tensor{},[&](const torch::Tensor& rows) {
    check(!rows.defined(),"entry=0 compacted old all-row path");return new_scores;
  });
  check(old_probability.equal(new_probability),"entry=0 probability parity");
  selectorRatioLoss(old_probability,.1).backward(); selectorRatioLoss(new_probability,.1).backward();
  check(old_scores.grad().equal(new_scores.grad()),"entry=0 gradient parity");
  const auto valid=torch::tensor({true,true,true,false},opts.dtype(torch::kBool));
  const auto all=torch::ones({4},valid.options());
  auto teacher=GITeacherBuilder().build(scores.detach(),valid,all,m,.1,0,0,.3);
  check(torch::allclose(teacher.weight,torch::tensor({0.f,0.f,.5f,0.f},opts)),"GI entry weights or invalid row");
  auto unit_teacher=GITeacherBuilder().build(scores.detach(),all,all,m,.1,0,0,.3);
  check(torch::allclose(unit_teacher.weight,torch::tensor({0.f,0.f,.5f,1.f},opts)),"GI full elastic weight");
  auto legacy=GITeacherBuilder().build(scores.detach(),valid,all,m,.1,0,0);
  auto zero_entry=GITeacherBuilder().build(scores.detach(),valid,all,m,.1,0,0,0);
  check(legacy.weight.equal(zero_entry.weight)&&legacy.label.equal(zero_entry.label),"GI entry=0 parity");
  TransformFieldConfig config;config.backend=device==torch::kCUDA?"cuda":"legacy";
  config.elastic_entry_maturity=.3;config.hexplane.spatial_resolution=4;config.hexplane.ratio_resolution=4;
  config.hexplane.feature_dim=2;config.hexplane.multires={1};config.hidden_dim=8;
  TransformField tf(torch::full({3},-2.,opts),torch::full({3},2.,opts),config); tf->to(device);
  {torch::NoGradGuard guard;for(const auto& param:tf->parameters())param.zero_();
    auto params=tf->named_parameters();params["position_out.bias"].fill_(.3);params["scale_out.bias"].fill_(.2);
    params["rotation_out.bias"][1].fill_(.6);}
  const auto xyz=torch::zeros({4,3},opts),scales=torch::ones({4,3},opts);
  auto q=torch::zeros({4,4},opts);q.select(1,0).fill_(1);
  check(tf->candidateMask(xyz,m).equal(torch::tensor({false,true,true,true},valid.options())),"TF entry candidates");
  const auto weights=elasticMaturity(m.index_select(0,indices),.3);
  const auto output=tf->applyResidual(xyz.index_select(0,indices),scales.index_select(0,indices),q.index_select(0,indices),.1,weights);
  check(output.xyz[0].equal(xyz[1])&&output.rotation[0].equal(q[1]),"boundary TF not identity");
  const auto full=tf->applyResidual(xyz,scales,q,.1);
  check(torch::allclose(output.xyz[1],full.xyz[2]*.5)&&torch::allclose(output.scaling[1].log(),full.scaling[2].log()*.5),"half TF correction");
  check(output.rotation[1][1].item<float>()>0&&output.rotation[1][1].item<float>()<full.rotation[2][1].item<float>(),"half TF rotation");
  check(output.xyz[2].equal(full.xyz[3])&&output.rotation[2].equal(full.rotation[3]),"full TF correction parity");
  GaussianProtection protection;protection.reset(4,opts);protection.maturity.copy_(m.detach());
  check(protection.indicesAbove(.3).numel()==3,"entry cache initial");
  protection.maturity[0].fill_(.65);check(protection.indicesAbove(.3).numel()==4,"entry cache missed maturity reversal");
  protection.append(1);check(protection.indicesAbove(.3).numel()==4,"entry cache missed append");
  config.start_mode="maturity_population";
  check(!config.ready(3000,-1,500)&&!config.ready(10,10,499)&&config.ready(10,10,500),"readiness boundary");
  check(elasticDomainMask(torch::tensor({{0.f,0.f,0.f},{100000.f,0.f,0.f}},opts),
      torch::tensor({.65f,.2f},opts),.3).equal(torch::tensor({true,false},valid.options())),"low point affects domain");
  const auto output_dir=std::filesystem::temp_directory_path()/
      ("cartgs-elastic-entry-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  EntryMapper mapper(device,output_dir);mapper.verify(device);
}
