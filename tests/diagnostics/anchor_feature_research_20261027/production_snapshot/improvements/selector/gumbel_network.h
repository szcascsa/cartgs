#pragma once

#include <torch/torch.h>
#include "include/model_snapshot.h"
#include "anchor_bank.h"
#include "gaussian_compensation_bank.h"
#include "spatial_compensation.h"
#ifdef CARTGS_FUSED_MAPPING_CUDA
#include "include/anchor_input_cuda.h"
#endif
#ifdef CARTGS_EXPERIMENTAL_SELECTOR_CUDA
#include "include/selector_cuda.h"
#endif

#include <cstdint>
#include <tuple>

namespace improvements::selector {

class GumbelNetworkImpl : public torch::nn::Module {
 public:
  static constexpr int64_t kEmbeddingWidth = 32;

  // The mapper config selects CUDA by default; bare CPU/oracle modules stay legacy.
  void setBackend(const std::string& selector_backend) {
    TORCH_CHECK(!anchors_ || selector_backend == "legacy", "Anchor Selector uses LibTorch; legacy CUDA kernel has a different layout");
    TORCH_CHECK(selector_backend == "legacy" || selector_backend == "cuda",
                "Selector.backend must be legacy or cuda");
#ifndef CARTGS_EXPERIMENTAL_SELECTOR_CUDA
    TORCH_CHECK(selector_backend == "legacy",
                "Selector CUDA backend was not compiled; rebuild with "
                "-DCARTGS_EXPERIMENTAL_SELECTOR_CUDA=ON");
#endif
    selector_backend_ = selector_backend;
  }
  const std::string& backend() const { return selector_backend_; }
  void setFusedInputPack(bool enabled) { fused_input_pack_ = enabled; }

  explicit GumbelNetworkImpl(double anchor_voxel_size = 0, bool anchor_gaussian_attributes = false,
                             bool anchor_global_xyz = false, bool explicit_blocks = false,
                             const std::vector<double>& ratio_knots = {}, int64_t anchor_feature_dim = 16,
                             int64_t ratio_shared_dim = 8)
      : anchor_gaussian_attributes_(anchor_gaussian_attributes), anchor_global_xyz_(anchor_global_xyz) {
    TORCH_CHECK(!anchor_global_xyz || anchor_gaussian_attributes, "Global XYZ requires Anchor Gaussian attributes");
    TORCH_CHECK(!explicit_blocks || (anchor_gaussian_attributes && !anchor_global_xyz && anchor_voxel_size>0),
                "Explicit Anchor map requires the attribute-conditioned Selector");
    TORCH_CHECK(!anchor_gaussian_attributes || anchor_voxel_size > 0,
                "Gaussian attribute inputs require Anchor Selector");
    TORCH_CHECK(ratio_knots.empty() || (explicit_blocks && anchor_gaussian_attributes && !anchor_global_xyz),
                "Ratio features require explicit attribute-conditioned Anchors");
    TORCH_CHECK(anchor_feature_dim==16 || ((anchor_feature_dim==0 || anchor_feature_dim==32) && ratio_knots.empty() && explicit_blocks),
                "0D/32D Anchor features require the non-ratio explicit Anchor path");
    if (anchor_voxel_size > 0) {
      anchors_ = register_module("anchors", AnchorBank(anchor_voxel_size,anchor_feature_dim));
      if (explicit_blocks) anchors_->enableExplicitBlocks();
      if (!ratio_knots.empty()) anchors_->enableRatioFeatures(ratio_knots,ratio_shared_dim);
      if (anchor_global_xyz_) anchors_->enableCellLocalOffsets();
      pos_emd_ = register_module("local_emd", makeEmbedding(!ratio_knots.empty() ? 26 : (anchor_feature_dim + 4 +
          (anchor_gaussian_attributes_ ? 7 : 0) + (anchor_global_xyz_ ? 3 : 0))));
      register_buffer("anchor_layout_version", torch::tensor(anchor_feature_dim==0 ? 10 : (anchor_feature_dim==32 ? 8 : (!ratio_knots.empty() ? (ratio_shared_dim==16?9:7) : (explicit_blocks ? 5 : (anchor_global_xyz_ ? 4 : (anchor_gaussian_attributes_ ? 3 : 2))))), torch::kLong));
    } else {
      pos_emd_ = register_module("pos_emd", makeEmbedding(3));
      rotation_emd_ = register_module("rotation_emd", makeEmbedding(4));
      scale_emd_ = register_module("scale_emd", makeEmbedding(3));
      time_emd_ = register_module("time_emd", makeEmbedding(1));
    }
    soft_net_ = register_module("soft_net", torch::nn::Linear(kEmbeddingWidth * (anchors_ ? 1 : 4), 2));
    if(anchors_ && anchor_feature_dim==0) {
      torch::NoGradGuard guard;soft_net_->weight.zero_();soft_net_->bias.zero_();
    }
    if(!ratio_knots.empty() && ratio_shared_dim==16) {
      ScopedTorchRNG rng{torch::Device(torch::kCPU)};
      auto next=makeEmbedding(34);
      {torch::NoGradGuard guard;
        // Extra shared features start at zero; retain the original input function and RNG clock.
        next->at<torch::nn::LinearImpl>(0).weight.slice(1,0,8).copy_(pos_emd_->at<torch::nn::LinearImpl>(0).weight.slice(1,0,8));
        next->at<torch::nn::LinearImpl>(0).weight.slice(1,16,34).copy_(pos_emd_->at<torch::nn::LinearImpl>(0).weight.slice(1,8,26));
        next->at<torch::nn::LinearImpl>(0).bias.copy_(pos_emd_->at<torch::nn::LinearImpl>(0).bias);
        next->at<torch::nn::LinearImpl>(2).weight.copy_(pos_emd_->at<torch::nn::LinearImpl>(2).weight);
        next->at<torch::nn::LinearImpl>(2).bias.copy_(pos_emd_->at<torch::nn::LinearImpl>(2).bias);}
      pos_emd_=replace_module("local_emd",next);
    }
  }
  bool anchorConditioned() const { return static_cast<bool>(anchors_); }
  bool anchorGaussianAttributes() const { return anchor_gaussian_attributes_; }
  bool anchorGlobalXYZ() const { return anchor_global_xyz_; }
  int64_t ratioScalarInput() const { return ratio_scalar_input_; }
  void enableRatioScalarInput(int64_t mode) {
    TORCH_CHECK(anchors_ && anchors_->ratioFeatures() && (mode==1 || mode==2),
                "Ratio scalar input requires ratio features and mode1(log)/2(raw)");
    TORCH_CHECK(!ratio_scalar_input_ || ratio_scalar_input_==mode,"Ratio scalar input mode mismatch");
    if(ratio_scalar_input_)return;
    ScopedTorchRNG rng{torch::Device(torch::kCPU)};
    const auto width=pos_emd_->at<torch::nn::LinearImpl>(0).weight.size(1);
    auto next=makeEmbedding(width+1);
    next->to(soft_net_->weight.device(),soft_net_->weight.scalar_type());
    {torch::NoGradGuard guard;
      next->at<torch::nn::LinearImpl>(0).weight.slice(1,0,width).copy_(pos_emd_->at<torch::nn::LinearImpl>(0).weight);
      next->at<torch::nn::LinearImpl>(0).weight.select(1,width).zero_();
      next->at<torch::nn::LinearImpl>(0).bias.copy_(pos_emd_->at<torch::nn::LinearImpl>(0).bias);
      next->at<torch::nn::LinearImpl>(2).weight.copy_(pos_emd_->at<torch::nn::LinearImpl>(2).weight);
      next->at<torch::nn::LinearImpl>(2).bias.copy_(pos_emd_->at<torch::nn::LinearImpl>(2).bias);}
    pos_emd_=replace_module("local_emd",next);
    ratio_scalar_input_=mode;
    register_buffer("anchor_ratio_scalar_input",torch::tensor(mode,anchors_->rows().options()));
  }
  bool compensationEnabled() const { return static_cast<bool>(compensation_) || static_cast<bool>(spatial_compensation_); }
  bool spatialCompensationEnabled() const { return static_cast<bool>(spatial_compensation_); }
  SpatialCompensation& spatialCompensation(){return spatial_compensation_;}
  void enableSpatialCompensation() {
    TORCH_CHECK(anchors_&&anchors_->variableMembers()&&!compensation_,"Spatial compensation requires Variable Anchors and its own head");
    if(spatial_compensation_)return;
    ScopedTorchRNG rng{torch::Device(torch::kCPU)};
    spatial_compensation_=register_module("spatial_compensation",SpatialCompensation());
    spatial_compensation_->to(soft_net_->weight.device(),soft_net_->weight.scalar_type());
    compensation_schema_=register_buffer("compensation_schema",torch::tensor(2,anchors_->rows().options()));
    register_buffer("spatial_compensation_schema",torch::tensor(1,anchors_->rows().options()));
  }
  int64_t compensationOutputWidth() const {
    if(spatial_compensation_)return 14;
    return compensation_ ? compensation_->at<torch::nn::LinearImpl>(compensation_->size()-1).weight.size(0) : 0;
  }
  int64_t compensationHiddenLayers() const { return compensation_?(compensation_->size()-1)/2:0; }
  void enableCompensation(int64_t output_width = 14) {
    TORCH_CHECK(anchors_ && anchors_->explicitBlocks() && anchor_gaussian_attributes_ && !anchor_global_xyz_,
                "Compensation requires the explicit attribute-conditioned Anchor Selector");
    TORCH_CHECK(output_width==10 || output_width==14,"Unknown compensation output layout");
    if(compensationEnabled()) {
      TORCH_CHECK(compensationOutputWidth()==output_width,"Compensation checkpoint layout mismatch");
      return;
    }
    ScopedTorchRNG rng{torch::Device(torch::kCPU)};
    compensation_=register_module("compensation",torch::nn::Sequential(
        torch::nn::Linear(32,32),torch::nn::ReLU(),torch::nn::Linear(32,output_width)));
    {torch::NoGradGuard guard;
      compensation_->at<torch::nn::LinearImpl>(2).weight.zero_();
      compensation_->at<torch::nn::LinearImpl>(2).bias.zero_();}
    compensation_->to(soft_net_->weight.device(),soft_net_->weight.scalar_type());
    compensation_schema_=register_buffer("compensation_schema",torch::tensor(output_width==14?2:1,anchors_->rows().options()));
  }
  struct SelectionFeatures { torch::Tensor probability, shared; };
  void enableAnchorRatioCompensation(const std::vector<double>& knots,int64_t feature_width=14) {
    const bool preset=(feature_width==8 || feature_width==50) && !anchors_->ratioFeatures() && anchors_->features().size(1)==0;
    const bool legacy=feature_width==14 && (anchors_->ratioFeatures()
        ?(anchors_->features().size(1)==8 || anchors_->features().size(1)==16):anchors_->features().size(1)==16);
    TORCH_CHECK(compensationOutputWidth()==14 && !gaussian_compensation_ && (preset || legacy),
                "Anchor compensation requires legacy latent14 or S00R00E08/E50D12");
    TORCH_CHECK(!anchors_->compensationFeatures(),"Anchor ratio compensation already enabled");
    ScopedTorchRNG rng{torch::Device(torch::kCPU)};
    auto old=compensation_;
    auto next=makeCompensationHead(32+feature_width,preset?2:1);
    next->to(soft_net_->weight.device(),soft_net_->weight.scalar_type());
    {torch::NoGradGuard guard;
      next->at<torch::nn::LinearImpl>(0).weight.slice(1,0,32).copy_(old->at<torch::nn::LinearImpl>(0).weight);
      next->at<torch::nn::LinearImpl>(0).bias.copy_(old->at<torch::nn::LinearImpl>(0).bias);
      if(preset)next->at<torch::nn::LinearImpl>(0).weight.slice(1,32,32+feature_width).uniform_(-.05,.05);
      next->at<torch::nn::LinearImpl>(next->size()-1).weight.copy_(old->at<torch::nn::LinearImpl>(2).weight);
      next->at<torch::nn::LinearImpl>(next->size()-1).bias.copy_(old->at<torch::nn::LinearImpl>(2).bias);}
    compensation_=replace_module("compensation",next);
    anchors_->enableCompensationFeatures(knots,feature_width);
    register_buffer("anchor_ratio_compensation_schema",torch::tensor(preset?(feature_width==50?3:2):1,anchors_->rows().options()));
  }
  SelectionFeatures forwardSelectionFeatures(const torch::Tensor& xyz,float ratio,double tau,
      const torch::Tensor& rows,const torch::Tensor& scale,const torch::Tensor& dc,const torch::Tensor& opacity) {
    TORCH_CHECK(compensationEnabled(),"Shared forward requires compensation");
    auto u=anchorInputs(xyz,ratio,rows,scale,dc,opacity);
    auto shared=torch::relu(pos_emd_->at<torch::nn::LinearImpl>(0).forward(u));
    auto hidden=torch::relu(pos_emd_->at<torch::nn::LinearImpl>(2).forward(shared));
    return {(soft_net_->forward(hidden)/tau).softmax(-1).select(1,1),shared};
  }
  void enablePrivateCompensation(bool independent,const std::vector<double>& knots) {
    TORCH_CHECK(compensationOutputWidth()==14 && anchors_->variableMembers(),
                "Gaussian-private compensation requires the variable Anchor 14D path");
    TORCH_CHECK(!gaussian_compensation_,"Private compensation already enabled");
    gaussian_compensation_=register_module("gaussian_compensation",GaussianCompensationBank(independent,knots));
    gaussian_compensation_->to(soft_net_->weight.device());
    private_compensation_mode_=register_buffer("private_compensation_mode",
        torch::tensor(independent?3:2,anchors_->rows().options()));
    for(auto parameter:compensation_->parameters())parameter.requires_grad_(false);
  }
  GaussianCompensationBank& privateCompensation() { return gaussian_compensation_; }
  const GaussianCompensationBank& privateCompensation() const { return gaussian_compensation_; }
  torch::Tensor compensationResidual(const torch::Tensor& selected_shared,
      const torch::Tensor& source_rows={},double ratio=0) {
    TORCH_CHECK(!spatial_compensation_,"Spatial compensation requires live Gaussian attributes, not the Selector's shared encoding");
    TORCH_CHECK(compensationEnabled() && selected_shared.dim()==2 && selected_shared.size(1)==32,
                "Invalid selected compensation features");
    auto input=selected_shared;
    if(anchors_->compensationFeatures())
      input=torch::cat({selected_shared,anchors_->compensationInput(source_rows,ratio)},1);
    auto baseline=compensation_->forward(input);
    if(!gaussian_compensation_)return baseline;
    auto ids=source_rows.defined()?source_rows:torch::arange(selected_shared.size(0),anchors_->rows().options());
    TORCH_CHECK(ids.numel()==selected_shared.size(0) && (source_rows.defined() || ids.numel()==gaussian_compensation_->size()),
                "Compact private compensation requires explicit source rows");
    return baseline.detach()+gaussian_compensation_->forward(ids,ratio);
  }
  static GaussianAttrState applyCompensationResidual(const GaussianAttrState& state,const torch::Tensor& residual) {
    auto identity=torch::zeros_like(residual.slice(1,6,10));identity.select(1,0).fill_(1);
    auto correction=quaternion_utils::normalize(identity+residual.slice(1,6,10));
    return {state.xyz+residual.slice(1,0,3),state.log_scaling+residual.slice(1,3,6),
        quaternion_utils::normalize(quaternion_utils::multiply(correction,state.rotation))};
  }
  // Keep logits explicit: logit(sigmoid(logit)) loses finite saturated values.
  static std::pair<torch::Tensor,torch::Tensor> applyAppearanceResidual(
      const torch::Tensor& dc,const torch::Tensor& opacity_logits,const torch::Tensor& residual) {
    if(residual.size(1)==10)return {dc,opacity_logits};
    return {dc+residual.slice(1,11,14).reshape_as(dc),opacity_logits+residual.slice(1,10,11)};
  }
  torch::Tensor forwardLogCompensationResidual(const GaussianAttrState& state,float ratio,const torch::Tensor& rows,
      const torch::Tensor& dc,const torch::Tensor& opacity) {
    auto selection=forwardSelectionFeatures(state.xyz,ratio,1,rows,state.log_scaling.exp(),dc,opacity);
    return compensationResidual(selection.shared,rows,ratio);
  }
  GaussianAttrState applyLogCompensation(const GaussianAttrState& state,float ratio,const torch::Tensor& rows,
      const torch::Tensor& dc,const torch::Tensor& opacity) {
    return applyCompensationResidual(state,forwardLogCompensationResidual(state,ratio,rows,dc,opacity));
  }
  void load(torch::serialize::InputArchive& archive) override {
    torch::Tensor layout;
    if(anchors_ && archive.try_read("anchor_layout_version",layout,true)) {
      if(layout.item<int64_t>()==10 && anchors_->features().size(1)!=0) {
        anchors_->configureFeatureWidth(0);
        pos_emd_=replace_module("local_emd",makeEmbedding(11));
        pos_emd_->to(soft_net_->weight.device(),soft_net_->weight.scalar_type());
      }
      TORCH_CHECK((layout.item<int64_t>()==10)==(anchors_->features().size(1)==0),"Selector zero-feature checkpoint mismatch");
      if(layout.item<int64_t>()==8 && anchors_->features().size(1)!=32) {
        anchors_->enableWideFeatures();
        pos_emd_=replace_module("local_emd",makeEmbedding(43));
        pos_emd_->to(soft_net_->weight.device(),soft_net_->weight.scalar_type());
      }
      if((layout.item<int64_t>()==7 || layout.item<int64_t>()==9) && !anchors_->ratioFeatures()) {
        torch::serialize::InputArchive bank;archive.read("anchors",bank);
        torch::Tensor knots;bank.read("ratio_knots",knots,true);
        const auto cpu=knots.to(torch::kCPU,torch::kFloat64).contiguous();
        const int64_t shared_width=layout.item<int64_t>()==9?16:8;
        anchors_->enableRatioFeatures(std::vector<double>(cpu.data_ptr<double>(),cpu.data_ptr<double>()+cpu.numel()),shared_width);
        pos_emd_=replace_module("local_emd",makeEmbedding(shared_width+18));
        pos_emd_->to(soft_net_->weight.device(),soft_net_->weight.scalar_type());
      }
      TORCH_CHECK((layout.item<int64_t>()==7 || layout.item<int64_t>()==9)==anchors_->ratioFeatures(),"Selector checkpoint feature schema mismatch");
      if(anchors_->ratioFeatures())TORCH_CHECK(anchors_->features().size(1)==(layout.item<int64_t>()==9?16:8),"Ratio shared Feature width mismatch");
      TORCH_CHECK((layout.item<int64_t>()==8)==(anchors_->features().size(1)==32),"Selector checkpoint Feature width mismatch");
    }
    torch::Tensor scalar_mode;
    if(archive.try_read("anchor_ratio_scalar_input",scalar_mode,true))enableRatioScalarInput(scalar_mode.item<int64_t>());
    else TORCH_CHECK(!ratio_scalar_input_,"Checkpoint has no ratio scalar input");
    torch::Tensor schema,spatial_schema;
    const bool has_spatial=archive.try_read("spatial_compensation_schema",spatial_schema,true);
    if(archive.try_read("compensation_schema",schema,true)) {
      const auto version=schema.item<int64_t>();
      TORCH_CHECK(version==1 || version==2,"Unknown compensation checkpoint schema");
      if(has_spatial) {
        TORCH_CHECK(version==2&&spatial_schema.item<int64_t>()==1,"Unknown Spatial compensation schema");
        if(!anchors_->variableMembers())anchors_->enableVariableMembers();
        enableSpatialCompensation();
      } else enableCompensation(version==1?10:14);
    }
    torch::Tensor mode;
    torch::Tensor anchor_compensation;
    if(archive.try_read("anchor_ratio_compensation_schema",anchor_compensation,true)) {
      const auto version=anchor_compensation.item<int64_t>();
      TORCH_CHECK(version>=1 && version<=3,"Unknown Anchor ratio compensation schema");
      const int64_t feature_width=version==1?14:(version==2?8:50);
      torch::serialize::InputArchive bank;archive.read("anchors",bank);
      torch::Tensor knots;bank.read("compensation_ratio_knots",knots,true);
      const auto cpu=knots.to(torch::kCPU,torch::kFloat64).contiguous();
      if(!anchors_->compensationFeatures())enableAnchorRatioCompensation(
          std::vector<double>(cpu.data_ptr<double>(),cpu.data_ptr<double>()+cpu.numel()),feature_width);
      TORCH_CHECK(anchors_->compensationFeatureWidth()==feature_width && compensationHiddenLayers()==(version==1?1:2),
                  "Anchor compensation checkpoint head/feature mismatch");
    }
    if(archive.try_read("private_compensation_mode",mode,true)) {
      torch::serialize::InputArchive bank;archive.read("gaussian_compensation",bank);
      torch::Tensor knots,uids;bank.read("ratio_knots",knots,true);bank.read("gaussian_uids",uids,true);
      const auto cpu=knots.to(torch::kCPU,torch::kFloat64).contiguous();
      if(!anchors_->variableMembers())anchors_->enableVariableMembers();
      if(!gaussian_compensation_)enablePrivateCompensation(mode.item<int64_t>()==3,
          std::vector<double>(cpu.data_ptr<double>(),cpu.data_ptr<double>()+cpu.numel()));
      TORCH_CHECK(mode.item<int64_t>()==2 || mode.item<int64_t>()==3,"Unknown private compensation mode");
      TORCH_CHECK(gaussian_compensation_->independent()==(mode.item<int64_t>()==3),"Private compensation checkpoint mismatch");
      TORCH_CHECK(gaussian_compensation_->size()==0,"Load private compensation into an empty identity bank");
      gaussian_compensation_->append(uids.numel(),{},{});
    }
    torch::nn::Module::load(archive);
    if(gaussian_compensation_) {
      TORCH_CHECK(gaussian_compensation_->size()==anchors_->rows().numel(),"Private checkpoint population mismatch");
      for(auto table:gaussian_compensation_->tables())
        TORCH_CHECK(table.dim()==2 && table.size(0)==gaussian_compensation_->size() && table.size(1)==14,
                    "Private checkpoint residual shape mismatch");
    }
  }
  AnchorBank& anchors() { return anchors_; }
  const AnchorBank& anchors() const { return anchors_; }
  torch::Tensor maskInactiveSlots(const torch::Tensor& probability, const torch::Tensor& opacity) const {
    return anchors_ ? anchors_->maskSlotProbability(probability,opacity) : probability;
  }
  torch::Tensor forwardAnchorProbability(const torch::Tensor& xyz, float ratio,
      double tau = 1.0, const torch::Tensor& source_rows = {},
      const torch::Tensor& scale = {}, const torch::Tensor& sh_dc = {}, const torch::Tensor& opacity = {}) {
    TORCH_CHECK(anchors_, "Anchor forward requested on legacy Selector");
    return forwardProbability(xyz, {}, scale, ratio, tau, source_rows, sh_dc, opacity);
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& position,
      const torch::Tensor& rotation,
      const torch::Tensor& scale,
      const torch::Tensor& target_ratio,
      const double tau = 1.0) {
    auto logits = logitsFor(position, rotation, scale, target_ratio);
    auto hard_output = gumbelSoftmax(logits, tau, true);
    auto soft_output = gumbelSoftmax(logits, tau, false);

    return {anchors_ ? logits.select(1, 1) - logits.select(1, 0) : logits.select(1, 1), hard_output.select(1, 1),
            soft_output.select(1, 1)};
  }

  // LOD only needs the hard straight-through sample, not a second soft sample.
  torch::Tensor forwardHard(const torch::Tensor& position,
                            const torch::Tensor& rotation,
                            const torch::Tensor& scale, float ratio,
                            double tau = 1.0,
                            torch::Tensor* keep_probability = nullptr) {
    auto logits = logitsFor(position, rotation, scale,
        torch::full({1, 1}, ratio, position.options()));
    // Gumbel argmax has softmax(logits) probabilities, independent of tau.
    // Reuse the logits for deterministic normal-training budget supervision.
    if (keep_probability) *keep_probability = logits.softmax(-1).select(1, 1);
    return gumbelSoftmax(logits, tau, true).select(1, 1);
  }

  // Deterministic keep probability used by Change Distill. It deliberately
  // avoids Gumbel noise so the teacher/student comparison is stable.
  torch::Tensor forwardKeepLogit(const torch::Tensor& position,
                                const torch::Tensor& rotation,
                                const torch::Tensor& scale, float ratio,
                                double tau = 1.0) {
    const auto logits = logitsFor(position, rotation, scale,
        torch::full({1, 1}, ratio, position.options()));
    return (logits.select(1, 1) - logits.select(1, 0)) / tau;
  }

  // Historical guard only: share one uniform draw across baseline and all
  // candidates. The forward has the SAME Bernoulli law as Gumbel argmax;
  // the backward is a straight-through relaxation. Normal training is unchanged.
  torch::Tensor forwardHardWithUniform(const torch::Tensor& position,
                                      const torch::Tensor& rotation,
                                      const torch::Tensor& scale, float ratio,
                                      const torch::Tensor& uniform,
                                      double tau = 1.0) {
    TORCH_CHECK(tau > 0.0 && uniform.numel() == position.size(0),
                "Fixed-uniform Selector input mismatch");
    auto logit = forwardKeepLogit(position, rotation, scale, ratio, 1.0);
    auto u = uniform.reshape({-1}).clamp(1e-7, 1.0 - 1e-7);
    auto threshold = u.log() - (1.0 - u).log();
    auto soft = ((logit - threshold) / tau).sigmoid();
    auto hard = (uniform.reshape({-1}) < logit.sigmoid()).to(soft.scalar_type());
    return hard - soft.detach() + soft;
  }

  torch::Tensor forwardProbability(const torch::Tensor& position,
                                   const torch::Tensor& rotation,
                                   const torch::Tensor& scale,
                                   const torch::Tensor& target_ratio,
                                   const double tau = 1.0,
                                   const torch::Tensor& source_rows = {},
                                   const torch::Tensor& sh_dc = {}, const torch::Tensor& opacity = {}) {
    const auto logits = logitsFor(position, rotation, scale, target_ratio, source_rows, sh_dc, opacity);
    return (logits / tau).softmax(-1).select(1, 1);
  }

  torch::Tensor forwardProbability(const torch::Tensor& position,
                                   const torch::Tensor& rotation,
                                   const torch::Tensor& scale, float ratio,
                                   double tau = 1.0, const torch::Tensor& source_rows = {},
                                   const torch::Tensor& sh_dc = {}, const torch::Tensor& opacity = {}) {
    if(anchors_ && anchors_->ratioFeatures())
      return (soft_net_->forward(pos_emd_->forward(anchorInputs(position,ratio,source_rows,scale,sh_dc,opacity)))/tau)
          .softmax(-1).select(1,1);
    return forwardProbability(position, rotation, scale,
        torch::full({1, 1}, ratio, position.options()), tau, source_rows, sh_dc, opacity);
  }

  std::shared_ptr<GumbelNetworkImpl> cloneFrozen() const {
    ScopedTorchRNG rng{torch::Device(torch::kCPU)};
    auto teacher = std::make_shared<GumbelNetworkImpl>(anchors_ ? anchors_->voxelSize() : 0,
        anchor_gaussian_attributes_, anchor_global_xyz_, anchors_ && anchors_->explicitBlocks(),
        anchors_ && anchors_->ratioFeatures() ? anchors_->ratioValues() : std::vector<double>{},
        anchors_ && !anchors_->ratioFeatures() ? anchors_->features().size(1) : 16,
        anchors_ && anchors_->ratioFeatures() ? anchors_->features().size(1) : 8);
    teacher->to(parameters().front().device(), parameters().front().scalar_type());
    if(ratio_scalar_input_)teacher->enableRatioScalarInput(ratio_scalar_input_);
    if (anchors_) teacher->anchors_->copyLayoutFrom(*anchors_);
    if(spatial_compensation_)teacher->enableSpatialCompensation();
    else if(compensationEnabled())teacher->enableCompensation(compensationOutputWidth());
    if(anchors_ && anchors_->compensationFeatures()) {
      // copyLayoutFrom already created the tables; widen the head without recreating them.
      teacher->compensation_=teacher->replace_module("compensation",
          makeCompensationHead(32+anchors_->compensationFeatureWidth(),compensationHiddenLayers()));
      teacher->compensation_->to(soft_net_->weight.device(),soft_net_->weight.scalar_type());
      teacher->register_buffer("anchor_ratio_compensation_schema",torch::tensor(anchors_->compensationFeatureWidth()==50?3:(anchors_->compensationFeatureWidth()==8?2:1),anchors_->rows().options()));
    }
    if(gaussian_compensation_) {
      teacher->enablePrivateCompensation(gaussian_compensation_->independent(),gaussian_compensation_->ratioValues());
      teacher->gaussian_compensation_->copyLayoutFrom(*gaussian_compensation_);
    }
    copyFrozenModule(*this, *teacher);
    if(spatial_compensation_)teacher->spatial_compensation_->syncReady();
    teacher->setBackend(selector_backend_);
    teacher->setFusedInputPack(fused_input_pack_);
    if(anchors_)teacher->anchors_->setFusedFullStats(anchors_->fusedFullStats());
    return teacher;
  }

  std::vector<torch::Tensor> forwardProbabilities(const torch::Tensor& position,
      const torch::Tensor& rotation, const torch::Tensor& scale,
      const std::vector<float>& ratios, double tau = 1.0, const torch::Tensor& source_rows = {},
      const torch::Tensor& sh_dc = {}, const torch::Tensor& opacity = {}) {
    std::vector<torch::Tensor> result;
    if(anchors_ && anchors_->ratioFeatures()) {
      for(float ratio:ratios)result.push_back(forwardProbability(position,rotation,scale,ratio,tau,source_rows,sh_dc,opacity));
      return result;
    }
    if (anchors_) {
      const auto input = anchors_->inputs(position, source_rows);
      auto local = torch::cat({input.first, input.second}, 1);
      if (anchor_global_xyz_) local = torch::cat({local, position}, 1);
      if (anchor_gaussian_attributes_)
        local = torch::cat({local, gaussianAttributes(position, scale, sh_dc, opacity)}, 1);
      for (float ratio : ratios) {
        const auto ratio_feature = torch::full({1, 1}, ratio, position.options()).log()
            .expand({position.size(0), 1});
        result.push_back((soft_net_->forward(pos_emd_->forward(torch::cat({local, ratio_feature}, 1))) / tau)
            .softmax(-1).select(1, 1));
      }
      return result;
    }
#ifdef CARTGS_EXPERIMENTAL_SELECTOR_CUDA
    if (selector_backend_ == "cuda" && ratios.size() > 1) {
      std::vector<torch::Tensor> ratio_inputs;
      for (float ratio : ratios)
        ratio_inputs.push_back(torch::full({position.size(0), 1}, ratio, position.options()));
      for (const auto& logits : selector_cuda::networkMany(
               {position, rotation, scale}, ratio_inputs, parameters()))
        result.push_back((logits / tau).softmax(-1).select(1, 1));
      return result;
    }
#endif
    for (float ratio : ratios)
      result.push_back(forwardProbability(position, rotation, scale,
          torch::full({position.size(0), 1}, ratio, position.options()), tau, source_rows, sh_dc, opacity));
    return result;
  }

 private:
  static torch::nn::Sequential makeCompensationHead(int64_t input_width,int64_t hidden_layers) {
    auto head=torch::nn::Sequential(torch::nn::Linear(input_width,32),torch::nn::ReLU());
    if(hidden_layers==2) { head->push_back(torch::nn::Linear(32,32));head->push_back(torch::nn::ReLU()); }
    head->push_back(torch::nn::Linear(32,14));
    return head;
  }
  torch::Tensor anchorInputs(const torch::Tensor& xyz,double ratio,const torch::Tensor& rows,
      const torch::Tensor& scale,const torch::Tensor& dc,const torch::Tensor& opacity) const {
#ifdef CARTGS_FUSED_MAPPING_CUDA
    if(fused_input_pack_ && xyz.is_cuda() && xyz.scalar_type()==torch::kFloat32 &&
        anchors_->cellLocalOffsets() && !anchors_->ratioFeatures() && anchors_->features().size(1)==16 &&
        anchor_gaussian_attributes_ && !anchor_global_xyz_) {
      auto owner=rows.defined()?anchors_->rows().index_select(0,rows):anchors_->rows();
      return mapping_cuda::anchorInput27(xyz,anchors_->features(),owner,anchors_->centers(),
          anchors_->offsetScales(),scale,dc,opacity,ratio);
    }
#endif
    auto input=anchors_->inputs(xyz,rows,ratio);
    std::vector<torch::Tensor> local{input.first,input.second};
    if(anchor_global_xyz_)local.push_back(xyz);
    if(anchor_gaussian_attributes_)local.push_back(gaussianAttributes(xyz,scale,dc,opacity));
    if(!anchors_->ratioFeatures())local.push_back(torch::full({xyz.size(0),1},ratio,xyz.options()).log());
    else if(ratio_scalar_input_) {
      auto scalar=torch::full({xyz.size(0),1},ratio,xyz.options());
      local.push_back(ratio_scalar_input_==1?scalar.log():scalar);
    }
    return torch::cat(local,1);
  }
  torch::Tensor logitsFor(const torch::Tensor& position,
                          const torch::Tensor& rotation,
                          const torch::Tensor& scale,
                          const torch::Tensor& target_ratio,
                          const torch::Tensor& source_rows = {},
                          const torch::Tensor& sh_dc = {}, const torch::Tensor& opacity = {}) {
#ifdef CARTGS_EXPERIMENTAL_SELECTOR_CUDA
    if (selector_backend_ == "cuda")
      return selector_cuda::network({position, rotation, scale, target_ratio},
          parameters(), selector_cuda::Candidate::CublasStrict);
#endif
    if (anchors_) {
      if(anchors_->ratioFeatures()) {
        TORCH_CHECK(target_ratio.numel()==1 || torch::all(target_ratio==target_ratio.reshape({-1})[0]).item<bool>(),
                    "Ratio-feature forward requires one model ratio");
        return soft_net_->forward(pos_emd_->forward(anchorInputs(position,target_ratio.reshape({-1})[0].item<double>(),
            source_rows,scale,sh_dc,opacity)));
      }
      auto input = anchors_->inputs(position, source_rows);
      auto ratio = target_ratio.log();
      if (ratio.size(0) == 1 && position.size(0) != 1) ratio = ratio.expand({position.size(0), 1});
      auto local = torch::cat({input.first, input.second}, 1);
      if (anchor_global_xyz_) local = torch::cat({local, position}, 1);
      if (anchor_gaussian_attributes_)
        local = torch::cat({local, gaussianAttributes(position, scale, sh_dc, opacity)}, 1);
      return soft_net_->forward(pos_emd_->forward(torch::cat({local, ratio}, 1)));
    }
    auto ratio_embedding = time_emd_->forward(target_ratio);
    if (ratio_embedding.size(0) == 1 && position.size(0) != 1)
      ratio_embedding = ratio_embedding.expand({position.size(0), kEmbeddingWidth});
    auto gumbel_input = torch::cat(
        {pos_emd_->forward(position), rotation_emd_->forward(rotation),
         scale_emd_->forward(scale), ratio_embedding},
        1);
    return soft_net_->forward(gumbel_input);
  }

  static torch::nn::Sequential makeEmbedding(const int64_t input_width) {
    return torch::nn::Sequential(
        torch::nn::Linear(input_width, kEmbeddingWidth), torch::nn::ReLU(),
        torch::nn::Linear(kEmbeddingWidth, kEmbeddingWidth),
        torch::nn::ReLU());
  }

  static torch::Tensor gaussianAttributes(const torch::Tensor& xyz, const torch::Tensor& scale,
      const torch::Tensor& sh_dc, const torch::Tensor& opacity) {
    TORCH_CHECK(scale.defined() && sh_dc.defined() && opacity.defined(),
                "Anchor attribute inputs require activated scale, SH DC and activated opacity");
    const auto dc = sh_dc.dim() == 3 ? sh_dc.squeeze(1) : sh_dc;
    TORCH_CHECK(scale.sizes() == xyz.sizes() && dc.sizes() == xyz.sizes() &&
                opacity.dim() == 2 && opacity.size(0) == xyz.size(0) && opacity.size(1) == 1,
                "Anchor attribute rows/shapes must match XYZ");
    return torch::cat({scale, dc, opacity}, 1);
  }

  static torch::Tensor gumbelSoftmax(const torch::Tensor& logits,
                                     const double tau,
                                     const bool hard) {
    auto gumbel_noise = -torch::empty_like(logits).exponential_().log();
    auto soft = ((logits + gumbel_noise) / tau).softmax(-1);
    if (!hard) {
      return soft;
    }

    auto indices = std::get<1>(soft.max(-1, true));
    auto one_hot = torch::zeros_like(logits).scatter_(-1, indices, 1.0);
    return one_hot - soft.detach() + soft;
  }

  torch::nn::Sequential pos_emd_{nullptr};
  torch::nn::Sequential compensation_{nullptr};
  SpatialCompensation spatial_compensation_{nullptr};
  torch::Tensor compensation_schema_;
  GaussianCompensationBank gaussian_compensation_{nullptr};
  torch::Tensor private_compensation_mode_;
  int64_t ratio_scalar_input_ = 0;
  torch::nn::Sequential rotation_emd_{nullptr};
  torch::nn::Sequential scale_emd_{nullptr};
  torch::nn::Sequential time_emd_{nullptr};
  torch::nn::Linear soft_net_{nullptr};
  AnchorBank anchors_{nullptr};
  bool anchor_gaussian_attributes_ = false;
  bool anchor_global_xyz_ = false;
  std::string selector_backend_ = "legacy";
  bool fused_input_pack_ = false;
};

TORCH_MODULE(GumbelNetwork);

}  // namespace improvements::selector
