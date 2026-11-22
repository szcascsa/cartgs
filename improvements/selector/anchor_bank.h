#pragma once

#include <torch/torch.h>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <vector>
#include "../../include/gaussian_transform_ops.h"
#include "../../include/mapping_audit_range.h"
#include "../../include/model_snapshot.h"
#ifdef CARTGS_FUSED_MAPPING_CUDA
#include "../../include/anchor_stats_cuda.h"
#endif

namespace improvements::selector {

// Spatial organization only. Explicit Gaussian XYZ remains the geometry owner.
// Topology/event operations run under the mapper's render lock, after backward.
class AnchorBankImpl : public torch::nn::Module {
 public:
  static constexpr int64_t kFeatureWidth = 16;
  using Key = std::array<int64_t, 3>;
  explicit AnchorBankImpl(double voxel_size, int64_t feature_width = kFeatureWidth) : voxel_size_(voxel_size) {
    TORCH_CHECK(std::isfinite(voxel_size) && voxel_size > 0, "Invalid anchor voxel size");
    TORCH_CHECK(feature_width==0 || feature_width==16 || feature_width==32,"Anchor Feature width must be 0, 16 or 32");
    features_ = register_parameter("features", torch::zeros({0, feature_width}),feature_width>0);
    centers_ = register_buffer("centers", torch::empty({0, 3}));
    rows_ = register_buffer("rows", torch::empty({0}, torch::kLong));
    keys_ = register_buffer("keys", torch::empty({0, 3}, torch::kLong));
    scale_ = register_buffer("voxel_size", torch::tensor(voxel_size, torch::kFloat64));
  }
  const torch::Tensor& features() const { return features_; }
  void enableWideFeatures() {
    TORCH_CHECK(explicitBlocks() && size()==0 && !ratioFeatures(),"32D features require an empty non-ratio explicit Anchor bank");
    configureFeatureWidth(32);
  }
  void configureFeatureWidth(int64_t width) {
    TORCH_CHECK(explicitBlocks() && size()==0 && !ratioFeatures() && (width==0 || width==16 || width==32),
                "Feature layout changes require an empty explicit Anchor bank");
    features_=torch::zeros({0,width},features_.options()).requires_grad_(width>0);
    parameters_["features"]=features_;
  }
  bool ratioFeatures() const { return ratio_knots_.defined(); }
  const torch::Tensor& ratioKnots() const { return ratio_knots_; }
  const std::vector<double>& ratioValues() const { return ratio_values_; }
  bool compensationFeatures() const { return compensation_knots_.defined(); }
  bool spatialFeatures() const { return spatial_features_.defined(); }
  const torch::Tensor& spatialFeaturesTable() const { return spatial_features_; }
  void enableSpatialFeatures(int64_t width=16) {
    TORCH_CHECK(explicitBlocks(),"Spatial Feature requires explicit Anchors");
    TORCH_CHECK(width==16||width==128,"Unsupported Spatial Feature width");
    if(spatialFeatures()){TORCH_CHECK(spatial_features_.size(1)==width,"Spatial Feature width mismatch");return;}
    auto value=torch::zeros({size(),width},features_.options());
    if(width==128)value.normal_(0,.01);
    spatial_features_=register_parameter("spatial_features",value.requires_grad_(true));
  }
  const std::vector<double>& compensationRatioValues() const { return compensation_values_; }
  const std::vector<torch::Tensor>& compensationFeatureTables() const { return compensation_features_; }
  int64_t compensationFeatureWidth() const { return compensationFeatures()?compensation_features_.front().size(1):0; }
  int compensationResidualMode() const { return compensation_residual_mode_; }
  void enableCompensationResiduals(int mode) {
    TORCH_CHECK(mode==1||mode==2,"Compensation residual mode must be cumulative or bandwidth");
    TORCH_CHECK(compensationFeatureWidth()==48&&compensation_values_.size()>=2&&compensation_values_.size()<=6,
                "Residual Features require two to six ratio48 tables");
    TORCH_CHECK(mode!=2||compensation_values_.size()==6,"Bandwidth Features require six ratio48 tables");
    if(compensation_residual_schema_.defined()) {
      TORCH_CHECK(compensation_residual_mode_==mode,"Compensation residual mode mismatch");return;
    }
    torch::NoGradGuard guard;
    compensation_residual_mode_=mode;
    compensation_residual_schema_=register_buffer("compensation_residual_schema",torch::tensor(mode,rows_.options()));
    for(size_t j=1;j<compensation_features_.size();++j)compensation_features_[j].zero_();
    if(mode==2) {
      auto options=features_.options().requires_grad(false);
      auto bands=torch::tensor({.25,.5,1.,1.5,2.},options).unsqueeze(1);
      auto fraction=(torch::arange(1,8,options)/7).unsqueeze(0);
      compensation_spatial_frequencies_=register_buffer("compensation_spatial_frequencies",bands*fraction*(2*M_PI));
    }
  }
  void enableCompensationFeatures(const std::vector<double>& knots,int64_t width=14) {
    TORCH_CHECK(explicitBlocks() && !compensationFeatures(),
                "Compensation ratio features require explicit Anchors");
    TORCH_CHECK(width==8 || width==14 || width==32 || width==48 || width==50,"Compensation feature width must be 8, 14, 32, 48 or 50");
    TORCH_CHECK(!ratioFeatures() || knots==ratio_values_,"Selector and compensation ratio knots must match");
    TORCH_CHECK(knots.size()>=2,"Compensation features need at least two ratios");
    for(size_t i=0;i<knots.size();++i)
      TORCH_CHECK(std::isfinite(knots[i]) && knots[i]>0 && knots[i]<=1 && (!i || knots[i]>knots[i-1]),
                  "Compensation ratio knots must increase in (0,1]");
    compensation_values_=knots;
    compensation_knots_=register_buffer("compensation_ratio_knots",torch::tensor(knots,torch::kFloat64).to(features_.device()));
    auto initial=torch::zeros({size(),width},features_.options());
    if(width==48&&size()) {ScopedTorchRNG rng(features_.device());initial.normal_(0,.01);}
    for(size_t i=0;i<knots.size();++i)
      compensation_features_.push_back(register_parameter("compensation_features_"+std::to_string(i),
          initial.clone()));
  }
  std::vector<double> compensationResidualWeights(double ratio) const {
    TORCH_CHECK(compensation_residual_mode_==1&&std::isfinite(ratio)&&ratio>0&&ratio<=1,"Residual48 fusion requires cumulative Features");
    std::vector<double> weights{1.};
    for(size_t j=1;j<compensation_values_.size();++j) {
      if(ratio<=compensation_values_[j-1]+1e-8)break;
      weights.push_back(std::min(1.,(ratio-compensation_values_[j-1])/(compensation_values_[j]-compensation_values_[j-1])));
    }
    return weights;
  }
  torch::Tensor compensationInput(const torch::Tensor& source_rows,double ratio,const torch::Tensor& local_offset={}) const {
    TORCH_CHECK(compensationFeatures() && std::isfinite(ratio) && ratio>0 && ratio<=1,
                "Invalid compensation feature query");
    const auto ids=source_rows.defined()?rows_.index_select(0,source_rows):rows_;
    if(compensation_residual_mode_) {
      if(compensation_residual_mode_==2)
        TORCH_CHECK(local_offset.defined()&&local_offset.sizes()==torch::IntArrayRef({ids.numel(),3}),"Bandwidth Features require aligned Anchor-unit offset3");
      auto value=compensation_features_[0].index_select(0,ids);
      torch::Tensor constant;
      for(size_t j=1;j<compensation_values_.size();++j) {
        if(ratio<=compensation_values_[j-1]+1e-8)break;
        const double weight=std::min(1.,(ratio-compensation_values_[j-1])/(compensation_values_[j]-compensation_values_[j-1]));
        auto residual=compensation_features_[j].index_select(0,ids);
        if(compensation_residual_mode_==2) {
          if(!constant.defined())constant=torch::ones({ids.numel(),6},local_offset.options());
          auto angle=local_offset.unsqueeze(2)*compensation_spatial_frequencies_.select(0,j-1);
          auto basis=torch::cat({constant,angle.sin().flatten(1),angle.cos().flatten(1)},1);
          residual=residual*basis;
        }
        value=value+weight*residual;
      }
      return value;
    }
    size_t hi=0;while(hi+1<compensation_values_.size() && ratio>compensation_values_[hi]+1e-8)++hi;
    auto value=compensation_features_[hi].index_select(0,ids);
    if(hi && std::abs(ratio-compensation_values_[hi])>1e-8 && ratio<compensation_values_.back()) {
      const double t=(ratio-compensation_values_[hi-1])/(compensation_values_[hi]-compensation_values_[hi-1]);
      value=(1-t)*compensation_features_[hi-1].index_select(0,ids)+t*value;
    }
    return value;
  }
  std::vector<torch::Tensor> featureTables() const {
    std::vector<torch::Tensor> result{features_};
    result.insert(result.end(), ratio_features_.begin(), ratio_features_.end());
    result.insert(result.end(), compensation_features_.begin(), compensation_features_.end());
    if(spatialFeatures())result.push_back(spatial_features_);
    return result;
  }
  torch::Tensor packedFeatures() const { return torch::cat(featureTables(),1); }
  void copyPackedFeatures(const torch::Tensor& values,int64_t start=0) {
    int64_t column=0;
    for(auto table:featureTables()) {
      table.slice(0,start,start+values.size(0)).copy_(values.slice(1,column,column+table.size(1)));
      column+=table.size(1);
    }
  }
  void enableRatioFeatures(const std::vector<double>& knots,int64_t shared_width=8) {
    TORCH_CHECK(explicitBlocks() && size()==0 && !ratioFeatures(), "Ratio features require an empty explicit Anchor bank");
    TORCH_CHECK(shared_width==8 || shared_width==16,"Ratio shared Feature width must be 8 or 16");
    TORCH_CHECK(knots.size()>=2, "Ratio features need at least two training ratios");
    for(size_t i=0;i<knots.size();++i)
      TORCH_CHECK(std::isfinite(knots[i]) && knots[i]>0 && knots[i]<=1 && (!i || knots[i]>knots[i-1]),
                  "Ratio feature knots must be strictly increasing in (0,1]");
    features_=torch::zeros({0,shared_width},features_.options()).requires_grad_(true);
    parameters_["features"]=features_;
    ratio_knots_=register_buffer("ratio_knots",torch::tensor(knots,torch::kFloat64).to(features_.device()));
    ratio_values_=knots;
    for(size_t i=0;i<knots.size();++i)
      ratio_features_.push_back(register_parameter("ratio_features_"+std::to_string(i),torch::zeros({0,8},features_.options())));
  }
  const torch::Tensor& rows() const { return rows_; }
  const torch::Tensor& centers() const { return centers_; }
  const torch::Tensor& offsetScales() const { return offset_scales_; }
  double voxelSize() const { return voxel_size_; }
  int64_t size() const { return features_.size(0); }
  bool explicitBlocks() const { return block_schema_.defined(); }
  bool adaptive() const { return levels_.defined() && !explicitBlocks(); }
  const torch::Tensor& levels() const { return levels_; }
  const torch::Tensor& branches() const { return branches_; }
  static constexpr int64_t kMaxLevel = 2, kLeafCapacity = 128;
  static constexpr int64_t kGaussiansPerBlock = 10;
  struct BlockSeeds { torch::Tensor xyz, color, sigma; };
  void enableExplicitBlocks();
  bool variableMembers() const { return member_counts_.defined(); }
  void enableVariableMembers();
  int fixedTenMode() const { return fixed_ten_mode_.defined()?fixed_ten_mode_.item<int>():0; }
  bool fixedTenMembers() const { return fixed_ten_mode_.defined(); }
  void enableFixedTenMembers(int mode) {
    TORCH_CHECK(variableMembers() && (mode==3 || mode==4),"Fixed-ten Anchors require Variable Anchor mode 3/4");
    TORCH_CHECK(!fixedTenMembers() || fixedTenMode()==mode,"Cannot change fixed-ten Anchor mode");
    TORCH_CHECK((member_counts_==10).all().item<bool>(),"Fixed-ten mode cannot convert partial Anchors; start a new map");
    if(mode==4)enableSlotDeactivation();
    TORCH_CHECK(slotDeactivation()==(mode==4),"Fixed-ten mode 3 disables slots; mode 4 enables slots");
    if(!fixedTenMembers())fixed_ten_mode_=register_buffer("fixed_ten_mode",torch::tensor(mode,rows_.options()));
  }
  const torch::Tensor& memberCounts() const { return member_counts_; }
  const torch::Tensor& topologyRound() const { return topology_round_; }
  BlockSeeds insertVariableBlocks(const std::map<std::array<int64_t,4>,std::vector<int64_t>>& groups,
      const torch::Tensor& xyz, const torch::Tensor& color, bool inherit,
      const std::vector<torch::optim::Adam*>& optimizers);
  void appendVariableMembers(const torch::Tensor& parents);
  torch::Tensor pruneVariableMembers(const torch::Tensor& survivors,
      const std::vector<torch::optim::Adam*>& optimizers);
  struct LocalMigration { torch::Tensor remove, changed; int64_t moved = 0, created = 0; };
  LocalMigration takeFixedMigration() {
    auto result=fixed_migration_;fixed_migration_=LocalMigration{};return result;
  }
  LocalMigration migrateLocalMembers(const torch::Tensor& xyz, const torch::Tensor& corrected,
      const std::vector<torch::optim::Adam*>& optimizers);
  BlockSeeds observeBlocks(const torch::Tensor& xyz, const torch::Tensor& color,
                          const std::vector<torch::optim::Adam*>& optimizers = {});
  BlockSeeds growBlocks(const torch::Tensor& xyz, const torch::Tensor& color,
      const torch::Tensor& gradient, const torch::Tensor& denom, double threshold, double alpha,
      const std::vector<torch::optim::Adam*>& optimizers = {}, bool refine_requires_full = false);
  torch::Tensor wholeBlockPrune(const torch::Tensor& mask) const;
  void pruneBlocks(const torch::Tensor& survivors, const std::vector<torch::optim::Adam*>& optimizers = {});
  void beginBlockGeometry();
  void transportBlocks(const GaussianTransformResult& correction);
  void finishBlockGeometry();
  void assertBlocks(int64_t population) const;
  void setFusedFullStats(bool enabled) { fused_full_stats_=enabled; }
  bool fusedFullStats() const { return fused_full_stats_; }
  const torch::Tensor& validatedMembers() const { return validated_members_; }
  const torch::Tensor& blockUIDs() const { return block_uids_; }
  bool slotDeactivation() const { return slot_inactive_.defined(); }
  const torch::Tensor& inactiveSlots() const { return slot_inactive_; }
  const torch::Tensor& recoveredSlots() const { return slot_recovered_; }
  void enableSlotDeactivation() {
    TORCH_CHECK(explicitBlocks(), "Slot deactivation requires explicit ten-slot anchors");
    if (slotDeactivation()) return;
    slot_inactive_ = register_buffer("slot_inactive", variableMembers()
        ? torch::zeros({rows_.numel()},rows_.options().dtype(torch::kBool))
        : torch::zeros({size(),10}, rows_.options().dtype(torch::kBool)));
    slot_recovered_ = register_buffer("slot_recovered", torch::zeros_like(slot_inactive_));
  }
  // Called before a real training forward/checkpoint, never on a trial candidate.
  void refreshSlotActivity(const torch::Tensor& opacity) {
    if (!slotDeactivation()) return;
    torch::NoGradGuard guard;
    auto recovered = slot_inactive_ & ~(opacity.detach().reshape_as(slot_inactive_) < 1.f/255.f);
    slot_recovered_.logical_or_(recovered);
    slot_inactive_.logical_and_(~recovered);
  }
  void closeSlotWindow(const torch::Tensor& enough, const torch::Tensor& visits,
      const torch::Tensor& below_cutoff, const torch::Tensor& opacity) {
    if (!slotDeactivation()) return;
    torch::NoGradGuard guard;
    refreshSlotActivity(opacity);
    const auto denominator=variableMembers()?visits.index_select(0,rows_):visits.unsqueeze(1);
    const auto complete=variableMembers()?enough.index_select(0,rows_):enough.unsqueeze(1);
    auto qualified = (below_cutoff == denominator) & ~slot_recovered_ &
        (opacity.detach().reshape_as(slot_inactive_) < 1.f/255.f);
    slot_inactive_.copy_(torch::where(complete, qualified, slot_inactive_));
    slot_recovered_.masked_fill_(complete, false);
  }
  // Pure query: prospective Guard states must not change live eligibility.
  torch::Tensor maskSlotProbability(const torch::Tensor& probability, const torch::Tensor& opacity,
      const torch::Tensor& source_rows = {}) const {
    if (!slotDeactivation()) return probability;
    const auto n=source_rows.defined()?source_rows.numel():rows_.numel();
    TORCH_CHECK(probability.numel()==n && opacity.numel()==n, "Slot mask population mismatch");
    auto inactive=slot_inactive_.reshape({-1});
    if(source_rows.defined())inactive=inactive.index_select(0,source_rows);
    inactive=inactive & (opacity.detach().reshape({-1}) < 1.f/255.f);
    return probability * (~inactive).to(probability.scalar_type());
  }
  bool cellLocalOffsets() const { return offset_scales_.defined(); }
  void enableCellLocalOffsets() {
    if (cellLocalOffsets()) return;
    offset_scales_ = register_buffer("offset_scales", torch::full({size(), 1}, voxel_size_, features_.options().requires_grad(false)));
    if (adaptive()) syncTreeBuffers();
  }

  // Parent nodes are retained as inheritance templates/history, never created
  // speculatively. v4 uses current-cell coordinates; older layouts keep L0 refs.
  void enableAdaptive() {
    TORCH_CHECK(!explicitBlocks(), "Explicit blocks cannot use occupancy refinement");
    if (adaptive()) return;
    levels_ = register_buffer("levels", torch::zeros({size()}, rows_.options()));
    branches_ = register_buffer("branches", torch::zeros({size()}, rows_.options().dtype(torch::kBool)));
    rebuildTreeLookup();
    refinement_pending_ = true;
  }

  void reset(const torch::Tensor& xyz, const std::vector<torch::optim::Adam*>& optimizers = {}) {
    TORCH_CHECK(!explicitBlocks(), "Explicit anchors must be initialized by the Gaussian map");
    torch::NoGradGuard guard;
    resizeFeatures(torch::empty({0}, rows_.options()), optimizers);
    keys_.set_data(torch::empty({0, 3}, keys_.options()));
    centers_.set_data(torch::empty({0, 3}, features_.options().requires_grad(false)));
    rows_.set_data(torch::empty({0}, rows_.options()));
    if (cellLocalOffsets()) offset_scales_.set_data(torch::empty({0,1}, features_.options().requires_grad(false)));
    lookup_.clear();
    if (adaptive()) {
      nodes_.clear(); tree_lookup_.clear(); split_.clear();
      syncTreeBuffers();
    }
    append(xyz, optimizers);
  }

  void append(const torch::Tensor& xyz, const std::vector<torch::optim::Adam*>& optimizers = {}) {
    TORCH_CHECK(!explicitBlocks(), "Explicit anchors require complete ten-slot map insertion");
    torch::NoGradGuard guard;
    TORCH_CHECK(xyz.dim() == 2 && xyz.size(1) == 3 && xyz.device() == features_.device(),
                "Anchor append requires aligned XYZ");
    if (!xyz.size(0)) return;
    if (adaptive()) {
      rows_.set_data(torch::cat({rows_, route(xyz, optimizers)}));
      refinement_pending_ = true;
      return;
    }
    const auto key_cpu = (xyz.detach().to(torch::kFloat64) / voxel_size_).floor().cpu().contiguous();
    TORCH_CHECK(torch::isfinite(key_cpu).all().item<bool>() &&
                (key_cpu.abs() < 9e18).all().item<bool>(), "Invalid anchor coordinate");
    const auto key_long = key_cpu.to(torch::kLong);
    const auto access = key_long.accessor<int64_t, 2>();
    std::vector<Key> keys(size());
    for (const auto& entry : lookup_) keys[entry.second] = entry.first;
    std::vector<int64_t> ids(xyz.size(0));
    const auto previous_size = size();
    for (int64_t i = 0; i < xyz.size(0); ++i) {
      Key key{access[i][0], access[i][1], access[i][2]};
      auto found = lookup_.find(key);
      if (found == lookup_.end()) {
        const auto id = static_cast<int64_t>(keys.size());
        found = lookup_.emplace(key, id).first;
        keys.push_back(key);
      }
      ids[i] = found->second;
    }
    if (static_cast<int64_t>(keys.size()) != previous_size) {
      auto old_ids = torch::cat({torch::arange(previous_size, rows_.options()),
          torch::full({static_cast<int64_t>(keys.size()) - previous_size}, -1, rows_.options())});
      resizeFeatures(old_ids, optimizers);
      setKeys(keys);
    }
    auto new_ids = torch::tensor(ids, torch::TensorOptions().dtype(torch::kLong)).to(rows_.device());
    rows_.set_data(torch::cat({rows_, new_ids}));
  }

  // Accepts survivor row indices, also covering an explicit row permutation.
  void selectRows(const torch::Tensor& indices, const std::vector<torch::optim::Adam*>& optimizers = {}) {
    TORCH_CHECK(!explicitBlocks(), "Explicit anchors require whole-block pruning");
    torch::NoGradGuard guard;
    rows_.set_data(rows_.index_select(0, indices));
    // Retain historical tree nodes so a pruned/revisited region cannot collapse
    // back to L0. Only occupied paths create new nodes on subsequent appends.
    if (adaptive()) return;
    auto occupancy = torch::zeros({size()}, rows_.options());
    occupancy.index_add_(0, rows_, torch::ones_like(rows_));
    auto alive = torch::nonzero(occupancy > 0).reshape({-1});
    if (alive.numel() == size()) return;
    auto remap = torch::full({size()}, -1, rows_.options());
    remap.index_copy_(0, alive, torch::arange(alive.numel(), rows_.options()));
    rows_.set_data(remap.index_select(0, rows_));
    const auto next_keys = keys_.index_select(0, alive).cpu().contiguous();
    auto access = next_keys.accessor<int64_t, 2>();
    std::vector<Key> keys(alive.numel());
    for (int64_t j = 0; j < alive.numel(); ++j) keys[j] = {access[j][0], access[j][1], access[j][2]};
    resizeFeatures(alive, optimizers);
    setKeys(keys);
  }

  // Geometry events rebuild associations once. Transport member features and
  // Adam moments, averaging only when formerly distinct anchors enter one voxel.
  void reassociate(const torch::Tensor& xyz, const std::vector<torch::optim::Adam*>& optimizers = {}) {
    TORCH_CHECK(!explicitBlocks(), "Explicit Gaussian owners must not be reassociated");
    torch::NoGradGuard guard;
    TORCH_CHECK(xyz.size(0) == rows_.numel(), "Anchor geometry population mismatch");
    const auto old_rows = rows_.clone();
    const auto old_features = features_.detach().clone();
    struct Moments { torch::optim::Adam* optimizer; torch::Tensor first, second, maximum; };
    std::vector<Moments> moments;
    for (auto* optimizer : optimizers) {
      if (!optimizer) continue;
      const auto found = optimizer->state().find(features_.unsafeGetTensorImpl());
      if (found == optimizer->state().end()) continue;
      auto& state = static_cast<torch::optim::AdamParamState&>(*found->second);
      moments.push_back({optimizer, state.exp_avg().clone(), state.exp_avg_sq().clone(),
          state.max_exp_avg_sq().defined() ? state.max_exp_avg_sq().clone() : torch::Tensor{}});
    }
    if (adaptive()) rows_.set_data(route(xyz, optimizers));
    else reset(xyz, optimizers);
    auto counts = torch::zeros({size()}, features_.options().requires_grad(false));
    counts.index_add_(0, rows_, torch::ones({rows_.numel()}, counts.options()));
    const auto transport = [&](const torch::Tensor& value, const torch::Tensor& fallback) {
      auto result = torch::zeros_like(features_);
      result.index_add_(0, rows_, value.index_select(0, old_rows));
      auto mean = result / counts.clamp_min(1).unsqueeze(1);
      return adaptive() ? torch::where((counts > 0).unsqueeze(1), mean, fallback) : mean;
    };
    features_.copy_(transport(old_features, features_.detach()));
    for (const auto& value : moments) {
      auto& state = static_cast<torch::optim::AdamParamState&>(
          *value.optimizer->state().at(features_.unsafeGetTensorImpl()));
      state.exp_avg(transport(value.first, state.exp_avg()));
      state.exp_avg_sq(transport(value.second, state.exp_avg_sq()));
      if (value.maximum.defined()) state.max_exp_avg_sq(transport(value.maximum, state.max_exp_avg_sq()));
    }
    refinement_pending_ = adaptive();
  }

  // Invoke only at a graph-free mapper boundary, under the render lock. The
  // supplied XYZ is the full explicit population; no geometry is modified.
  int64_t refine(const torch::Tensor& xyz, const std::vector<torch::optim::Adam*>& optimizers = {}) {
    if (!adaptive() || !refinement_pending_) return 0;
    torch::NoGradGuard guard;
    const auto start = std::chrono::steady_clock::now();
    TORCH_CHECK(xyz.size(0) == rows_.numel(), "Refinement population mismatch");
    auto coordinates = fineKeys(xyz);
    auto access = coordinates.accessor<int64_t, 2>();
    auto ids_cpu = rows_.cpu().contiguous();
    auto ids = ids_cpu.accessor<int64_t, 1>();
    std::vector<int64_t> sources(size());
    std::iota(sources.begin(), sources.end(), 0);
    const int64_t previous_size = size();
    int64_t splits = 0;
    for (int64_t level = 0; level < kMaxLevel; ++level) {
      std::map<int64_t, std::vector<int64_t>> members;
      for (int64_t i = 0; i < xyz.size(0); ++i)
        if (nodes_[ids[i]][0] == level && !split_[ids[i]]) members[ids[i]].push_back(i);
      for (const auto& entry : members) {
        if (static_cast<int64_t>(entry.second.size()) <= kLeafCapacity) continue;
        const auto parent = entry.first;
        split_[parent] = true; ++splits;
        for (auto i : entry.second) {
          // An explicit Gaussian may drift since its last geometry event.
          // Split its existing parent by the three midpoint planes.
          auto key = nodes_[parent]; key[0] = level + 1;
          const int64_t divisor = int64_t(1) << (kMaxLevel - level - 1);
          for (int c = 0; c < 3; ++c)
            key[c+1] = nodes_[parent][c+1] * 2 + (access[i][c] >= (nodes_[parent][c+1]*2+1)*divisor);
          ids[i] = ensureNode(key, sources[parent], sources);
        }
      }
    }
    if (size_t(previous_size) != nodes_.size()) {
      resizeFeatures(torch::tensor(sources, rows_.options().device(torch::kCPU)).to(rows_.device()), optimizers);
    }
    if (splits) {
      rows_.set_data(ids_cpu.to(rows_.device()));
      syncTreeBuffers();
    }
    refinement_pending_ = false;
    std::cout << "[Anchor Octree] splits=" << splits << " nodes=" << size()
              << " population=" << rows_.numel() << " refinement_ms="
              << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count() << '\n';
    return splits;
  }

  std::pair<torch::Tensor, torch::Tensor> inputs(const torch::Tensor& xyz,
                                               const torch::Tensor& source_rows = {}, double ratio = 0) const {
    const auto ids = source_rows.defined() ? rows_.index_select(0, source_rows) : rows_;
    TORCH_CHECK(xyz.size(0) == ids.numel(), "Anchor forward requires explicit source rows for compact inputs");
    auto local = xyz - centers_.index_select(0, ids);
    auto feature=features_.index_select(0,ids);
    if(ratioFeatures()) {
      TORCH_CHECK(std::isfinite(ratio) && ratio>0 && ratio<=1,"Invalid ratio feature query");
      size_t hi=0;while(hi+1<ratio_values_.size() && ratio>ratio_values_[hi]+1e-8)++hi;
      torch::Tensor specific;
      if(hi==0 || std::abs(ratio-ratio_values_[hi])<=1e-8 || ratio>=ratio_values_.back())
        specific=ratio_features_[hi].index_select(0,ids);
      else {
        const double t=(ratio-ratio_values_[hi-1])/(ratio_values_[hi]-ratio_values_[hi-1]);
        specific=(1-t)*ratio_features_[hi-1].index_select(0,ids)+t*ratio_features_[hi].index_select(0,ids);
      }
      feature=torch::cat({feature,specific},1);
    }
    return {feature, cellLocalOffsets()
        ? local / offset_scales_.index_select(0, ids) : local / voxel_size_};
  }

  void copyLayoutFrom(const AnchorBankImpl& source) {
    torch::NoGradGuard guard;
    if(source.ratioFeatures() && !ratioFeatures()) {
      enableExplicitBlocks();enableRatioFeatures(source.ratio_values_,source.features_.size(1));
    }
    TORCH_CHECK(source.ratioFeatures()==ratioFeatures(),"Anchor Feature schema mismatch");
    features_.set_data(torch::zeros_like(source.features_));
    for(size_t i=0;i<ratio_features_.size();++i)ratio_features_[i].set_data(torch::zeros_like(source.ratio_features_[i]));
    if(source.compensationFeatures() && !compensationFeatures())enableCompensationFeatures(source.compensation_values_,source.compensationFeatureWidth());
    if(source.compensationResidualMode())enableCompensationResiduals(source.compensationResidualMode());
    TORCH_CHECK(source.compensationResidualMode()==compensationResidualMode(),"Compensation residual layout mismatch");
    TORCH_CHECK(source.compensationFeatures()==compensationFeatures(),"Compensation feature schema mismatch");
    TORCH_CHECK(source.compensationFeatureWidth()==compensationFeatureWidth(),"Compensation feature width mismatch");
    for(size_t i=0;i<compensation_features_.size();++i)
      compensation_features_[i].set_data(torch::zeros_like(source.compensation_features_[i]));
    if(source.spatialFeatures() && !spatialFeatures())enableSpatialFeatures(source.spatial_features_.size(1));
    TORCH_CHECK(source.spatialFeatures()==spatialFeatures(),"Spatial Feature schema mismatch");
    if(spatialFeatures())TORCH_CHECK(spatial_features_.size(1)==source.spatial_features_.size(1),"Spatial Feature width mismatch");
    if(spatialFeatures())spatial_features_.set_data(torch::zeros_like(source.spatial_features_));
    centers_.set_data(source.centers_.clone());
    keys_.set_data(source.keys_.clone());
    rows_.set_data(source.rows_.clone());
    scale_.set_data(source.scale_.clone());
    lookup_ = source.lookup_;
    if (source.explicitBlocks()) {
      enableExplicitBlocks();
      levels_.set_data(source.levels_.clone()); branches_.set_data(source.branches_.clone());
      block_uids_.set_data(source.block_uids_.clone()); next_uid_.set_data(source.next_uid_.clone());
      if (source.variableMembers()) {
        enableVariableMembers();member_counts_.set_data(source.member_counts_.clone());
        topology_round_.copy_(source.topology_round_);
      }
      if (source.slotDeactivation()) {
        enableSlotDeactivation();
        slot_inactive_.set_data(source.slot_inactive_.clone());
        slot_recovered_.set_data(source.slot_recovered_.clone());
      }
      if(source.fixedTenMembers())enableFixedTenMembers(source.fixedTenMode());
    }
    if (source.adaptive()) {
      enableAdaptive();
      levels_.set_data(source.levels_.clone()); branches_.set_data(source.branches_.clone());
      nodes_ = source.nodes_; split_ = source.split_; tree_lookup_ = source.tree_lookup_;
      refinement_pending_ = source.refinement_pending_;
    }
    if (source.cellLocalOffsets()) {
      enableCellLocalOffsets();
      centers_.set_data(source.centers_.clone());
      offset_scales_.set_data(source.offset_scales_.clone());
    }
  }

  void load(torch::serialize::InputArchive& archive) override {
    torch::Tensor block_schema;
    if (archive.try_read("explicit_blocks", block_schema, true)) enableExplicitBlocks();
    torch::Tensor spatial_feature;
    if(archive.try_read("spatial_features",spatial_feature)) {
      TORCH_CHECK(spatial_feature.dim()==2&&(spatial_feature.size(1)==16||spatial_feature.size(1)==128),"Invalid Spatial Feature shape");
      enableSpatialFeatures(spatial_feature.size(1));
    }else TORCH_CHECK(!spatialFeatures(),"Checkpoint has no Early16 Feature");
    torch::Tensor knots;
    if(archive.try_read("ratio_knots",knots,true)) {
      const auto cpu=knots.to(torch::kCPU,torch::kFloat64).contiguous();
      std::vector<double> values(cpu.data_ptr<double>(),cpu.data_ptr<double>()+cpu.numel());
      torch::Tensor saved_features;archive.read("features",saved_features);
      if(!ratioFeatures())enableRatioFeatures(values,saved_features.size(1));
      TORCH_CHECK(features_.size(1)==saved_features.size(1),"Ratio shared Feature width mismatch");
      TORCH_CHECK(values==ratio_values_,"Ratio feature checkpoint knots mismatch");
    } else TORCH_CHECK(!ratioFeatures(),"Cannot load legacy features into ratio-feature layout");
    torch::Tensor compensation_knots;
    if(archive.try_read("compensation_ratio_knots",compensation_knots,true)) {
      const auto cpu=compensation_knots.to(torch::kCPU,torch::kFloat64).contiguous();
      std::vector<double> values(cpu.data_ptr<double>(),cpu.data_ptr<double>()+cpu.numel());
      torch::Tensor table;archive.read("compensation_features_0",table);
      if(!compensationFeatures())enableCompensationFeatures(values,table.size(1));
      TORCH_CHECK(values==compensation_values_,"Compensation feature knots mismatch");
      TORCH_CHECK(table.size(1)==compensationFeatureWidth(),"Compensation feature width mismatch");
    } else TORCH_CHECK(!compensationFeatures(),"Checkpoint has no Anchor compensation features");
    torch::Tensor residual_schema;
    if(archive.try_read("compensation_residual_schema",residual_schema,true))enableCompensationResiduals(residual_schema.item<int64_t>());
    else TORCH_CHECK(!compensation_residual_mode_,"Checkpoint has no residual Feature schema");
    torch::Tensor counts;
    if (archive.try_read("member_counts", counts, true)) enableVariableMembers();
    torch::Tensor fixed_mode;
    if(archive.try_read("fixed_ten_mode",fixed_mode,true)) {
      const auto mode=fixed_mode.item<int>();
      TORCH_CHECK(variableMembers() && (mode==3 || mode==4),"Invalid fixed-ten checkpoint mode");
      TORCH_CHECK(!fixedTenMembers() || fixedTenMode()==mode,"Fixed-ten checkpoint mode mismatch");
      if(!fixedTenMembers())fixed_ten_mode_=register_buffer("fixed_ten_mode",torch::tensor(mode,rows_.options()));
    } else TORCH_CHECK(!fixedTenMembers(),"Fixed-ten mode requires a mode 3/4 checkpoint; legacy conversion is not supported");
    torch::Tensor inactive;
    if (archive.try_read("slot_inactive", inactive, true)) enableSlotDeactivation();
    torch::Tensor levels;
    if (!explicitBlocks() && archive.try_read("levels", levels, true)) enableAdaptive();
    torch::Tensor offsets;
    if (archive.try_read("offset_scales", offsets, true)) enableCellLocalOffsets();
    torch::nn::Module::load(archive);
    if(fixedTenMembers())TORCH_CHECK(slotDeactivation()==(fixedTenMode()==4),"Fixed-ten checkpoint slot mode mismatch");
    voxel_size_ = scale_.item<double>();
    const auto cpu = keys_.cpu().contiguous();
    const auto access = cpu.accessor<int64_t, 2>();
    lookup_.clear();
    for (int64_t j = 0; j < size(); ++j) lookup_.emplace(Key{access[j][0], access[j][1], access[j][2]}, j);
    if (adaptive()) { rebuildTreeLookup(); refinement_pending_ = true; }
    if (explicitBlocks()) { rebuildBlockLookup(); assertBlocks(rows_.numel()); }
  }

 private:
  using NodeKey = std::array<int64_t, 4>; // level, integer XYZ at that level
  BlockSeeds insertBlocks(const std::map<NodeKey, std::vector<int64_t>>& groups,
      const torch::Tensor& xyz, const torch::Tensor& color, bool inherit,
      const std::vector<torch::optim::Adam*>& optimizers);
  void rebuildBlockLookup();
  void prepareFixedMigration();
  static int64_t floorDivide(int64_t value, int64_t divisor) {
    return value / divisor - (value % divisor < 0 ? 1 : 0);
  }
  static NodeKey nodeKey(const torch::TensorAccessor<int64_t, 1>& fine, int64_t level) {
    const int64_t divisor = int64_t(1) << (kMaxLevel - level);
    return {level, floorDivide(fine[0], divisor), floorDivide(fine[1], divisor), floorDivide(fine[2], divisor)};
  }
  torch::Tensor fineKeys(const torch::Tensor& xyz) const {
    auto value = (xyz.detach().to(torch::kFloat64) / (voxel_size_ / 4)).floor().cpu().contiguous();
    TORCH_CHECK(torch::isfinite(value).all().item<bool>() && (value.abs() < 9e18).all().item<bool>(),
                "Invalid octree coordinate");
    return value.to(torch::kLong);
  }
  int64_t ensureNode(const NodeKey& key, int64_t source, std::vector<int64_t>& sources) {
    const auto found = tree_lookup_.find(key);
    if (found != tree_lookup_.end()) return found->second;
    const int64_t id = nodes_.size();
    tree_lookup_.emplace(key, id); nodes_.push_back(key); split_.push_back(false); sources.push_back(source);
    return id;
  }
  torch::Tensor route(const torch::Tensor& xyz, const std::vector<torch::optim::Adam*>& optimizers) {
    auto coordinates = fineKeys(xyz);
    auto access = coordinates.accessor<int64_t, 2>();
    std::vector<int64_t> sources(size()), ids(xyz.size(0));
    std::iota(sources.begin(), sources.end(), 0);
    for (int64_t i = 0; i < xyz.size(0); ++i) {
      auto id = ensureNode(nodeKey(access[i], 0), -1, sources);
      for (int64_t level = 1; level <= kMaxLevel && split_[id]; ++level)
        id = ensureNode(nodeKey(access[i], level), sources[id], sources);
      ids[i] = id;
    }
    if (sources.size() != size_t(size())) {
      resizeFeatures(torch::tensor(sources, rows_.options().device(torch::kCPU)).to(rows_.device()), optimizers);
      syncTreeBuffers();
    }
    return torch::tensor(ids, rows_.options().device(torch::kCPU)).to(rows_.device());
  }
  void syncTreeBuffers() {
    auto keys = torch::empty({int64_t(nodes_.size()), 3}, torch::kLong);
    auto levels = torch::empty({int64_t(nodes_.size())}, torch::kLong);
    auto branches = torch::empty({int64_t(nodes_.size())}, torch::kBool);
    auto ka = keys.accessor<int64_t, 2>();
    auto la = levels.accessor<int64_t, 1>(); auto ba = branches.accessor<bool, 1>();
    for (size_t j = 0; j < nodes_.size(); ++j) {
      la[j] = nodes_[j][0]; ba[j] = split_[j];
      for (int c = 0; c < 3; ++c) ka[j][c] = nodes_[j][c+1];
    }
    keys_.set_data(keys.to(rows_.device())); levels_.set_data(levels.to(rows_.device()));
    branches_.set_data(branches.to(rows_.device()));
    if (cellLocalOffsets()) {
      const auto cell_sizes = voxel_size_ / torch::pow(2.0, levels_.to(features_.scalar_type()));
      offset_scales_.set_data(cell_sizes.unsqueeze(1));
      centers_.set_data((keys_.to(features_.scalar_type()) + 0.5) * offset_scales_);
      return;
    }
    // Derive the identical L0 reference, not the child center/size.
    auto root_keys = keys.clone(); auto ra = root_keys.accessor<int64_t, 2>();
    for (size_t j = 0; j < nodes_.size(); ++j)
      for (int c = 0; c < 3; ++c) ra[j][c] = floorDivide(ra[j][c], int64_t(1) << la[j]);
    centers_.set_data((root_keys.to(features_.device(), features_.scalar_type()) + 0.5) * voxel_size_);
  }
  void rebuildTreeLookup() {
    auto keys = keys_.cpu().contiguous(), levels = levels_.cpu().contiguous(), branches = branches_.cpu().contiguous();
    auto ka = keys.accessor<int64_t, 2>(); auto la = levels.accessor<int64_t, 1>(); auto ba = branches.accessor<bool, 1>();
    nodes_.resize(size()); split_.resize(size()); tree_lookup_.clear();
    for (int64_t j = 0; j < size(); ++j) {
      nodes_[j] = {la[j], ka[j][0], ka[j][1], ka[j][2]}; split_[j] = ba[j];
      tree_lookup_.emplace(nodes_[j], j);
    }
  }
  void setKeys(const std::vector<Key>& keys) {
    auto cpu = torch::empty({static_cast<int64_t>(keys.size()), 3}, torch::kLong);
    auto access = cpu.accessor<int64_t, 2>();
    lookup_.clear();
    for (int64_t j = 0; j < static_cast<int64_t>(keys.size()); ++j) {
      lookup_.emplace(keys[j], j);
      for (int c = 0; c < 3; ++c) access[j][c] = keys[j][c];
    }
    keys_.set_data(cpu.to(rows_.device()));
    centers_.set_data((keys_.to(features_.scalar_type()) + 0.5) * voxel_size_);
    if (cellLocalOffsets())
      offset_scales_.set_data(torch::full({size(), 1}, voxel_size_, features_.options().requires_grad(false)));
  }

  void initializeCompensationFeatureRows(torch::Tensor& packed) const {
    if(compensationFeatureWidth()!=48||!packed.size(0))return;
    ScopedTorchRNG rng(packed.device());
    auto initial=torch::randn({packed.size(0),48},packed.options())*.01;
    int64_t column=features_.size(1);
    for(const auto& table:ratio_features_)column+=table.size(1);
    for(size_t i=0;i<(compensation_residual_mode_?size_t(1):compensation_features_.size());++i)packed.slice(1,column+48*i,column+48*(i+1)).copy_(initial);
  }
  void resizeFeatures(const torch::Tensor& old_ids, const std::vector<torch::optim::Adam*>& optimizers) {
    MappingAuditRange trace("topology/anchor_feature_moments");
    resizeFeatureTable(features_,"features",old_ids,optimizers);
    for(size_t i=0;i<ratio_features_.size();++i)
      resizeFeatureTable(ratio_features_[i],"ratio_features_"+std::to_string(i),old_ids,optimizers);
    for(size_t i=0;i<compensation_features_.size();++i)
      resizeFeatureTable(compensation_features_[i],"compensation_features_"+std::to_string(i),old_ids,optimizers);
    if(spatialFeatures())resizeFeatureTable(spatial_features_,"spatial_features",old_ids,optimizers);
  }
  void resizeFeatureTable(torch::Tensor& table,const std::string& name,const torch::Tensor& old_ids,
      const std::vector<torch::optim::Adam*>& optimizers) {
    // Replace the leaf as GaussianModel does: resizing only .data leaves an old
    // AccumulateGrad shape after a prior backward. Update every optimizer alias.
    const auto gather = [&](const torch::Tensor& t) {
      auto padded = torch::cat({t.detach(), torch::zeros({1, t.size(1)}, t.options())});
      return padded.index_select(0, torch::where(old_ids >= 0, old_ids, table.size(0)));
    };
    auto next = gather(table).requires_grad_(table.requires_grad());
    for (auto* optimizer : optimizers) {
      if (!optimizer) continue;
      for (auto& group : optimizer->param_groups())
        for (auto& parameter : group.params())
          if (parameter.is_same(table)) parameter = next;
      auto found = optimizer->state().find(table.unsafeGetTensorImpl());
      if (found == optimizer->state().end()) continue;
      auto& state = static_cast<torch::optim::AdamParamState&>(*found->second);
      state.exp_avg(gather(state.exp_avg()));
      state.exp_avg_sq(gather(state.exp_avg_sq()));
      if (state.max_exp_avg_sq().defined()) state.max_exp_avg_sq(gather(state.max_exp_avg_sq()));
      auto migrated = std::move(found->second);
      optimizer->state().erase(found);
      optimizer->state()[next.unsafeGetTensorImpl()] = std::move(migrated);
    }
    table = next;
    parameters_[name] = table;
  }
  double voxel_size_;
  torch::Tensor features_, centers_, rows_, keys_, scale_;
  torch::Tensor ratio_knots_;
  std::vector<double> ratio_values_;
  std::vector<torch::Tensor> ratio_features_;
  torch::Tensor compensation_knots_;
  torch::Tensor compensation_residual_schema_,compensation_spatial_frequencies_;
  int compensation_residual_mode_=0;
  torch::Tensor spatial_features_;
  std::vector<double> compensation_values_;
  std::vector<torch::Tensor> compensation_features_;
  std::map<Key, int64_t> lookup_;
  torch::Tensor levels_, branches_;
  torch::Tensor offset_scales_;
  std::vector<NodeKey> nodes_;
  std::vector<bool> split_;
  std::map<NodeKey, int64_t> tree_lookup_;
  bool refinement_pending_ = false;
  torch::Tensor block_schema_, block_uids_, next_uid_;
  torch::Tensor member_counts_, topology_round_;
  torch::Tensor fixed_ten_mode_, geometry_keys_, geometry_touched_;
  LocalMigration fixed_migration_;
  torch::Tensor slot_inactive_, slot_recovered_;
  bool fused_full_stats_ = false;
  mutable torch::Tensor validated_members_;
  torch::Tensor geometry_centers_, geometry_scales_, geometry_delta_, geometry_scale_delta_;
  std::map<NodeKey, std::vector<int64_t>> block_lookup_;
};
TORCH_MODULE(AnchorBank);

}  // namespace improvements::selector

#include "explicit_anchor_blocks.h"
#include "variable_anchor_blocks.h"
