#pragma once

#include <algorithm>
#include <limits>
#include <set>

namespace improvements::selector {

inline void AnchorBankImpl::enableExplicitBlocks() {
  if (explicitBlocks()) return;
  TORCH_CHECK(!adaptive(), "Explicit blocks cannot reuse an occupancy tree");
  block_schema_ = register_buffer("explicit_blocks", torch::tensor(1, rows_.options()));
  block_uids_ = register_buffer("block_uids", torch::arange(size(), rows_.options()));
  next_uid_ = register_buffer("next_uid", torch::tensor(size(), rows_.options()));
  levels_ = register_buffer("levels", torch::zeros({size()}, rows_.options()));
  branches_ = register_buffer("branches", torch::zeros({size()}, rows_.options().dtype(torch::kBool)));
  enableCellLocalOffsets();
}

inline void AnchorBankImpl::assertBlocks(int64_t population) const {
  if (variableMembers()) {
    validated_members_=torch::Tensor();
    TORCH_CHECK(explicitBlocks() && rows_.numel()==population && member_counts_.numel()==size(),"Variable Anchor population mismatch");
    if(fixedTenMembers())TORCH_CHECK(population==size()*10 && (member_counts_==10).all().item<bool>(),
        "Fixed-ten Anchor population must contain exactly ten members per Anchor");
#ifdef CARTGS_FUSED_MAPPING_CUDA
    if(fused_full_stats_ && rows_.is_cuda() && centers_.scalar_type()==torch::kFloat32 && rows_.is_contiguous() &&
        member_counts_.is_contiguous() && centers_.is_contiguous() && offset_scales_.is_contiguous() && levels_.is_contiguous()) {
      TORCH_CHECK(levels_.numel()==size() && block_uids_.numel()==size(),"Invalid variable Anchor layout");
      validated_members_=mapping_cuda::validateAnchors(rows_,member_counts_,levels_,centers_,offset_scales_);
      return;
    }
#endif
    TORCH_CHECK((rows_>=0).all().item<bool>() && (rows_<size()).all().item<bool>(),"Invalid Gaussian owner");
    auto counts=torch::zeros_like(member_counts_);
    counts.index_add_(0,rows_,torch::ones_like(rows_));
    TORCH_CHECK(torch::equal(counts,member_counts_) && ((counts>=1)&(counts<=10)).all().item<bool>(),"Anchor capacity must be 1..10");
    TORCH_CHECK(levels_.numel()==size() && block_uids_.numel()==size() &&
        ((levels_>=0)&(levels_<=kMaxLevel)).all().item<bool>() && torch::isfinite(centers_).all().item<bool>() &&
        torch::isfinite(offset_scales_).all().item<bool>() && (offset_scales_>0).all().item<bool>(),"Invalid variable Anchor layout");
    return;
  }
  TORCH_CHECK(explicitBlocks() && population == size()*kGaussiansPerBlock && rows_.numel()==population,
              "Explicit anchor population must equal ten times the anchor count");
  TORCH_CHECK(torch::equal(rows_, torch::arange(size(), rows_.options()).repeat_interleave(kGaussiansPerBlock)),
              "Explicit anchor slots must retain block ownership");
  TORCH_CHECK(levels_.numel()==size() && block_uids_.numel()==size() &&
      ((levels_>=0)&(levels_<=kMaxLevel)).all().item<bool>() &&
      torch::isfinite(centers_).all().item<bool>() &&
      torch::isfinite(offset_scales_).all().item<bool>() && (offset_scales_>0).all().item<bool>(),
      "Invalid explicit anchor layout");
}

inline void AnchorBankImpl::rebuildBlockLookup() {
  MappingAuditRange trace("topology/spatial_lookup_rebuild");
  block_lookup_.clear();
  auto keys=keys_.cpu().contiguous(), levels=levels_.cpu().contiguous();
  auto k=keys.accessor<int64_t,2>(); auto l=levels.accessor<int64_t,1>();
  for(int64_t j=0;j<size();++j) block_lookup_[{l[j],k[j][0],k[j][1],k[j][2]}].push_back(j);
}

inline AnchorBankImpl::BlockSeeds AnchorBankImpl::observeBlocks(
    const torch::Tensor& xyz, const torch::Tensor& color,
    const std::vector<torch::optim::Adam*>& optimizers) {
  TORCH_CHECK(explicitBlocks(), "Explicit map is not enabled");
  auto points=xyz.detach().to(torch::kCPU,torch::kFloat64).contiguous();
  TORCH_CHECK(points.dim()==2 && points.size(1)==3 && color.sizes()==xyz.sizes() &&
      torch::isfinite(points).all().item<bool>() && torch::isfinite(color).all().item<bool>(), "Invalid observed points");
  auto p=points.accessor<double,2>();
  std::map<NodeKey,std::vector<int64_t>> groups;
  for(int64_t i=0;i<points.size(0);++i) for(int64_t level=0;level<kMaxLevel;++level) {
    double h=voxel_size_/double(int64_t(1)<<level);
    NodeKey key{level,int64_t(std::floor(p[i][0]/h)),int64_t(std::floor(p[i][1]/h)),int64_t(std::floor(p[i][2]/h))};
    if(!block_lookup_.count(key)) groups[key].push_back(i);
  }
  return insertBlocks(groups,xyz,color,false,optimizers);
}

inline AnchorBankImpl::BlockSeeds AnchorBankImpl::growBlocks(
    const torch::Tensor& xyz, const torch::Tensor& color, const torch::Tensor& gradient,
    const torch::Tensor& denom, double threshold, double alpha,
    const std::vector<torch::optim::Adam*>& optimizers, bool refine_requires_full) {
  assertBlocks(xyz.size(0));
  TORCH_CHECK(std::isfinite(threshold)&&threshold>0&&std::isfinite(alpha)&&alpha>1,
              "Invalid explicit anchor growth thresholds");
  auto points=xyz.detach().to(torch::kCPU,torch::kFloat64).contiguous();
  auto gradients=gradient.reshape({-1}).to(torch::kCPU,torch::kFloat64).contiguous();
  auto counts=denom.reshape({-1}).to(torch::kCPU,torch::kFloat64).contiguous();
  auto levels=levels_.cpu().contiguous();
  auto owners=rows_.cpu().contiguous();auto owner_rows=owners.accessor<int64_t,1>();
  auto members=variableMembers()?member_counts_.cpu().contiguous():torch::full({size()},10,owners.options());
  auto member_count=members.accessor<int64_t,1>();
  auto p=points.accessor<double,2>(); auto g=gradients.accessor<double,1>();
  auto n=counts.accessor<double,1>(); auto l=levels.accessor<int64_t,1>();
  std::map<NodeKey,std::vector<int64_t>> groups;
  int64_t limited=0,same=0,fine=0;
  int64_t fine_blocked=0,fine_full=0,fine_partial=0;
  for(int64_t i=0;i<xyz.size(0);++i) {
    if(n[i]<=0||!std::isfinite(g[i])) continue;
    int64_t owner=variableMembers()?owner_rows[i]:i/kGaussiansPerBlock;
    int64_t level=l[owner];
    double low=threshold*std::pow(alpha,level), high=low*alpha;
    if(g[i]<low) continue;
    if(g[i]>=high) {
      if(level==kMaxLevel) { ++limited; continue; }
      ++level;
    }
    double h=voxel_size_/double(int64_t(1)<<level);
    NodeKey key{level,int64_t(std::floor(p[i][0]/h)),int64_t(std::floor(p[i][1]/h)),int64_t(std::floor(p[i][2]/h))};
    if(!block_lookup_.count(key)) {
      if(level!=l[owner]) {
        if(refine_requires_full && member_count[owner]<kGaussiansPerBlock) {++fine_blocked;continue;}
        (member_count[owner]==kGaussiansPerBlock?fine_full:fine_partial)++;
      }
      groups[key].push_back(i); (level==l[owner]?same:fine)++;
    }
  }
  std::cout<<"[Explicit Anchor Growing] tau0="<<threshold<<" alpha="<<alpha
           <<" same_candidates="<<same<<" fine_candidates="<<fine<<" max_level_limited="<<limited
           <<" new_anchors="<<groups.size()<<" refine_requires_full="<<refine_requires_full
           <<" fine_blocked_not_full="<<fine_blocked<<" fine_full_candidates="<<fine_full
           <<" fine_partial_candidates="<<fine_partial<<'\n';
  return insertBlocks(groups,xyz,color,true,optimizers);
}

inline AnchorBankImpl::BlockSeeds AnchorBankImpl::insertBlocks(
    const std::map<NodeKey,std::vector<int64_t>>& groups,
    const torch::Tensor& xyz,const torch::Tensor& color,bool inherit,
    const std::vector<torch::optim::Adam*>& optimizers) {
  torch::NoGradGuard guard;
  if(groups.empty()) return {torch::empty({0,3},xyz.options()),torch::empty({0,3},color.options()),torch::empty({0,3},xyz.options())};
  if(variableMembers()) return insertVariableBlocks(groups,xyz,color,inherit,optimizers);
  const auto started=std::chrono::steady_clock::now();
  auto points=xyz.detach().to(torch::kCPU,torch::kFloat64).contiguous();
  auto colors=color.detach().to(torch::kCPU,torch::kFloat64).contiguous();
  auto old_features=packedFeatures().detach().to(torch::kCPU).contiguous();
  auto old_levels=levels_.cpu().contiguous();
  auto p=points.accessor<double,2>(), c=colors.accessor<double,2>();
  const int64_t old_size=size(), added=groups.size();
  auto seeds=torch::empty({added*kGaussiansPerBlock,3},torch::kFloat64);
  auto rgb=torch::empty_like(seeds), sigma=torch::empty_like(seeds);
  auto key_tensor=torch::empty({added,3},torch::kLong), level_tensor=torch::empty({added},torch::kLong);
  auto centers=torch::empty({added,3},torch::kFloat64), sizes=torch::empty({added,1},torch::kFloat64);
  auto new_features=torch::zeros({added,old_features.size(1)},old_features.options());
  initializeCompensationFeatureRows(new_features);
  if(spatialFeatures()&&spatial_features_.size(1)==128)new_features.slice(1,new_features.size(1)-128).normal_(0,.01);
  auto out=seeds.accessor<double,2>(), col=rgb.accessor<double,2>(), sig=sigma.accessor<double,2>();
  auto keys=key_tensor.accessor<int64_t,2>(); auto levels=level_tensor.accessor<int64_t,1>();
  auto ctr=centers.accessor<double,2>(), sz=sizes.accessor<double,2>();
  // Ten distinct bounded directions, including a center sample. No RNG consumption.
  const double directions[10][3]={{0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},
      {0,0,1},{0,0,-1},{.7,.7,.7},{-.7,.7,-.7},{.7,-.7,-.7}};
  int64_t j=0;
  for(const auto& entry:groups) {
    const auto& key=entry.first; double h=voxel_size_/double(int64_t(1)<<key[0]);
    levels[j]=key[0];sz[j][0]=h;
    for(int a=0;a<3;++a) {keys[j][a]=key[a+1];ctr[j][a]=(key[a+1]+.5)*h;}
    std::vector<int64_t> distinct;
    std::set<std::array<double,3>> seen;
    for(auto i:entry.second) if(seen.insert({p[i][0],p[i][1],p[i][2]}).second) distinct.push_back(i);
    std::vector<int64_t> chosen;
    if(distinct.size()>=kGaussiansPerBlock) {
      chosen.push_back(distinct.front());
      std::vector<double> distance(distinct.size(),std::numeric_limits<double>::infinity());
      for(int k=1;k<kGaussiansPerBlock;++k) {
        double maximum=-1;int64_t best=0;
        for(size_t b=0;b<distinct.size();++b) {
          double d=0;for(int a=0;a<3;++a) d+=std::pow(p[distinct[b]][a]-p[chosen.back()][a],2);
          distance[b]=std::min(distance[b],d);
          if(distance[b]>maximum){maximum=distance[b];best=distinct[b];}
        }
        chosen.push_back(best);
      }
    }
    for(int k=0;k<kGaussiansPerBlock;++k) {
      auto i=chosen.empty()?distinct[k%distinct.size()]:chosen[k];
      double nearest=std::numeric_limits<double>::infinity();
      for(auto other:distinct) if(other!=i) {
        double d=0;for(int a=0;a<3;++a)d+=std::pow(p[i][a]-p[other][a],2);
        nearest=std::min(nearest,d);
      }
      double initial_sigma=std::isfinite(nearest)?std::sqrt(nearest)*.1:h*.05;
      initial_sigma=std::clamp(initial_sigma,h*.001,h*.25);
      for(int a=0;a<3;++a) {
        double x=p[i][a];
        if(chosen.empty()) {
          double radius=h*.025, lower=key[a+1]*h;
          x=std::clamp(x,lower+radius,lower+h-radius)+radius*directions[k][a];
        }
        out[j*kGaussiansPerBlock+k][a]=x;
        col[j*kGaussiansPerBlock+k][a]=c[i][a];
        sig[j*kGaussiansPerBlock+k][a]=initial_sigma;
      }
    }
    if(inherit) {
      std::vector<int64_t> sources;
      bool fine_source=false;
      for(auto i:entry.second) {
        if(old_levels.accessor<int64_t,1>()[i/kGaussiansPerBlock]==key[0]) sources.push_back(i/kGaussiansPerBlock);
        else fine_source=true;
      }
      // Match coarse-to-fine insertion: a new fine-level key is initialized
      // at zero before the next source level can propose that same key.
      if(!fine_source&&!sources.empty()) new_features[j].copy_(std::get<0>(old_features.index_select(0,torch::tensor(sources,torch::kLong)).max(0)));
    }
    block_lookup_[key].push_back(old_size+j);
    ++j;
  }
  resizeFeatures(torch::cat({torch::arange(old_size,rows_.options()),torch::full({added},-1,rows_.options())}),optimizers);
  copyPackedFeatures(new_features.to(features_.device()),old_size);
  centers_.set_data(torch::cat({centers_,centers.to(centers_.options())}));
  offset_scales_.set_data(torch::cat({offset_scales_,sizes.to(offset_scales_.options())}));
  keys_.set_data(torch::cat({keys_,key_tensor.to(keys_.device())}));
  levels_.set_data(torch::cat({levels_,level_tensor.to(levels_.device())}));
  branches_.set_data(torch::zeros({size()},branches_.options()));
  const auto next=next_uid_.item<int64_t>();
  block_uids_.set_data(torch::cat({block_uids_,torch::arange(next,next+added,rows_.options())}));
  next_uid_.fill_(next+added);
  if (slotDeactivation()) {
    slot_inactive_.set_data(torch::cat({slot_inactive_,torch::zeros({added,10},slot_inactive_.options())}));
    slot_recovered_.set_data(torch::cat({slot_recovered_,torch::zeros({added,10},slot_recovered_.options())}));
  }
  rows_.set_data(torch::arange(size(),rows_.options()).repeat_interleave(kGaussiansPerBlock));
  if(geometry_centers_.defined()) {
    geometry_centers_=torch::cat({geometry_centers_,centers.to(centers_.options())});
    geometry_scales_=torch::cat({geometry_scales_,sizes.to(offset_scales_.options())});
    geometry_delta_=torch::cat({geometry_delta_,torch::zeros({added,3},centers_.options())});
    geometry_scale_delta_=torch::cat({geometry_scale_delta_,torch::zeros({added,1},offset_scales_.options())});
  }
  assertBlocks(rows_.numel());
  std::cout<<"[Explicit Anchor Insert] added="<<added<<" anchors="<<size()<<" gaussians="<<rows_.numel()
           <<" maintenance_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()<<'\n';
  return {seeds.to(xyz.options()),rgb.to(color.options()),sigma.to(xyz.options())};
}

inline torch::Tensor AnchorBankImpl::wholeBlockPrune(const torch::Tensor& mask) const {
  assertBlocks(mask.numel());
  return mask.reshape({size(),kGaussiansPerBlock}).all(1).repeat_interleave(kGaussiansPerBlock);
}

inline void AnchorBankImpl::pruneBlocks(const torch::Tensor& survivors,
    const std::vector<torch::optim::Adam*>& optimizers) {
  torch::NoGradGuard guard;
  TORCH_CHECK(!geometry_centers_.defined(),"Cannot prune during a geometry/CD event");
  auto keep=survivors.reshape({size(),kGaussiansPerBlock});
  TORCH_CHECK(torch::equal(keep,keep.slice(1,0,1).expand_as(keep)),"Partial explicit anchor pruning");
  auto ids=torch::nonzero(keep.select(1,0)).reshape({-1});
  const auto removed=size()-ids.numel();
  resizeFeatures(ids,optimizers);
  for(auto* t:{&centers_,&offset_scales_,&keys_,&levels_,&branches_,&block_uids_}) t->set_data(t->index_select(0,ids));
  if (slotDeactivation()) {
    slot_inactive_.set_data(slot_inactive_.index_select(0,ids));
    slot_recovered_.set_data(slot_recovered_.index_select(0,ids));
  }
  rows_.set_data(torch::arange(size(),rows_.options()).repeat_interleave(kGaussiansPerBlock));
  rebuildBlockLookup();assertBlocks(rows_.numel());
  std::cout<<"[Explicit Anchor Prune] removed="<<removed<<" anchors="<<size()<<" gaussians="<<rows_.numel()<<'\n';
}

inline void AnchorBankImpl::beginBlockGeometry() {
  TORCH_CHECK(!geometry_centers_.defined(),"Nested explicit anchor geometry event");
  if(fixedTenMembers()) {
    TORCH_CHECK(!fixed_migration_.remove.defined(),"Previous fixed-ten geometry migration was not completed");
    geometry_keys_=keys_.clone();geometry_touched_=torch::zeros({size()},rows_.options().dtype(torch::kBool));
  }
  geometry_centers_=centers_.detach().clone();geometry_scales_=offset_scales_.clone();
  geometry_delta_=torch::zeros_like(geometry_centers_);geometry_scale_delta_=torch::zeros_like(geometry_scales_);
}

inline void AnchorBankImpl::transportBlocks(const GaussianTransformResult& correction) {
  torch::NoGradGuard guard;
  if(!geometry_centers_.defined()) beginBlockGeometry();
  auto ids=correction.affects_all?torch::arange(correction.population_count,rows_.options()):correction.indices;
  if(!ids.defined()||!ids.numel())return;
  auto count=torch::zeros({size()},centers_.options());
  count.index_add_(0,rows_.index_select(0,ids),torch::ones({ids.numel()},count.options()));
  if(fixedTenMembers())geometry_touched_.logical_or_(count>0);
  auto fraction=count.unsqueeze(1)/(variableMembers()?member_counts_.to(count.scalar_type()).unsqueeze(1):
      torch::full({size(),1},kGaussiansPerBlock,count.options()));
  if(correction.changed_attributes&toBits(GaussianAttrFlag::XYZ)) {
    const auto& m=correction.transform_4x4;
    auto moved=torch::matmul(geometry_centers_*correction.scale,m.slice(0,0,3).slice(1,0,3))+m.select(0,3).slice(0,0,3);
    geometry_delta_+=fraction*(moved-geometry_centers_);
  }
  geometry_scale_delta_+=fraction*(correction.scale-1)*geometry_scales_;
  centers_.copy_(geometry_centers_+geometry_delta_);
  offset_scales_.copy_(geometry_scales_+geometry_scale_delta_);
  auto h=voxel_size_/torch::pow(2.,levels_.to(torch::kFloat64));
  keys_.set_data((centers_.to(torch::kFloat64)/h.unsqueeze(1)).floor().to(torch::kLong));
  rebuildBlockLookup();
}

inline void AnchorBankImpl::finishBlockGeometry() {
  torch::NoGradGuard guard;
  geometry_centers_=torch::Tensor();geometry_scales_=torch::Tensor();
  geometry_delta_=torch::Tensor();geometry_scale_delta_=torch::Tensor();
  auto h=voxel_size_/torch::pow(2.,levels_.to(torch::kFloat64));
  keys_.set_data((centers_.to(torch::kFloat64)/h.unsqueeze(1)).floor().to(torch::kLong));
  if(fixedTenMembers() && geometry_keys_.defined())prepareFixedMigration();
  geometry_keys_=torch::Tensor();geometry_touched_=torch::Tensor();
  rebuildBlockLookup();assertBlocks(rows_.numel());
}

}  // namespace improvements::selector
