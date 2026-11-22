#pragma once

namespace improvements::selector {

inline void AnchorBankImpl::enableVariableMembers() {
  TORCH_CHECK(explicitBlocks(),"Variable members require explicit Anchor ownership");
  if(variableMembers())return;
  member_counts_=register_buffer("member_counts",torch::full({size()},10,rows_.options()));
  topology_round_=register_buffer("topology_round",torch::zeros({},rows_.options()));
  if(slotDeactivation()) {
    slot_inactive_.set_data(slot_inactive_.reshape({-1}));
    slot_recovered_.set_data(slot_recovered_.reshape({-1}));
  }
}

inline AnchorBankImpl::BlockSeeds AnchorBankImpl::insertVariableBlocks(
    const std::map<NodeKey,std::vector<int64_t>>& groups,const torch::Tensor& xyz,const torch::Tensor& color,
    bool inherit,const std::vector<torch::optim::Adam*>& optimizers) {
  torch::NoGradGuard guard;
  const auto old=size(),added=int64_t(groups.size());
  auto points=xyz.detach().to(torch::kCPU,torch::kFloat64).contiguous();
  auto p=points.accessor<double,2>();
  auto old_features=packedFeatures().detach().cpu(),old_levels=levels_.cpu().contiguous(),owners=rows_.cpu().contiguous();
  auto l=old_levels.accessor<int64_t,1>(),owner=owners.accessor<int64_t,1>();
  std::vector<int64_t> selected,new_owners,counts,levels,keys;
  std::vector<double> scales,centers,sigma;
  auto new_features=torch::zeros({added,old_features.size(1)},old_features.options());
  int64_t j=0;
  for(const auto& entry:groups) {
    const auto& key=entry.first;const double h=voxel_size_/double(int64_t(1)<<key[0]);
    std::vector<int64_t> unique;
    std::set<std::array<double,3>> seen;
    for(auto i:entry.second)if(seen.insert({p[i][0],p[i][1],p[i][2]}).second)unique.push_back(i);
    std::vector<int64_t> chosen;
    if(unique.size()<=10)chosen=unique;
    else {
      chosen.push_back(unique.front());
      std::vector<double> distance(unique.size(),std::numeric_limits<double>::infinity());
      while(chosen.size()<10) {
        double maximum=-1;int64_t best=0;
        for(size_t b=0;b<unique.size();++b) {
          double d=0;for(int a=0;a<3;++a)d+=std::pow(p[unique[b]][a]-p[chosen.back()][a],2);
          distance[b]=std::min(distance[b],d);
          if(distance[b]>maximum){maximum=distance[b];best=unique[b];}
        }
        chosen.push_back(best);
      }
    }
    for(auto i:chosen) {
      double nearest=std::numeric_limits<double>::infinity();
      for(auto other:unique)if(other!=i) {
        double d=0;for(int a=0;a<3;++a)d+=std::pow(p[i][a]-p[other][a],2);nearest=std::min(nearest,d);
      }
      const double initial=std::clamp(std::isfinite(nearest)?std::sqrt(nearest)*.1:h*.05,h*.001,h*.25);
      selected.push_back(i);new_owners.push_back(old+j);
      for(int a=0;a<3;++a)sigma.push_back(initial);
    }
    counts.push_back(chosen.size());levels.push_back(key[0]);scales.push_back(h);
    for(int a=0;a<3;++a){keys.push_back(key[a+1]);centers.push_back((key[a+1]+.5)*h);}
    if(inherit) {
      bool fine=false;std::vector<int64_t> sources;
      for(auto i:entry.second) {if(l[owner[i]]==key[0])sources.push_back(owner[i]);else fine=true;}
      if(!fine&&!sources.empty())new_features[j].copy_(std::get<0>(old_features.index_select(0,torch::tensor(sources,torch::kLong)).max(0)));
    }
    block_lookup_[key].push_back(old+j);++j;
  }
  auto options=rows_.options();
  resizeFeatures(torch::cat({torch::arange(old,options),torch::full({added},-1,options)}),optimizers);
  copyPackedFeatures(new_features.to(features_.device()),old);
  auto new_centers=torch::tensor(centers,torch::kFloat64).reshape({added,3}).to(centers_.options());
  auto new_scales=torch::tensor(scales,torch::kFloat64).reshape({added,1}).to(offset_scales_.options());
  centers_.set_data(torch::cat({centers_,new_centers}));offset_scales_.set_data(torch::cat({offset_scales_,new_scales}));
  keys_.set_data(torch::cat({keys_,torch::tensor(keys,options).reshape({added,3})}));
  levels_.set_data(torch::cat({levels_,torch::tensor(levels,options)}));
  branches_.set_data(torch::zeros({size()},branches_.options()));
  const auto next=next_uid_.item<int64_t>();
  block_uids_.set_data(torch::cat({block_uids_,torch::arange(next,next+added,options)}));next_uid_.fill_(next+added);
  member_counts_.set_data(torch::cat({member_counts_,torch::tensor(counts,options)}));
  rows_.set_data(torch::cat({rows_,torch::tensor(new_owners,options)}));
  if(slotDeactivation()) {
    slot_inactive_.set_data(torch::cat({slot_inactive_,torch::zeros({int64_t(selected.size())},slot_inactive_.options())}));
    slot_recovered_.set_data(torch::cat({slot_recovered_,torch::zeros({int64_t(selected.size())},slot_recovered_.options())}));
  }
  if(geometry_centers_.defined()) {
    geometry_centers_=torch::cat({geometry_centers_,new_centers});geometry_scales_=torch::cat({geometry_scales_,new_scales});
    geometry_delta_=torch::cat({geometry_delta_,torch::zeros_like(new_centers)});
    geometry_scale_delta_=torch::cat({geometry_scale_delta_,torch::zeros_like(new_scales)});
  }
  auto ids=torch::tensor(selected,options);
  assertBlocks(rows_.numel());
  std::cout<<"[Variable Anchor Insert] anchors="<<added<<" members="<<selected.size()<<" total_anchors="<<size()<<'\n';
  return {xyz.detach().index_select(0,ids),color.detach().index_select(0,ids),
      torch::tensor(sigma,torch::kFloat64).reshape({int64_t(selected.size()),3}).to(xyz.options())};
}

inline void AnchorBankImpl::appendVariableMembers(const torch::Tensor& parents) {
  torch::NoGradGuard guard;
  auto owners=rows_.index_select(0,parents);
  rows_.set_data(torch::cat({rows_,owners}));
  member_counts_.index_add_(0,owners,torch::ones_like(owners));
  if(slotDeactivation()) {
    slot_inactive_.set_data(torch::cat({slot_inactive_,torch::zeros({parents.numel()},slot_inactive_.options())}));
    slot_recovered_.set_data(torch::cat({slot_recovered_,torch::zeros({parents.numel()},slot_recovered_.options())}));
  }
}

inline torch::Tensor AnchorBankImpl::pruneVariableMembers(const torch::Tensor& survivors,
    const std::vector<torch::optim::Adam*>& optimizers) {
  MappingAuditRange trace("topology/anchor_members_prune");
  torch::NoGradGuard guard;
  TORCH_CHECK(!geometry_centers_.defined(),"Cannot prune during a geometry/CD event");
  const auto previous_anchor_count=size();
  auto next_rows=rows_.index({survivors});
  auto counts=torch::zeros_like(member_counts_);counts.index_add_(0,next_rows,torch::ones_like(next_rows));
  auto alive=torch::nonzero(counts>0).reshape({-1});
  auto remap=torch::full_like(counts,-1);remap.index_copy_(0,alive,torch::arange(alive.numel(),rows_.options()));
  if(alive.numel()!=size()) {
    resizeFeatures(alive,optimizers);
    for(auto* value:{&centers_,&offset_scales_,&keys_,&levels_,&branches_,&block_uids_})value->set_data(value->index_select(0,alive));
  }
  member_counts_.set_data(counts.index_select(0,alive));rows_.set_data(remap.index_select(0,next_rows));
  if(slotDeactivation()) {
    slot_inactive_.set_data(slot_inactive_.index({survivors}));slot_recovered_.set_data(slot_recovered_.index({survivors}));
  }
  // Partial Gaussian pruning does not change any key, level or Anchor row.
  if(!fused_full_stats_ || size()!=previous_anchor_count)rebuildBlockLookup();
  assertBlocks(rows_.numel());
  return alive;
}

inline AnchorBankImpl::LocalMigration AnchorBankImpl::migrateLocalMembers(
    const torch::Tensor& xyz,const torch::Tensor& corrected,
    const std::vector<torch::optim::Adam*>& optimizers) {
  torch::NoGradGuard guard;
  TORCH_CHECK(variableMembers() && !geometry_centers_.defined(),"Local migration requires a completed variable Anchor geometry event");
  assertBlocks(xyz.size(0));
  const auto old=size(),n=rows_.numel();
  auto points=xyz.detach().to(torch::kCPU,torch::kFloat64).contiguous();
  auto owners=rows_.cpu().contiguous(),levels=levels_.cpu().contiguous(),keys=keys_.cpu().contiguous();
  auto members=member_counts_.cpu().contiguous(),ids=corrected.to(torch::kCPU,torch::kLong).contiguous();
  const auto p=points.accessor<double,2>();const auto a=owners.accessor<int64_t,1>();
  const auto l=levels.accessor<int64_t,1>();const auto k=keys.accessor<int64_t,2>();
  std::vector<int64_t> order(ids.data_ptr<int64_t>(),ids.data_ptr<int64_t>()+ids.numel());
  std::sort(order.begin(),order.end());order.erase(std::unique(order.begin(),order.end()),order.end());
  std::vector<int64_t> next(a.data(),a.data()+n),counts(members.data_ptr<int64_t>(),members.data_ptr<int64_t>()+old);
  std::vector<std::pair<int64_t,NodeKey>> departures;
  for(auto i:order) {
    TORCH_CHECK(i>=0 && i<n,"Invalid local correction row");
    TORCH_CHECK(std::isfinite(p[i][0])&&std::isfinite(p[i][1])&&std::isfinite(p[i][2]),"Nonfinite local correction XYZ");
    const auto source=a[i];const double h=voxel_size_/double(int64_t(1)<<l[source]);
    NodeKey target{l[source],int64_t(std::floor(p[i][0]/h)),int64_t(std::floor(p[i][1]/h)),int64_t(std::floor(p[i][2]/h))};
    if(target!=NodeKey{l[source],k[source][0],k[source][1],k[source][2]}) {
      departures.emplace_back(i,target);--counts[source];
    }
  }
  // Release all departures first, so reciprocal moves do not depend on KF order.
  auto resident=counts;
  auto destinations=block_lookup_;
  std::vector<NodeKey> added_keys;
  std::vector<int64_t> removed,sources,targets,changed;
  for(const auto& departure:departures) {
    const auto i=departure.first,source=a[i];const auto& key=departure.second;
    auto found=destinations.find(key);
    if(found==destinations.end()) {
      const auto id=old+int64_t(added_keys.size());added_keys.push_back(key);
      counts.push_back(0);resident.push_back(0);destinations[key]={id};found=destinations.find(key);
    }
    int64_t target=-1;
    for(auto id:found->second)if(counts[id]<kGaussiansPerBlock){target=id;break;}
    changed.push_back(source);
    if(target<0) {removed.push_back(i);continue;}
    next[i]=target;++counts[target];sources.push_back(source);targets.push_back(target);changed.push_back(target);
  }
  const auto added=int64_t(added_keys.size());const auto options=rows_.options();
  auto before=packedFeatures().detach();
  if(added) {
    resizeFeatures(torch::cat({torch::arange(old,options),torch::full({added},-1,options)}),optimizers);
    std::vector<int64_t> new_keys,new_levels;std::vector<double> centers,scales;
    for(const auto& key:added_keys) {
      const double h=voxel_size_/double(int64_t(1)<<key[0]);new_levels.push_back(key[0]);scales.push_back(h);
      for(int axis=0;axis<3;++axis){new_keys.push_back(key[axis+1]);centers.push_back((key[axis+1]+.5)*h);}
    }
    centers_.set_data(torch::cat({centers_,torch::tensor(centers,torch::kFloat64).reshape({added,3}).to(centers_.options())}));
    offset_scales_.set_data(torch::cat({offset_scales_,torch::tensor(scales,torch::kFloat64).reshape({added,1}).to(offset_scales_.options())}));
    keys_.set_data(torch::cat({keys_,torch::tensor(new_keys,options).reshape({added,3})}));
    levels_.set_data(torch::cat({levels_,torch::tensor(new_levels,options)}));
    branches_.set_data(torch::cat({branches_,torch::zeros({added},branches_.options())}));
    const auto uid=next_uid_.item<int64_t>();
    block_uids_.set_data(torch::cat({block_uids_,torch::arange(uid,uid+added,options)}));next_uid_.fill_(uid+added);
  }
  if(!targets.empty()) {
    const auto source_ids=torch::tensor(sources,options),target_ids=torch::tensor(targets,options);
    auto numerator=torch::cat({before,torch::zeros({added,before.size(1)},before.options())})*
        torch::tensor(resident,options).to(before.scalar_type()).unsqueeze(1);
    numerator.index_add_(0,target_ids,before.index_select(0,source_ids));
    auto receiving=torch::zeros({size()},options.dtype(torch::kBool));receiving.index_fill_(0,target_ids,true);
    const auto rows=torch::nonzero(receiving).reshape({-1});
    const auto denominator=torch::tensor(counts,options).index_select(0,rows).to(before.scalar_type()).unsqueeze(1);
    auto mixed=numerator.index_select(0,rows)/denominator;
    int64_t column=0;
    for(auto table:featureTables()) {
      table.index_copy_(0,rows,mixed.slice(1,column,column+table.size(1)));column+=table.size(1);
      // A mixed Feature has a new optimization history; keep all unaffected moments.
      for(auto* optimizer:optimizers)if(optimizer) {
        auto found=optimizer->state().find(table.unsafeGetTensorImpl());
        if(found==optimizer->state().end())continue;
        auto& state=static_cast<torch::optim::AdamParamState&>(*found->second);
        state.exp_avg().index_fill_(0,rows,0);state.exp_avg_sq().index_fill_(0,rows,0);
        if(state.max_exp_avg_sq().defined())state.max_exp_avg_sq().index_fill_(0,rows,0);
      }
    }
  }
  LocalMigration result;result.moved=targets.size();result.created=added;
  result.remove=torch::zeros({n},options.dtype(torch::kBool));
  result.changed=torch::zeros({size()},options.dtype(torch::kBool));
  if(!removed.empty())result.remove.index_fill_(0,torch::tensor(removed,options),true);
  if(!changed.empty())result.changed.index_fill_(0,torch::tensor(changed,options),true);
  rows_.set_data(torch::tensor(next,options));
  // Rejected rows retain their old owner only until the caller immediately prunes
  // them and empty Anchors in the same render-locked transaction.
  member_counts_.set_data(torch::zeros({size()},options));
  member_counts_.index_add_(0,rows_,torch::ones_like(rows_));
  if(slotDeactivation()) {
    auto affected=result.changed.index_select(0,rows_);
    slot_inactive_.masked_fill_(affected,false);slot_recovered_.masked_fill_(affected,false);
  }
  rebuildBlockLookup();
  return result;
}

} // namespace improvements::selector
