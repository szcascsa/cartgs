#include "include/gaussian_mapper.h"
#include "include/gaussian_renderer.h"
#include "include/flexgs_importance.h"
#include "include/kernel_fusion.h"
#include "include/flexgs_selector_cuda.h"
#include "include/anchor_selector_cuda.h"
#include "include/loss_utils.h"
#include "include/anchor_frustum.h"

// One full-map population, one LoD image, one backward. The caller owns zero_grad
// and the single Full+LoD Adam commit; no group may update shared parameters here.
GaussianMapper::LodGradientResult GaussianMapper::accumulateSpatialRatioGradients(
    const std::shared_ptr<GaussianKeyframe>& camera,torch::Tensor& truth,size_t ratio_index) {
  TORCH_CHECK((mixed_ratio_config_.enabled||selector_view_residual_enabled_||selector_view_input_enabled_)&&real_lod_.optimizingLod(),"Spatial ratio outside RealLoD");
  LodGradientResult result;
  auto& anchors=selector_network_->anchors();
  const bool detach=selector_detach_gaussian_parameters_;
  auto master=[detach](const torch::Tensor& value){return detach?value.detach():value;};
  auto xyz=master(gaussians_->getXYZ()),scale=master(gaussians_->getScalingActivation());
  auto log_scale=master(gaussians_->scaling_),rotation=master(gaussians_->getRotationActivation());
  auto opacity=master(gaussians_->getOpacityActivation()),logits=master(gaussians_->opacity_);
  auto dc=master(gaussians_->features_dc_),rest=master(gaussians_->features_rest_);
  const auto total=xyz.size(0);
  const double target=selector_target_ratios_[ratio_index];
  const double low=selector_target_ratios_.front(),high=selector_target_ratios_.back();
  const bool mixed=mixed_ratio_config_.enabled&&real_lod_.mixedRound(selector_target_ratios_.size());
  const bool view_round=(selector_view_residual_enabled_||selector_view_input_enabled_)&&real_lod_.viewResidualRound(selector_target_ratios_.size());
  const bool visible_budget=selector_view_input_enabled_&&view_round&&selector_view_visible_budget_;
  int64_t visible_n=total;
  if(visible_budget) {
    auto visible=anchorNominalFrustumMask(anchors->centers(),anchors->offsetScales(),camera->full_proj_transform_);
    visible_n=visible.index_select(0,anchors->rows()).sum().item<int64_t>();
  }
  const double budget_ratio=visible_budget?selector_view_budget_ratio_values_[ratio_index]:target;
  const auto target_k=static_cast<int64_t>(std::floor(budget_ratio*(visible_budget?visible_n:total)));
  const double gi_weight=selector_view_input_enabled_&&view_round?
      (selector_view_gi_lowest_two_only_&&ratio_index>=2?0.:selector_view_gi_weight_):selector_gi_loss_weight_;
  // Endpoints have only one feasible distribution; do not impose artificial group quotas.
  const bool grouped=mixed&&target>low&&target<high;
  torch::Tensor anchor_ids,rows,group_ids;
  std::array<int64_t,3> counts{0,0,0};
  std::array<double,3> local_ratios{target,target,target},shares{1,0,0};
  mixed_ratio::Draw draw;
  {
    torch::NoGradGuard no_grad;
    auto occupancy=torch::bincount(anchors->rows(),{},anchors->size());
    anchor_ids=torch::nonzero(occupancy>0).reshape({-1});
    rows=torch::arange(total,anchors->rows().options());
    group_ids=torch::zeros({rows.numel()},rows.options());
    if(grouped&&rows.numel()) {
      draw=mixed_ratio::draw(mixed_ratio_config_,target,low,high,real_lod_.lod_steps);
      shares=draw.shares;
      auto metric=mixed_ratio::projectedLongestSide(anchors->centers().index_select(0,anchor_ids),
          anchors->offsetScales().index_select(0,anchor_ids),camera->full_proj_transform_,
          camera->world_view_transform_,camera->image_height_,camera->image_width_,camera->znear_);
      auto levels=mixed_ratio::levels(metric,occupancy.index_select(0,anchor_ids),
          anchors->blockUIDs().index_select(0,anchor_ids),shares);
      auto all_levels=torch::full({anchors->size()},-1,rows.options()).index_copy(0,anchor_ids,levels);
      group_ids=all_levels.index_select(0,anchors->rows().index_select(0,rows));
    }
    auto host_counts=torch::bincount(group_ids,{},3).cpu();
    for(int g=0;g<3;++g)counts[g]=host_counts[g].item<int64_t>();
    if(grouped&&rows.numel())local_ratios=mixed_ratio::ratios(draw,counts,target,low,high);
  }
  if(selector_slot_deactivation_)anchors->refreshSlotActivity(opacity);
  GaussianRenderInput full{xyz,opacity,scale,rotation,dc,rest,
      gaussians_->active_sh_degree_,gaussians_->max_sh_degree_};
  torch::Tensor view_codes;
  if(view_round)view_codes=improvements::selector::anchorViewCode(anchors->centers(),anchors->offsetScales(),
      camera->camera_center_,camera->world_view_transform_,camera->full_proj_transform_,
      camera->image_height_,camera->image_width_,camera->znear_,selector_view_input_enabled_?
          selector_network_->selectorViewSceneScale():selector_network_->selectorViewResidual()->scene_scale);
  std::vector<GaussianRenderInput> packets;
  std::vector<torch::Tensor> selected_rows,probabilities,eligibilities;
  auto ratio_loss=torch::zeros({},xyz.options()),gi_loss=torch::zeros_like(ratio_loss);
  auto& trace=result.spatial_plan;
  trace["step"]=Json::UInt64(real_lod_.lod_steps+1);
  trace["round"]=mixed?"mixed":"uniform";trace["mixed_spatial"]=grouped;
  if(selector_view_residual_enabled_||selector_view_input_enabled_) {
    trace["round"]=view_round?(selector_view_input_enabled_?"view_input":"view_residual"):"global";
    trace["keyframe_id"]=Json::UInt64(camera->fid_);
    trace["view_conditioned"]=view_round;
    trace["importance_source"]=view_round?"current_full_view":"all_training_views";
  }
  trace["seed"]=mixed_ratio_config_.seed;trace["target_ratio"]=target;
  trace["map_N"]=Json::Int64(total);trace["candidate_N"]=Json::Int64(rows.numel());
  trace["candidate_scope"]="full_map";trace["anchor_frustum_culling"]=false;
  trace["visible_point_budget"]=visible_budget;
  trace["budget_N"]=Json::Int64(visible_budget?visible_n:total);
  trace["budget_ratio"]=budget_ratio;trace["budget_target_K"]=Json::Int64(target_k);
  trace["budget_scope"]=visible_budget?"nominal_visible_anchor_members":"full_map";
  trace["candidate_anchors"]=Json::Int64(anchor_ids.numel());trace["sigma"]=draw.sigma;
  double effective=0;
  if(flexgs_importance_enabled_)
    TORCH_CHECK(flexgs_importance_.defined()&&flexgs_importance_.numel()==total&&
        flexgs_importance_topology_==topology_version_,"MixedRatio importance/topology mismatch");
  for(int g=0;g<(grouped?3:1);++g) {
    auto group_local=torch::nonzero(group_ids==g).reshape({-1});
    auto source=rows.index_select(0,group_local);
    const auto n=source.numel();
    const float ratio=static_cast<float>(local_ratios[g]);
    auto input=full.indexSelect(source);
    auto eligible=selector_network_->maskInactiveSlots(torch::ones({n},xyz.options()),input.opacity,source).to(torch::kBool);
    torch::Tensor probability,mask,chosen;
    if(n) {
      auto timing=mapping_profiler_.scope(MappingProfiler::SelectorForward);
      const bool fast=fused_flexgs_selector_&&xyz.is_cuda()&&xyz.scalar_type()==torch::kFloat32&&
          mapping_cuda::SelectorExecutionDispatcher::plan(n,xyz.get_device()).supported();
      improvements::selector::GumbelNetworkImpl::SelectionFeatures selection;
      {
        torch::AutoGradMode mode(torch::GradMode::is_enabled()&&(!view_round||selector_view_input_enabled_));
        auto code=selector_view_input_enabled_&&view_round?view_codes.index_select(0,anchors->rows().index_select(0,source)):torch::Tensor{};
        selection=fast?selector_network_->forwardFlexgsCUDA(input.xyz,ratio,input.scaling,input.features_dc,input.opacity,source,eligible,code):
            selector_network_->forwardSelectionFeatures(input.xyz,ratio,selector_temperature_,source,input.scaling,input.features_dc,input.opacity,code);
      }
      if(view_round&&selector_view_residual_enabled_)selection=selector_network_->applyViewResidual(selection,
          view_codes.index_select(0,anchors->rows().index_select(0,source)));
      auto noise=torch::empty_like(selection.logits).exponential_();
      if(xyz.is_cuda()) {
        auto sampled=mapping_cuda::flexgsSampleCompact(selection.logits,noise,eligible,selector_temperature_);
        probability=sampled[0];mask=sampled[1];chosen=sampled[2];
      } else {
        probability=(selection.logits/selector_temperature_).softmax(-1).select(1,1)*eligible;
        mask=sampleSelectorGumbelST(selection.logits,selector_temperature_,-noise.log())*eligible;
        chosen=torch::nonzero(mask.detach()>0).reshape({-1});
      }
      const double weight=double(n)/rows.numel();
      ratio_loss=ratio_loss+weight*(visible_budget?
          selectorGumbelBudgetLoss(mask,target_k,selector_gumbel_budget_normalize_):
          selectorGumbelRatioLoss(mask,ratio,selector_gumbel_budget_normalize_));
      if(flexgs_importance_enabled_&&(selector_gi_loss_weight_!=0||selector_view_input_enabled_)) {
        auto timing=mapping_profiler_.scope(MappingProfiler::GITeacher);
        const auto& importance=view_round?view_importance_:flexgs_importance_;
        TORCH_CHECK(importance.defined()&&importance.numel()==total&&!importance.requires_grad(),"Invalid detached view/global GI");
        auto primary=importance.index_select(0,source);
        auto labels=visible_budget?
            budgetAlignedImportanceTargetK(primary,flexgs_importance_.index_select(0,source),source,eligible,target_k):selector_view_input_enabled_?
            budgetAlignedImportanceTarget(primary,flexgs_importance_.index_select(0,source),source,eligible,ratio):
            flexgsImportanceTarget(primary,ratio);
        if(gi_weight!=0)gi_loss=gi_loss+weight*(selector_view_input_enabled_?
            budgetAlignedImportanceLoss(selection.logits,labels,eligible,selector_temperature_):(mask-labels).abs().mean());
        if(selector_view_input_enabled_) {
          torch::NoGradGuard guard;
          auto legacy=flexgsImportanceTarget(primary,ratio);
          const auto requested=visible_budget?target_k:static_cast<int64_t>(std::floor(double(ratio)*n));
          trace["requested_label_K"]=Json::Int64(requested);
          trace["label_K"]=Json::Int64(labels.sum().item<int64_t>());
          trace["eligible_N"]=Json::Int64(eligible.sum().item<int64_t>());
          trace["label_budget_shortfall"]=Json::Int64(std::max<int64_t>(0,requested-eligible.sum().item<int64_t>()));
          trace["legacy_label_K"]=Json::Int64((legacy*eligible).sum().item<int64_t>());
          trace["gi_positive_ratio"]=labels.mean().item<double>();
          trace["gi_positive_eligible_ratio"]=(labels*eligible).mean().item<double>();
          trace["gi_nonzero_ratio"]=(primary>0).to(torch::kFloat32).mean().item<double>();
          trace["gi_supervision"]="deterministic_logit_BCE_exact_K";
          trace["gi_weight"]=gi_weight;
        }
      }
    } else {
      probability=torch::empty({0},xyz.options());mask=probability;chosen=source;
    }
    auto ids=source.index_select(0,chosen);
    auto selected=full.indexSelect(ids);
    const auto selected_mask=mask.index_select(0,chosen);
    bool decoded=false;
    if(selector_compensation_&&getIteration()>=selector_compensation_start_iter_&&ids.numel()) {
      auto timing=mapping_profiler_.scope(MappingProfiler::Compensation);
      auto& spatial=selector_network_->spatialCompensation();
      if(!spatial->ready())spatial->initialize(xyz,log_scale,rotation,dc,logits,
          anchors->centers().index_select(0,anchors->rows()),anchors->offsetScales().index_select(0,anchors->rows()),true);
      auto owners=anchors->rows().index_select(0,ids);
      auto centers=anchors->centers().index_select(0,owners),references=anchors->offsetScales().index_select(0,owners);
      torch::Tensor packed;
      if(fused_residual48_training_input_&&xyz.is_cuda()) {
        auto tables=anchors->compensationFeatureTables();
        auto weights=anchors->compensationResidualWeights(ratio);
        std::vector<torch::Tensor> features;
        for(size_t j=0;j<weights.size();++j)features.push_back(tables[j].index_select(0,owners));
        packed=mapping_cuda::residual48TrainingPack(selected.xyz,centers,references,spatial->mean,spatial->stddev,features,weights);
      }
      auto residual=spatial->forward(selected.xyz,log_scale.index_select(0,ids),selected.rotation,selected.features_dc,
          logits.index_select(0,ids),centers,references,ratio,true,
          packed.defined()?torch::Tensor{}:anchors->compensationInput(ids,ratio),packed);
      if(fused_compensation_decode_&&xyz.is_cuda()&&residual.scalar_type()==torch::kFloat32) {
        auto output=(spatial->softplusScale()?mapping_cuda::decode14SoftplusScale:
            (spatial->physicalScaleAdditive()?mapping_cuda::decode14AdditiveScale:mapping_cuda::decode14LogScale))(
                selected.xyz,log_scale.index_select(0,ids),selected.rotation,logits.index_select(0,ids),selected.features_dc,residual,selected_mask);
        selected.xyz=output[0];selected.scaling=output[1];selected.rotation=output[2];
        selected.opacity=output[3];selected.features_dc=output[4];decoded=true;
      } else {
        using Network=improvements::selector::GumbelNetworkImpl;
        auto geometry=Network::applyCompensationResidual({selected.xyz,log_scale.index_select(0,ids),selected.rotation},residual);
        selected.xyz=geometry.xyz;selected.rotation=geometry.rotation;
        selected.scaling=spatial->decodeScale(log_scale.index_select(0,ids),residual.slice(1,3,6));
        auto appearance=Network::applyAppearanceResidual(selected.features_dc,logits.index_select(0,ids),residual);
        selected.features_dc=appearance.first;selected.opacity=appearance.second.sigmoid();
      }
      result.compensated_gaussians+=ids.numel();++compensation_forwards_;
      compensation_decoded_rows_+=ids.numel();
    }
    if(!decoded)selected.opacity=selected_mask.unsqueeze(1)*selected.opacity;
    packets.push_back(selected);selected_rows.push_back(ids);
    probabilities.push_back(probability);eligibilities.push_back(eligible);
    Json::Value group;
    group["level"]=g;group["ratio"]=ratio;group["target_share"]=shares[g];
    group["actual_share"]=rows.numel()?double(n)/rows.numel():0.;
    group["N"]=Json::Int64(n);group["K"]=Json::Int64(ids.numel());
    group["eligible"]=Json::Int64(eligible.sum().item<int64_t>());
    group["target_K"]=visible_budget?double(target_k):double(n)*ratio;
    group["slot_shortfall"]=std::max(0.,group["target_K"].asDouble()-group["eligible"].asDouble());
    trace["groups"].append(group);effective+=n*double(ratio)/std::max<int64_t>(1,rows.numel());
  }
  auto ids=torch::cat(selected_rows);
  auto order=at::argsort(ids,true,0,false);ids=ids.index_select(0,order);
  auto combine=[&](auto field) {
    std::vector<torch::Tensor> values;
    for(const auto& packet:packets)values.push_back(packet.*field);
    return torch::cat(values).index_select(0,order);
  };
  GaussianRenderInput input{combine(&GaussianRenderInput::xyz),combine(&GaussianRenderInput::opacity),
      combine(&GaussianRenderInput::scaling),combine(&GaussianRenderInput::rotation),
      combine(&GaussianRenderInput::features_dc),combine(&GaussianRenderInput::features_rest),
      full.active_sh_degree,full.max_sh_degree};
  result.active_gaussians=ids.numel();
  if(lod_health_guard_.enabled) {
    auto reason=lod_health_guard_.selection(ratio_index,mapping_iter_+1,target,rows.numel(),
        ids.numel(),torch::cat(probabilities),torch::cat(eligibilities));
    lod_health_guard_.sample["lod_step"]=Json::UInt64(real_lod_.lod_steps+1);
    if(!reason.empty())failLodHealth(reason);
    auto reference=anchors->offsetScales().index_select(0,anchors->rows().index_select(0,ids));
    reason=lod_health_guard_.compensation(input.xyz,input.scaling,input.opacity,
        xyz.index_select(0,ids),scale.index_select(0,ids),reference);
    if(!reason.empty())failLodHealth(reason);
    lod_health_guard_.record(result_dir_);
  }
  RenderPackage rendered;
  {
    auto timing=mapping_profiler_.scope(MappingProfiler::LodRender);
    rendered=GaussianRenderer::render(camera,camera->image_height_,camera->image_width_,input,
        pipe_params_,background_,override_color_,1.f,false,false);
  }
  result.rendered=true;result.render_height=camera->image_height_;result.render_width=camera->image_width_;
  if(rendered.visibility_indices.defined()&&rendered.visibility_indices.numel()) {
    result.supervised_indices=ids.index_select(0,rendered.visibility_indices);
    gaussians_->updateElasticSupervision(result.supervised_indices,static_cast<int64_t>(mapping_iter_));
  }
  torch::Tensor loss;
  {
    auto timing=mapping_profiler_.scope(MappingProfiler::LodLoss);
    auto l1=opt_params_.smooth_l1_?loss_utils::smooth_l1_loss:loss_utils::l1_loss;
    auto photo=(1.-lambdaDssim())*l1(rendered.image,truth,1.f)+
        lambdaDssim()*(1.-loss_utils::fast_ssim(rendered.image,truth));
    loss=selector_render_loss_weight_*photo+selector_ratio_loss_weight_*ratio_loss+gi_weight*gi_loss;
    if(selector_view_input_enabled_)trace["photo_loss"]=photo.detach().item<double>();
  }
  result.loss=loss.detach().item<double>();result.finite_loss=std::isfinite(result.loss);
  if(!result.finite_loss)return result;
  trace["backward_performed"]=loss.requires_grad();
  if(loss.requires_grad()) {
    auto timing=mapping_profiler_.scope(MappingProfiler::LodBackward);loss.backward();
  }
  trace["weighted_ratio"]=effective;trace["weighted_ratio_error"]=effective-target;
  trace["K"]=Json::Int64(ids.numel());trace["actual_ratio"]=rows.numel()?double(ids.numel())/rows.numel():0.;
  if(visible_budget)trace["actual_visible_ratio"]=visible_n?double(ids.numel())/visible_n:0.;
  trace["budget_loss"]=ratio_loss.detach().item<double>();trace["gi_loss"]=gi_loss.detach().item<double>();
  trace["empty_map"]=rows.numel()==0;
  result.trained=true;
  return result;
}
