#include "include/gaussian_mapper.h"
#include "include/gaussian_renderer.h"
#include "include/anchor_frustum.h"
#include "include/anchor_selector_cuda.h"
#include "include/flexgs_selector_cuda.h"
#include "include/kernel_fusion.h"
#include "include/loss_utils.h"
#include "include/view_budget_training.h"
#include "include/local_importance.h"
#include <cmath>

// Called after Full and global LoD backward, before their shared optimizer step.
// Every tensor below reads the same pre-Adam Master/network state.
GaussianMapper::LodGradientResult GaussianMapper::accumulateViewBudgetGradients(
    const std::shared_ptr<GaussianKeyframe>& camera,torch::Tensor& truth,size_t ratio_index) {
  TORCH_CHECK(view_budget_enabled_&&real_lod_.optimizingLod(),"ViewBudget outside Fine");
  LodGradientResult result;
  auto& anchors=selector_network_->anchors();
  const auto xyz=gaussians_->getXYZ(),scale=gaussians_->getScalingActivation();
  const auto opacity=gaussians_->getOpacityActivation(),rotation=gaussians_->getRotationActivation();
  const auto total=xyz.size(0);
  const double ratio=selector_target_ratios_.at(ratio_index);
  torch::Tensor source;
  {
    torch::NoGradGuard no_grad;
    if(view_budget_global_control_)source=torch::arange(total,anchors->rows().options());
    else {
      auto visible=anchorNominalFrustumMask(anchors->centers(),anchors->offsetScales(),camera->full_proj_transform_);
      source=torch::nonzero(visible.index_select(0,anchors->rows())).reshape({-1});
    }
    if(!view_budget_generator_.defined()) {
      view_budget_generator_=at::globalContext().defaultGenerator(xyz.device()).clone();
      view_budget_generator_.set_current_seed(view_budget_seed_);
    }
  }
  GaussianRenderInput master{xyz,opacity,scale,rotation,gaussians_->features_dc_,gaussians_->features_rest_,
      gaussians_->active_sh_degree_,gaussians_->max_sh_degree_};
  auto candidate=master.indexSelect(source);
  const auto n=source.numel();
  const double target=ratio*n;
  if(selector_slot_deactivation_)anchors->refreshSlotActivity(opacity);
  auto eligible=selector_network_->maskInactiveSlots(torch::ones({n},xyz.options()),candidate.opacity,source).to(torch::kBool);
  auto& trace=result.spatial_plan;
  trace["step"]=Json::UInt64(real_lod_.lod_steps+1);trace["keyframe_id"]=Json::UInt64(camera->fid_);
  trace["scope"]=view_budget_global_control_?"global_control":"nominal_anchor_frustum";
  trace["N"]=Json::Int64(total);trace["candidate_N"]=Json::Int64(n);
  trace["target_K"]=target;trace["ratio"]=ratio;
  trace["loss_weight"]=view_budget_loss_weight_;trace["gi_weight"]=0;
  trace["li_weight"]=view_budget_li_weight_;trace["li_loss"]=0.;trace["li_supervised"]=false;
  trace["view_residual"]=view_budget_view_residual_;
  trace["conditioning"]=view_budget_conditioning_;
  trace["selector_real_view"]=view_budget_conditioning_==2||view_budget_conditioning_==3;
  trace["compensator_real_view"]=view_budget_conditioning_==2||view_budget_conditioning_==4;
  trace["li_gradient_scope"]=view_budget_view_residual_?"selector_view_residual":"local_emd_and_soft_net";
  trace["sampler"]="gumbel_st";trace["strict_topk"]=false;
  trace["budget_normalize_target"]=selector_gumbel_budget_normalize_;
  trace["seed"]=view_budget_seed_;trace["rng_stream"]="private_advancing";
  const auto available=eligible.sum().item<int64_t>();
  trace["eligible_N"]=Json::Int64(available);
  trace["infeasible_budget_excess"]=std::max(0.,target-available);
  if(!n) {
    if(view_budget_health_.enabled) {
      view_budget_health_.selection(ratio_index,mapping_iter_+1,ratio,0,0,torch::empty({0},xyz.options()),eligible);
      view_budget_health_.sample["branch"]="view_budget";
      view_budget_health_.record(result_dir_);
    }
    trace["K"]=0;trace["rendered"]=false;trace["empty_candidates"]=true;
    trace["photo_loss"]=0.;trace["budget_loss"]=0.;trace["weighted_loss"]=0.;
    result.trained=true;
    return result;
  }
  improvements::selector::GumbelNetworkImpl::SelectionFeatures selection;
  ViewBudgetSelection sampled;
  torch::Tensor anchor_view,view_code;
  if(view_budget_conditioning_) {
    if(view_budget_conditioning_==1)anchor_view=torch::zeros({anchors->size(),7},xyz.options());
    else anchor_view=improvements::selector::anchorViewCode(anchors->centers(),anchors->offsetScales(),
        camera->camera_center_,camera->world_view_transform_,camera->full_proj_transform_,
        camera->image_height_,camera->image_width_,camera->znear_,selector_network_->selectorViewSceneScale());
    view_code=anchor_view.index_select(0,anchors->rows().index_select(0,source));
  }
  {
    auto timing=mapping_profiler_.scope(MappingProfiler::SelectorForward);
    const bool fast=fused_flexgs_selector_&&xyz.is_cuda()&&
        mapping_cuda::SelectorExecutionDispatcher::plan(n,xyz.get_device()).supported();
    selection=fast?selector_network_->forwardFlexgsCUDA(candidate.xyz,ratio,candidate.scaling,
        candidate.features_dc,candidate.opacity,source,eligible,view_code):
        selector_network_->forwardSelectionFeatures(candidate.xyz,ratio,1.f,source,candidate.scaling,
        candidate.features_dc,candidate.opacity,view_code);
    if(view_budget_view_residual_) {
      auto& branch=selector_network_->selectorViewResidual();
      branch->initialize(xyz);
      auto code=improvements::selector::anchorViewCode(anchors->centers(),anchors->offsetScales(),
          camera->camera_center_,camera->world_view_transform_,camera->full_proj_transform_,
          camera->image_height_,camera->image_width_,camera->znear_,branch->scene_scale);
      auto owners=anchors->rows().index_select(0,source);
      // The residual reads detached hidden/view features; local Photo/Budget
      // still differentiates the original logits. LI uses a targeted VJP below.
      selection=selector_network_->applyViewResidual(selection,code.index_select(0,owners),false);
    }
    auto exponentials=torch::empty_like(selection.logits).exponential_(1.,view_budget_generator_);
    sampled=viewBudgetSelection(selection.logits,eligible,exponentials,selector_temperature_);
  }
  auto budget=selectorGumbelRatioLoss(sampled.mask,ratio,selector_gumbel_budget_normalize_);
  auto ids=source.index_select(0,sampled.chosen);
  auto selected=master.indexSelect(ids);
  auto mask=sampled.mask.index_select(0,sampled.chosen);
  result.active_gaussians=ids.numel();
  if(view_budget_health_.enabled) {
    auto reason=view_budget_health_.selection(ratio_index,mapping_iter_+1,ratio,n,ids.numel(),sampled.probability,eligible);
    view_budget_health_.sample["branch"]="view_budget";
    if(!reason.empty()) {
      lod_health_guard_.sample=view_budget_health_.sample;
      failLodHealth("view_budget_"+reason);
    }
  }
  if(ids.numel()&&getIteration()>=selector_compensation_start_iter_) {
    auto timing=mapping_profiler_.scope(MappingProfiler::Compensation);
    auto& spatial=selector_network_->spatialCompensation();
    if(!spatial->ready())spatial->initialize(xyz,gaussians_->scaling_,rotation,gaussians_->features_dc_,gaussians_->opacity_,
        anchors->centers().index_select(0,anchors->rows()),anchors->offsetScales().index_select(0,anchors->rows()),true);
    auto owners=anchors->rows().index_select(0,ids);
    auto centers=anchors->centers().index_select(0,owners),reference=anchors->offsetScales().index_select(0,owners);
    torch::Tensor packed;
    if(fused_residual48_training_input_&&xyz.is_cuda()) {
      auto tables=anchors->compensationFeatureTables();auto weights=anchors->compensationResidualWeights(ratio);
      std::vector<torch::Tensor> features;
      for(size_t i=0;i<weights.size();++i)features.push_back(tables[i].index_select(0,owners));
      packed=mapping_cuda::residual48TrainingPack(selected.xyz,centers,reference,spatial->mean,spatial->stddev,features,weights);
    }
    auto residual=spatial->forward(selected.xyz,gaussians_->scaling_.index_select(0,ids),selected.rotation,
        selected.features_dc,gaussians_->opacity_.index_select(0,ids),centers,reference,ratio,true,
        packed.defined()?torch::Tensor{}:anchors->compensationInput(ids,ratio),packed,
        view_budget_conditioning_?selector_network_->packCompensationViewInput(anchor_view.index_select(0,owners)):torch::Tensor{});
    // Keep all four Heads, scale decoding and Master/Feature gradients intact.
    if(fused_compensation_decode_&&xyz.is_cuda()) {
      auto output=(spatial->softplusScale()?mapping_cuda::decode14SoftplusScale:
          (spatial->physicalScaleAdditive()?mapping_cuda::decode14AdditiveScale:mapping_cuda::decode14LogScale))(
              selected.xyz,gaussians_->scaling_.index_select(0,ids),selected.rotation,
              gaussians_->opacity_.index_select(0,ids),selected.features_dc,residual,mask);
      selected.xyz=output[0];selected.scaling=output[1];selected.rotation=output[2];
      selected.opacity=output[3];selected.features_dc=output[4];
    } else {
      using Network=improvements::selector::GumbelNetworkImpl;
      auto geometry=Network::applyCompensationResidual({selected.xyz,gaussians_->scaling_.index_select(0,ids),selected.rotation},residual);
      selected.xyz=geometry.xyz;selected.rotation=geometry.rotation;
      selected.scaling=spatial->decodeScale(gaussians_->scaling_.index_select(0,ids),residual.slice(1,3,6));
      auto appearance=Network::applyAppearanceResidual(selected.features_dc,gaussians_->opacity_.index_select(0,ids),residual);
      selected.features_dc=appearance.first;selected.opacity=appearance.second.sigmoid();
    }
    if(!(fused_compensation_decode_&&xyz.is_cuda()))selected.opacity=selected.opacity*mask.unsqueeze(1);
    result.compensated_gaussians=ids.numel();++compensation_forwards_;compensation_decoded_rows_+=ids.numel();
  } else selected.opacity=selected.opacity*mask.unsqueeze(1);
  if(view_budget_health_.enabled) {
    auto reference=anchors->offsetScales().index_select(0,anchors->rows().index_select(0,ids));
    auto reason=view_budget_health_.compensation(selected.xyz,selected.scaling,selected.opacity,
        xyz.index_select(0,ids),scale.index_select(0,ids),reference);
    if(!reason.empty()) {
      lod_health_guard_.sample=view_budget_health_.sample;
      failLodHealth("view_budget_"+reason);
    }
    view_budget_health_.record(result_dir_);
  }
  RenderPackage rendered;
  {
    auto timing=mapping_profiler_.scope(MappingProfiler::LodRender);
    rendered=GaussianRenderer::render(camera,camera->image_height_,camera->image_width_,selected,
        pipe_params_,background_,override_color_,1.f,false,false);
    ++view_budget_renders_;
  }
  result.rendered=true;result.render_height=camera->image_height_;result.render_width=camera->image_width_;
  auto l1=opt_params_.smooth_l1_?loss_utils::smooth_l1_loss:loss_utils::l1_loss;
  auto photo=(1.-lambdaDssim())*l1(rendered.image,truth,1.f)+lambdaDssim()*(1.-loss_utils::fast_ssim(rendered.image,truth));
  auto loss=view_budget_loss_weight_*(selector_render_loss_weight_*photo+selector_ratio_loss_weight_*budget);
  torch::Tensor li_loss;
  if(view_budget_li_weight_>0) {
    TORCH_CHECK(view_budget_li_.defined()&&view_budget_li_.numel()==total&&!view_budget_li_.requires_grad()&&
        view_budget_li_step_==real_lod_.lod_steps+1&&view_budget_li_camera_==camera->fid_&&
        view_budget_li_topology_==topology_version_,"Stale or missing current Full LI");
    auto local=view_budget_li_.index_select(0,source);
    auto labels=localImportanceTarget(local,flexgs_importance_.index_select(0,source),source,eligible,ratio);
    if(labels.defined()) {
      li_loss=localImportanceLoss(sampled.mask,labels);
      trace["li_loss"]=li_loss.detach().item<double>();trace["li_supervised"]=true;
      trace["li_teacher_K"]=Json::Int64(labels.sum().item<int64_t>());
      // The scalar includes LI for reporting; its gradient is routed explicitly below.
      loss=loss+view_budget_loss_weight_*view_budget_li_weight_*li_loss.detach();
      if((real_lod_.lod_steps+1)%100==0||real_lod_.lod_steps<6) {
        torch::NoGradGuard guard;
        auto global_labels=flexgsImportanceTarget(flexgs_importance_,ratio).index_select(0,source);
        trace["li_global_label_disagreement"]=(labels-global_labels).abs().mul(eligible).sum().item<double>()/std::max<int64_t>(1,available);
        trace["li_pool_full_contribution_fraction"]=local.sum().item<double>()/std::max(1e-20,view_budget_li_.sum().item<double>());
        trace["li_selected_mass_fraction"]=local.index_select(0,sampled.chosen).sum().item<double>()/std::max(1e-20,local.sum().item<double>());
        trace["li_positive_contributors"]=Json::Int64(((local>0)&eligible).sum().item<int64_t>());
        trace["li_teacher_zero_contributors"]=Json::Int64(((labels>0)&(local==0)).sum().item<int64_t>());
        trace["li_teacher_mass_fraction"]=(local*labels).sum().item<double>()/std::max(1e-20,local.sum().item<double>());
        trace["li_sample_teacher_overlap_fraction"]=labels.index_select(0,sampled.chosen).sum().item<double>()/std::max<int64_t>(1,ids.numel());
      }
    }
  }
  result.loss=loss.detach().item<double>();result.finite_loss=std::isfinite(result.loss);
  trace["K"]=Json::Int64(ids.numel());trace["rendered"]=true;
  trace["photo_loss"]=photo.detach().item<double>();trace["budget_loss"]=budget.detach().item<double>();
  trace["actual_ratio"]=double(ids.numel())/n;
  trace["absolute_budget_error"]=std::abs(double(ids.numel())-target);
  trace["weighted_loss"]=result.loss;trace["projected_K"]=Json::Int64(rendered.visibility_indices.numel());
  if(result.finite_loss&&loss.requires_grad()) {
    auto timing=mapping_profiler_.scope(MappingProfiler::LodBackward);
    if(li_loss.defined()) {
      std::vector<torch::Tensor> parameters;
      if(view_budget_view_residual_)parameters=selector_network_->selectorViewResidual()->parameters();
      else for(const auto& p:selector_network_->named_parameters())
        if(p.key().rfind("local_emd.",0)==0||p.key().rfind("soft_net.",0)==0)parameters.push_back(p.value());
      TORCH_CHECK(parameters.size()==(view_budget_view_residual_?4:6),"Unexpected LI gradient scope");
      const bool diagnose=(real_lod_.lod_steps+1)%100==0||real_lod_.lod_steps<6;
      std::vector<torch::Tensor> previous;
      if(diagnose)for(auto p:parameters)previous.push_back(p.grad().defined()?p.grad().detach().clone():torch::zeros_like(p));
      auto gradients=accumulateLocalImportanceGradients(view_budget_loss_weight_*view_budget_li_weight_*li_loss,parameters);
      if(diagnose) {
        torch::NoGradGuard guard;
        std::vector<torch::Tensor> squares,prior,dots;
        for(size_t i=0;i<gradients.size();++i) {
          squares.push_back(gradients[i].square().sum());prior.push_back(previous[i].square().sum());
          dots.push_back((gradients[i]*previous[i]).sum());
        }
        const double norm=torch::stack(squares).sum().sqrt().item<double>();
        const double old_norm=torch::stack(prior).sum().sqrt().item<double>();
        trace["li_selector_grad_norm"]=norm;trace["pre_local_selector_grad_norm"]=old_norm;
        trace["li_vs_pre_local_gradient_cosine"]=torch::stack(dots).sum().item<double>()/std::max(1e-20,norm*old_norm);
      }
      ++view_budget_li_backwards_;
    }
    loss.backward();++view_budget_backwards_;
  }
  result.trained=true;
  return result;
}
