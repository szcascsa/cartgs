#pragma once

#include <torch/torch.h>
#include <cmath>
#include <vector>
#include "../../include/gaussian_transform_ops.h"
#include "../../include/mapping_audit_range.h"

namespace improvements::selector {

// Gaussian identity state, independent of Anchor ownership. Only selected rows
// are gathered by forward; topology changes migrate both leaves and Adam aliases.
class GaussianCompensationBankImpl : public torch::nn::Module {
 public:
  GaussianCompensationBankImpl(bool independent, const std::vector<double>& knots)
      : independent_(independent), values_(knots) {
    TORCH_CHECK(!knots.empty(),"Private compensation requires training ratios");
    for(size_t i=0;i<knots.size();++i)
      TORCH_CHECK(std::isfinite(knots[i]) && knots[i]>0 && knots[i]<=1 && (!i || knots[i]>knots[i-1]),
                  "Compensation ratios must increase in (0,1]");
    knots_=register_buffer("ratio_knots",torch::tensor(knots,torch::kFloat64));
    uids_=register_buffer("gaussian_uids",torch::empty({0},torch::kLong));
    next_uid_=register_buffer("next_uid",torch::zeros({},torch::kLong));
    for(size_t j=0;j<(independent?knots.size():1);++j)
      tables_.push_back(register_parameter("residual_"+std::to_string(j),torch::zeros({0,14})));
  }
  bool independent() const { return independent_; }
  const std::vector<double>& ratioValues() const { return values_; }
  const torch::Tensor& uids() const { return uids_; }
  int64_t size() const { return uids_.numel(); }
  const std::vector<torch::Tensor>& tables() const { return tables_; }

  torch::Tensor forward(const torch::Tensor& source_rows,double ratio) const {
    TORCH_CHECK(source_rows.defined() && source_rows.dim()==1,"Private compensation requires Gaussian source rows");
    TORCH_CHECK(std::isfinite(ratio) && ratio>0 && ratio<=1,"Invalid compensation ratio");
    if(!independent_)return tables_[0].index_select(0,source_rows);
    size_t hi=0;while(hi+1<values_.size() && ratio>values_[hi]+1e-8)++hi;
    auto upper=tables_[hi].index_select(0,source_rows);
    if(hi==0 || ratio>=values_.back() || std::abs(ratio-values_[hi])<=1e-8)return upper;
    const double t=(ratio-values_[hi-1])/(values_[hi]-values_[hi-1]);
    return (1-t)*tables_[hi-1].index_select(0,source_rows)+t*upper;
  }

  void append(int64_t count,const torch::Tensor& parents,const std::vector<torch::optim::Adam*>& optimizers) {
    if(!count)return;
    TORCH_CHECK(!parents.defined() || parents.numel()==count,"Compensation append parent count mismatch");
    auto fresh=torch::full({count},-1,uids_.options());
    auto keep=torch::arange(size(),uids_.options());
    resize(torch::cat({keep,fresh}),torch::cat({keep,parents.defined()?parents:fresh}),optimizers);
  }
  void select(const torch::Tensor& rows,const std::vector<torch::optim::Adam*>& optimizers) {
    resize(rows,rows,optimizers);
  }
  void copyLayoutFrom(const GaussianCompensationBankImpl& source) {
    torch::NoGradGuard guard;
    TORCH_CHECK(independent_==source.independent_ && values_==source.values_,"Compensation schema mismatch");
    for(size_t j=0;j<tables_.size();++j)tables_[j].set_data(torch::zeros_like(source.tables_[j]));
    uids_.set_data(source.uids_.clone());next_uid_.copy_(source.next_uid_);
  }

  void transport(const GaussianTransformResult& correction,const std::vector<torch::optim::Adam*>& optimizers) {
    if(!size() || !(correction.changed_attributes & (toBits(GaussianAttrFlag::XYZ)|toBits(GaussianAttrFlag::ROTATION))))return;
    torch::NoGradGuard guard;
    auto ids=correction.affects_all?torch::arange(size(),uids_.options()):correction.indices;
    if(!ids.numel())return;
    for(auto table:tables_) {
      auto value=table.index_select(0,ids).clone();
      if(correction.changed_attributes & toBits(GaussianAttrFlag::XYZ))
        value.slice(1,0,3).copy_(torch::matmul(value.slice(1,0,3)*correction.scale,
            correction.transform_4x4.slice(0,0,3).slice(1,0,3)));
      if(correction.changed_attributes & toBits(GaussianAttrFlag::ROTATION)) {
        auto q=correction.rotation_wxyz.reshape({1,4}).expand({ids.numel(),4});
        auto inverse=q.clone();inverse.slice(1,1,4).neg_();
        auto raw=value.slice(1,6,10).clone();raw.select(1,0).add_(1);
        raw=quaternion_utils::multiply(quaternion_utils::multiply(q,raw),inverse);
        raw.select(1,0).sub_(1);value.slice(1,6,10).copy_(raw);
      }
      table.index_copy_(0,ids,value);
      // Adam's diagonal second moment cannot be exactly rotated. Reset only
      // transported identities; preserve all unrelated rows and the table clock.
      for(auto* optimizer:optimizers)if(optimizer) {
        auto found=optimizer->state().find(table.unsafeGetTensorImpl());
        if(found==optimizer->state().end())continue;
        auto& state=static_cast<torch::optim::AdamParamState&>(*found->second);
        state.exp_avg().index_fill_(0,ids,0);state.exp_avg_sq().index_fill_(0,ids,0);
        if(state.max_exp_avg_sq().defined())state.max_exp_avg_sq().index_fill_(0,ids,0);
      }
    }
  }

 private:
  void resize(const torch::Tensor& state_rows,const torch::Tensor& value_rows,
              const std::vector<torch::optim::Adam*>& optimizers) {
    MappingAuditRange trace("topology/r3_resize_and_moments");
    torch::NoGradGuard guard;
    const auto old=size();
    const auto gather=[&](const torch::Tensor& value,const torch::Tensor& rows) {
      auto padded=torch::cat({value.detach(),torch::zeros({1,14},value.options())});
      return padded.index_select(0,torch::where(rows>=0,rows,old));
    };
    for(size_t j=0;j<tables_.size();++j) {
      auto& table=tables_[j];auto next=gather(table,value_rows).requires_grad_(table.requires_grad());
      for(auto* optimizer:optimizers)if(optimizer) {
        for(auto& group:optimizer->param_groups())for(auto& parameter:group.params())
          if(parameter.is_same(table))parameter=next;
        auto found=optimizer->state().find(table.unsafeGetTensorImpl());
        if(found==optimizer->state().end())continue;
        auto& state=static_cast<torch::optim::AdamParamState&>(*found->second);
        state.exp_avg(gather(state.exp_avg(),state_rows));state.exp_avg_sq(gather(state.exp_avg_sq(),state_rows));
        if(state.max_exp_avg_sq().defined())state.max_exp_avg_sq(gather(state.max_exp_avg_sq(),state_rows));
        auto migrated=std::move(found->second);optimizer->state().erase(found);
        optimizer->state()[next.unsafeGetTensorImpl()]=std::move(migrated);
      }
      table=next;parameters_["residual_"+std::to_string(j)]=table;
    }
    auto fresh=torch::nonzero(state_rows<0).reshape({-1});
    auto padded=torch::cat({uids_,torch::zeros({1},uids_.options())});
    auto ids=padded.index_select(0,torch::where(state_rows>=0,state_rows,old));
    const auto begin=next_uid_.item<int64_t>();
    ids.index_copy_(0,fresh,torch::arange(begin,begin+fresh.numel(),uids_.options()));
    uids_.set_data(ids);next_uid_.add_(fresh.numel());
  }
  bool independent_;
  std::vector<double> values_;
  std::vector<torch::Tensor> tables_;
  torch::Tensor knots_,uids_,next_uid_;
};
TORCH_MODULE(GaussianCompensationBank);

}  // namespace improvements::selector
