#include "include/anchor_si.h"
#include "include/anchor_stats_cuda.h"

namespace anchor_si {
namespace {
class Mean : public torch::autograd::Function<Mean> {
 public:
  static torch::autograd::tensor_list forward(torch::autograd::AutogradContext* ctx,
      torch::Tensor value,torch::Tensor owner,torch::Tensor members,
      torch::Tensor counts,torch::Tensor opacity,torch::Tensor inactive) {
    auto output=reduceForward({value,owner,members,counts,opacity,inactive});
    ctx->save_for_backward({value,owner,members,counts,opacity,inactive,output[1]});
    ctx->mark_non_differentiable({output[1],output[2]});ctx->set_materialize_grads(false);return output;
  }
  static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad) {
    if(!grad[0].defined())return {{},{},{},{},{},{}};
    auto saved=ctx->get_saved_variables();auto count=saved.back();saved.pop_back();
    return {reduceBackward(saved,count,grad[0].contiguous()),{},{},{},{},{}};
  }
};
}
std::vector<torch::Tensor> reduce(const torch::Tensor& v,const torch::Tensor& owner,
    const torch::Tensor& members,const torch::Tensor& counts,
    const torch::Tensor& opacity,const torch::Tensor& inactive) {
  auto mask=inactive.defined()?inactive:torch::empty({0},v.options().dtype(torch::kBool));
  return Mean::apply(v,owner,members,counts,opacity,mask);
}

void State::sync(const improvements::selector::GumbelNetwork& network,uint64_t version) {
  if(version_==version && visits_.defined())return;
  torch::NoGradGuard guard;auto bank=network->anchors();
  members_=bank->validatedMembers();
  if(!members_.defined())members_=mapping_cuda::validateAnchors(bank->rows(),bank->memberCounts(),bank->levels(),bank->centers(),bank->offsetScales());
  auto valid=torch::arange(10,bank->rows().options()).unsqueeze(0)<bank->memberCounts().unsqueeze(1);
  const auto sentinel=std::numeric_limits<int64_t>::max();
  auto padded=torch::cat({network->privateCompensation()->uids(),torch::full({1},sentinel,bank->rows().options())});
  auto indices=torch::where(valid,members_,torch::full_like(members_,padded.numel()-1));
  auto sorted=padded.index_select(0,indices.reshape({-1})).reshape_as(members_).sort(1);
  auto signature=torch::where(std::get<0>(sorted)==sentinel,-torch::ones_like(std::get<0>(sorted)),std::get<0>(sorted));
  // Slot bits describe stable Gaussian identities, never incidental row order.
  members_=members_.gather(1,std::get<1>(sorted)).contiguous();
  auto uid=bank->blockUIDs();auto options=bank->features().options().requires_grad(false);
  auto next_ema=torch::zeros({bank->size()},options),next_active=torch::zeros_like(next_ema),next_activity=torch::zeros_like(next_ema);
  auto next_visits=torch::zeros({bank->size()},uid.options()),next_last=torch::zeros_like(next_visits);
  if(visits_.defined() && uids_.numel()) {
    auto sorted=uids_.sort();auto sorted_uid=std::get<0>(sorted),order=std::get<1>(sorted);
    auto at=torch::searchsorted(sorted_uid,uid).clamp_max(uids_.numel()-1);
    auto old=order.index_select(0,at);
    auto keep=(sorted_uid.index_select(0,at)==uid)&(member_uids_.index_select(0,old)==signature).all(1);
    next_ema.copy_(torch::where(keep,ema_.index_select(0,old),next_ema));
    next_visits.copy_(torch::where(keep,visits_.index_select(0,old),next_visits));
    next_last.copy_(torch::where(keep,last_.index_select(0,old),next_last));
    next_active.copy_(torch::where(keep,active_.index_select(0,old),next_active));
    next_activity.copy_(torch::where(keep,activity_.index_select(0,old),next_activity));
  }
  uids_=uid.clone();member_uids_=signature;ema_=next_ema;visits_=next_visits;
  last_=next_last;active_=next_active;activity_=next_activity;labels_=torch::Tensor();version_=version;
}

void State::observe(const improvements::selector::GumbelNetwork& network,
    const torch::Tensor& contribution,const torch::Tensor& opacity,int64_t pixels,
    uint64_t version,const std::vector<float>& ratios) {
  torch::NoGradGuard guard;sync(network,version);auto bank=network->anchors();
  auto result=reduce(contribution,bank->rows(),members_,bank->memberCounts(),opacity,bank->inactiveSlots());
  auto value=result[0]/double(pixels),count=result[1];
  auto changed=(active_!=count)|(activity_!=result[2]);visits_.masked_fill_(changed,0);
  auto seen=(value>0)&torch::isfinite(value)&(count>0);
  auto update=torch::where(visits_>0,config_.decay*ema_+(1-config_.decay)*value,value);
  ema_.copy_(torch::where(seen,update,ema_));visits_.add_(seen.to(torch::kLong));
  last_.masked_fill_(seen,full_clock_);active_=count;activity_=result[2];
  auto knots=torch::tensor(ratios,ema_.options());
  labels_=targets(ema_,visits_,last_,active_,knots,full_clock_,contribution.numel(),
      config_.minimum_visits,config_.maximum_age_full,config_.temperature);
}

torch::Tensor State::loss(const improvements::selector::GumbelNetwork& network,
    const torch::Tensor& probability,const torch::Tensor& opacity,int ratio_index,uint64_t version) {
  sync(network,version);if(!labels_.defined())return {};
  auto bank=network->anchors();auto result=reduce(probability,bank->rows(),members_,bank->memberCounts(),opacity.detach(),bank->inactiveSlots());
  auto valid=(visits_>=config_.minimum_visits)&(full_clock_-last_<=config_.maximum_age_full)&
      (result[1]==active_)&(result[2]==activity_)&(active_>0)&(labels_[ratio_index]>=0);
  auto weights=valid.to(probability.scalar_type());
  return config_.weight*((result[0]-labels_[ratio_index].clamp_min(0)).square()*weights).sum()/weights.sum().clamp_min(1);
}

void State::accumulateHeadGradients(const torch::Tensor& loss,
    const improvements::selector::GumbelNetwork& network) const {
  if(!loss.defined())return;
  std::vector<torch::Tensor> parameters;
  for(auto& item:network->named_parameters())
    if(item.key().rfind("local_emd.2.",0)==0 || item.key().rfind("soft_net.",0)==0)parameters.push_back(item.value());
  auto gradients=torch::autograd::grad({loss},parameters,{},true,false,true);
  torch::NoGradGuard guard;
  for(size_t i=0;i<parameters.size();++i)if(gradients[i].defined()) {
    if(parameters[i].grad().defined())parameters[i].mutable_grad().add_(gradients[i]);
    else parameters[i].mutable_grad()=gradients[i].contiguous().clone();
  }
}
void State::save(const std::filesystem::path& file) const {
  if(!visits_.defined())return;torch::serialize::OutputArchive archive;
  archive.write("anchor_uids",uids_);archive.write("member_uids",member_uids_);
  archive.write("ema",ema_);archive.write("visits",visits_);archive.write("last",last_);archive.write("active",active_);archive.write("activity",activity_);
  archive.write("full_clock",torch::tensor(full_clock_,torch::kLong));archive.save_to(file.string());
}
}
