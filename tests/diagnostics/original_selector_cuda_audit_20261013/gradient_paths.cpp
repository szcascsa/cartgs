#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include "improvements/selector/gumbel_network.h"
#include "include/gaussian_renderer.h"
#include "include/transform_field.h"
#include "include/loss_utils.h"
#include "include/elastic_budget.h"
#include "include/gi_teacher.h"

using improvements::selector::GumbelNetwork;
torch::Tensor read(const std::string& path) {
  std::ifstream f(path,std::ios::binary);int64_t n,w;f.read(reinterpret_cast<char*>(&n),8);f.read(reinterpret_cast<char*>(&w),8);std::vector<float> v(n*w);f.read(reinterpret_cast<char*>(v.data()),v.size()*4);TORCH_CHECK(f,"input");return torch::from_blob(v.data(),{n,w}).clone().to(torch::kCUDA);
}
void gradient(std::ofstream& csv,const std::string& test,const std::string& module,const std::string& name,const torch::Tensor& parameter) {
  auto g=parameter.grad();bool defined=g.defined();double norm=defined?g.norm().item<double>():0,frac=defined?(g!=0).to(torch::kFloat32).mean().item<double>():0;bool finite=!defined||torch::isfinite(g).all().item<bool>();
  csv<<test<<','<<module<<','<<name<<','<<parameter.numel()<<','<<defined<<','<<norm<<','<<frac<<','<<finite<<'\n';
}
int main(int argc,char** argv) {
  TORCH_CHECK(argc==2,"gradient_paths DIAGNOSTICS");torch::set_num_threads(1);std::string dir=argv[1];
  const std::string checkpoint="/home/prmi/zwj/mycartgs/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply";
  auto data=read(dir+"/input.bin"),meta=read(dir+"/metadata.bin");int64_t n=data.size(0),mapping;std::ifstream(dir+"/spec.txt")>>mapping;
  std::vector<float> camera_data(37);std::ifstream f(dir+"/camera.bin",std::ios::binary);f.read(reinterpret_cast<char*>(camera_data.data()),37*4);
  auto camera=std::make_shared<GaussianKeyframe>(0);camera->world_view_transform_=torch::from_blob(camera_data.data(),{4,4}).clone().cuda();camera->full_proj_transform_=torch::from_blob(camera_data.data()+16,{4,4}).clone().cuda();camera->camera_center_=torch::from_blob(camera_data.data()+32,{3}).clone().cuda();camera->FoVx_=camera_data[35];camera->FoVy_=camera_data[36];
  auto gt=read(dir+"/gt.bin").reshape({3,240,320});
  GaussianPipelineParams pipe;auto background=torch::zeros({3},data.options()),override_color=torch::Tensor();
  auto mature=(mapping-meta.select(1,0)>=100)&(meta.select(1,1)>=80)&(meta.select(1,2)>=4);auto ids=torch::nonzero(mature).flatten();
  auto conf=torch::minimum(torch::minimum(((mapping-meta.select(1,0))/100).clamp(0,1),(meta.select(1,1)/80).clamp(0,1)),(meta.select(1,2)/4).clamp(0,1));
  ElasticBudgetManager budget_manager;budget_manager.performance_logging=false;budget_manager.useUniform(meta.select(1,3));auto population=budget_manager.population(mature,1-conf,ids.numel());ProtectionConfig protection;protection.performance_logging=false;const float ratio=.05;auto budget=budget_manager.plan(population,ratio,protection);
  std::ofstream csv(dir+"/gradient_paths.csv"),summary(dir+"/gradient_losses.csv");csv<<"test,module,parameter,numel,grad_defined,norm,nonzero_fraction,finite\n";summary<<"test,N,M,K,visible,tf,loss,finite\n";
  torch::Tensor frame_gi;
  for(const std::string test:{"ratio","gi","photo","photo_tf","combined","full_photo","photo_detach","guard_uniform","cd_probability","dsd_bce","tf_regularization"}) {
    torch::manual_seed(123);GumbelNetwork network;torch::serialize::InputArchive a;a.load_from(checkpoint+"/selector.pt",torch::kCPU);network->load(a);network->to(torch::kCUDA);
    auto leaf=[&](int64_t begin,int64_t end){return data.slice(1,begin,end).contiguous().detach().set_requires_grad(true);};
    auto x=leaf(0,3),qr=leaf(3,7),sr=leaf(7,10),op=leaf(10,11),dc=leaf(11,14).reshape({n,1,3}).detach().set_requires_grad(true),rest=leaf(14,data.size(1)).reshape({n,3,-1}).transpose(1,2).contiguous().detach().set_requires_grad(true);
    auto q=torch::nn::functional::normalize(qr),s=sr.exp(),alpha=op.sigmoid();
    bool detached=test=="photo_detach"||test=="cd_probability"||test=="dsd_bce";
    GaussianRenderInput full{detached?x.detach():x,detached?alpha.detach():alpha,detached?s.detach():s,detached?q.detach():q,detached?dc.detach():dc,detached?rest.detach():rest,3,3};
    if(!frame_gi.defined()){torch::NoGradGuard no_grad;frame_gi=GaussianRenderer::render(camera,240,320,full,pipe,background,override_color,1,false,true,RasterizeMode::Inference).frame_importance.detach();auto cpu=frame_gi.cpu().contiguous();std::ofstream out(dir+"/frame_gi.bin",std::ios::binary);out.write(reinterpret_cast<const char*>(cpu.data_ptr<float>()),cpu.numel()*4);}
    auto xm=full.xyz.index_select(0,ids),qm=full.rotation.index_select(0,ids),sm=full.scaling.index_select(0,ids);
    torch::Tensor probability,hard,loss;int64_t k=0,visible=0,tf_count=0;TransformField field{nullptr};
    if(test=="cd_probability"||test=="dsd_bce") {
      auto teacher=network->forwardProbability(xm,qm,sm,ratio).detach();
      // Simulate a changed teacher target while keeping the real detached boundary.
      teacher=(teacher+.01).clamp(0,1);
      if(test=="cd_probability")loss=(network->forwardProbability(xm,qm,sm,ratio)-teacher).abs().mean();
      else loss=torch::nn::functional::binary_cross_entropy_with_logits(network->forwardKeepLogit(xm,qm,sm,ratio),teacher);
    } else if(test=="full_photo") {
      auto rendered=GaussianRenderer::render(camera,240,320,full,pipe,background,override_color);visible=rendered.visibility_indices.numel();k=n;loss=.8*loss_utils::l1_loss(rendered.image,gt)+.2*(1-loss_utils::fast_ssim(rendered.image,gt));
    } else {
      if(test=="guard_uniform")hard=network->forwardHardWithUniform(xm,qm,sm,ratio,torch::rand({ids.numel()},data.options()));
      else hard=network->forwardHard(xm,qm,sm,ratio,1,&probability);
      auto selector_hard=torch::zeros({n},data.options()).index_copy(0,ids,hard);auto pf=budget.admitted.to(torch::kFloat32);auto mask=pf+(1-pf)*selector_hard;
      auto selected=torch::nonzero(mask.detach()>0).flatten();k=selected.numel();
      if(test=="ratio")loss=.01*torch::abs(probability.sum()-budget.target.mature)/std::max(1.0,budget.target.total);
      else if(test=="gi") {auto teacher=GITeacherBuilder().build(frame_gi,mature,~mature,budget.mature_ratio,1,1,ids.numel(),torch::ones_like(mature));auto pred=selector_hard.index({teacher.supervised_mask});auto target=teacher.label.index({teacher.supervised_mask});loss=loss_utils::l1_loss(pred,target);}
      else {
        auto compact=full.indexSelect(selected);compact.opacity=compact.opacity*mask.index_select(0,selected).unsqueeze(1);
        if(test=="photo_tf"||test=="combined"||test=="tf_regularization") {
          TransformFieldConfig cfg;int levels;std::ifstream spec(dir+"/tf_fixture_spec.txt");spec>>cfg.hexplane.spatial_resolution>>cfg.hexplane.ratio_resolution>>cfg.hexplane.feature_dim>>cfg.hidden_dim>>levels;cfg.hexplane.multires.resize(levels);for(auto& v:cfg.hexplane.multires)spec>>v;std::vector<float> lo(3),hi(3);for(auto& v:lo)spec>>v;for(auto& v:hi)spec>>v;TORCH_CHECK(spec,"TF fixture spec");cfg.backend="legacy";
          field=TransformField(torch::tensor(lo),torch::tensor(hi),cfg);torch::serialize::InputArchive tf_archive;tf_archive.load_from(checkpoint+"/transform_field.pt",torch::kCPU);field->load(tf_archive);field->to(torch::kCUDA);
          auto tf_ids=torch::nonzero(mature.index_select(0,selected)&field->insideMask(compact.xyz)).flatten();tf_count=tf_ids.numel();auto transformed=field->applyResidual(compact.xyz.index_select(0,tf_ids),compact.scaling.index_select(0,tf_ids),compact.rotation.index_select(0,tf_ids),ratio);compact.xyz=compact.xyz.index_copy(0,tf_ids,transformed.xyz);compact.scaling=compact.scaling.index_copy(0,tf_ids,transformed.scaling);compact.rotation=compact.rotation.index_copy(0,tf_ids,transformed.rotation);
        }
        if(test=="tf_regularization")loss=field->regularizationLoss();
        else {auto rendered=GaussianRenderer::render(camera,240,320,compact,pipe,background,override_color);visible=rendered.visibility_indices.numel();loss=.8*loss_utils::l1_loss(rendered.image,gt)+.2*(1-loss_utils::fast_ssim(rendered.image,gt));}
        if(test=="combined") {auto teacher=GITeacherBuilder().build(frame_gi,mature,~mature,budget.mature_ratio,1,1,ids.numel(),torch::ones_like(mature));auto pred=selector_hard.index({teacher.supervised_mask});auto target=teacher.label.index({teacher.supervised_mask});loss=loss+.01*torch::abs(probability.sum()-budget.target.mature)/std::max(1.0,budget.target.total)+loss_utils::l1_loss(pred,target)+field->regularizationLoss();}
      }
    }
    TORCH_CHECK(loss.requires_grad(),"missing loss graph ",test);loss.backward();
    summary<<test<<','<<n<<','<<ids.numel()<<','<<k<<','<<visible<<','<<tf_count<<','<<loss.item<double>()<<','<<torch::isfinite(loss).item<bool>()<<'\n';summary.flush();
    for(auto& p:network->named_parameters())gradient(csv,test,"selector",p.key(),p.value());
    for(auto p:std::vector<std::pair<std::string,torch::Tensor>>{{"xyz",x},{"raw_rotation",qr},{"log_scale",sr},{"raw_opacity",op},{"SH_dc",dc},{"SH_rest",rest}})gradient(csv,test,"gaussian",p.first,p.second);
    if(field)for(auto& p:field->named_parameters())gradient(csv,test,"tf",p.key(),p.value());csv.flush();std::cout<<test<<" complete"<<std::endl;
  }
}
