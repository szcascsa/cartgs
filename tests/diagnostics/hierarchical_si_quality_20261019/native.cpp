#include <torch/torch.h>
#include <ATen/Context.h>
#include "include/gaussian_renderer.h"
#include "include/transform_field.h"
#include "include/loss_utils.h"
#include <json/json.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
using T=torch::Tensor;
Json::Value readJson(const std::string& path){std::ifstream f(path);Json::Value v;f>>v;return v;}
class FrozenScene {
  std::shared_ptr<GaussianModel> model_;
  GaussianRenderInput raw_;
  GaussianPipelineParams pipe_;
  TransformField tf_{nullptr};
  T gt_,world_,proj_,center_,params_,bg_,color_,maturity_;
  std::vector<T> originals_,copies_;
  std::shared_ptr<GaussianKeyframe> camera(int i){
    auto c=std::make_shared<GaussianKeyframe>(i);c->world_view_transform_=world_[i];c->full_proj_transform_=proj_[i];c->camera_center_=center_[i];
    c->FoVx_=params_[i][2].item<float>();c->FoVy_=params_[i][3].item<float>();return c;
  }
  RenderPackage render(const GaussianRenderInput& x,int frame,bool importance){
    return GaussianRenderer::render(camera(frame),params_[frame][0].item<int>(),params_[frame][1].item<int>(),x,pipe_,bg_,color_,1.f,false,importance,RasterizeMode::Inference);
  }
 public:
  FrozenScene(const Json::Value& cfg){
    model_=std::make_shared<GaussianModel>(3);model_->loadPly(cfg["master"].asString());
    originals_={model_->xyz_,model_->features_dc_,model_->features_rest_,model_->opacity_,model_->scaling_,model_->rotation_};
    raw_={model_->getXYZ().detach(),model_->getOpacityActivation().detach(),model_->getScalingActivation().detach(),model_->getRotationActivation().detach(),model_->features_dc_.detach(),model_->features_rest_.detach(),model_->active_sh_degree_,model_->max_sh_degree_};
    maturity_=torch::ones({raw_.xyz.size(0)},raw_.xyz.options());bg_=torch::zeros({3},raw_.xyz.options());
    cv::FileStorage f(cfg["config"].asString(),cv::FileStorage::READ);TORCH_CHECK(f.isOpened(),"Missing config");
    TransformFieldConfig c;c.backend=std::string(f["TransformField.backend"]);c.hexplane.spatial_resolution=int(f["TransformField.spatial_resolution"]);c.hexplane.ratio_resolution=int(f["TransformField.ratio_resolution"]);c.hexplane.feature_dim=int(f["TransformField.feature_dim"]);c.hexplane.multires.clear();for(auto v:f["TransformField.multires"])c.hexplane.multires.push_back(int(v));c.hidden_dim=int(f["TransformField.hidden_dim"]);
    c.maturity_compute_epsilon=0;c.elastic_entry_maturity=0;
    tf_=TransformField(torch::zeros({3}),torch::ones({3}),c);tf_->to(torch::kCUDA);torch::serialize::InputArchive a;a.load_from(cfg["tf"].asString(),torch::kCUDA);tf_->load(a);tf_->eval();
    for(auto& t:tf_->parameters())originals_.push_back(t);for(auto& t:tf_->buffers())originals_.push_back(t);
    for(auto& t:originals_){t.set_requires_grad(false);copies_.push_back(t.clone());}
    torch::serialize::InputArchive cams;cams.load_from(cfg["cameras"].asString(),torch::kCUDA);auto get=[&](const std::string& n){T t;cams.read(n,t,true);return t;};
    gt_=get("gt");world_=get("world");proj_=get("projection");center_=get("center");params_=get("params").cpu();
  }
  int64_t count(){return raw_.xyz.size(0);}
  void verify(){for(size_t i=0;i<originals_.size();++i)TORCH_CHECK(originals_[i].equal(copies_[i]),"Frozen tensor changed ",i);TORCH_CHECK(maturity_.eq(1).all().item<bool>(),"Maturity changed");}
  Json::Value command(const Json::Value& c,float* memory){
    Json::Value r;r["ok"]=true;auto method=c["method"].asString();
    if(method=="verify"){verify();r["N"]=Json::Int64(count());r["maturity_all_one"]=true;return r;}
    if(method=="export"){
      std::vector<T> contributions;for(auto i:c["bank_frames"]){auto p=render(raw_,i.asInt(),true);TORCH_CHECK(p.frame_importance.numel()==count(),"Bad contribution shape");contributions.push_back(p.frame_importance.cpu());}
      torch::serialize::OutputArchive out;auto put=[&](std::string n,T t){out.write(n,t.detach().cpu(),true);};
      put("xyz",raw_.xyz);put("scale",raw_.scaling);put("rotation",raw_.rotation);put("maturity",maturity_);put("contribution",torch::stack(contributions));out.save_to(c["output"].asString());return r;
    }
    TORCH_CHECK(method=="images","Unknown method");std::vector<int64_t> list;for(auto i:c["rows"])list.push_back(i.asInt64());auto rows=torch::tensor(list,torch::kInt64).cuda();auto x=raw_.indexSelect(rows);
    if(c["apply_tf"].asBool()){
      auto local=torch::nonzero(tf_->candidateMask(x.xyz,maturity_.index_select(0,rows))).flatten();
      if(local.numel()){
        auto a=tf_->applyResidual(x.xyz.index_select(0,local),x.scaling.index_select(0,local),x.rotation.index_select(0,local),c["ratio"].asFloat(),torch::ones({local.numel()},x.xyz.options()));
        x.xyz=x.xyz.index_copy(0,local,a.xyz);x.scaling=x.scaling.index_copy(0,local,a.scaling);x.rotation=x.rotation.index_copy(0,local,a.rotation);
      }
    }
    int k=0;for(auto j:c["frames"]){int frame=j.asInt();auto image=render(x,frame,false).image;
      auto truth=gt_[frame];auto loss=.8f*loss_utils::l1_loss(image,truth,1.f)+.2f*(1-loss_utils::fast_ssim(image,truth));r["photo"].append(loss.item<double>());
      auto cpu=image.cpu().contiguous();std::memcpy(memory+k*image.numel(),cpu.data_ptr<float>(),image.numel()*sizeof(float));++k;
    }return r;
  }
};
int main(int argc,char** argv){
  TORCH_CHECK(argc==3,"audit_worker input_manifest.json mmap_file");torch::set_num_threads(2);torch::NoGradGuard guard;at::globalContext().setAllowTF32CuBLAS(false);
  FrozenScene scene(readJson(argv[1]));size_t bytes=4*3*480*640*sizeof(float);int fd=open(argv[2],O_RDWR);TORCH_CHECK(fd>=0,"mmap open");void* memory=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);TORCH_CHECK(memory!=MAP_FAILED,"mmap map");
  for(std::string line;std::getline(std::cin,line);){Json::Value c;std::istringstream in(line);in>>c;if(c["method"].asString()=="quit")break;auto r=scene.command(c,static_cast<float*>(memory));Json::StreamWriterBuilder w;w["indentation"]="";std::cout<<"AUDIT_RESPONSE "<<Json::writeString(w,r)<<std::endl;}
  scene.verify();munmap(memory,bytes);close(fd);
}
