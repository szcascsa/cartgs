#include "include/change_management.h"
#include "perf_scope.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/version.h>
#include <cuda_profiler_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

namespace fs = std::filesystem;
using Selector = improvements::selector::GumbelNetwork;
Json::Value readJson(const fs::path& path) {
  Json::Value value; std::ifstream file(path); TORCH_CHECK(file.good(), path.string()); file >> value; return value;
}
void writeJson(const fs::path& path, const Json::Value& value) {
  std::ofstream file(path); Json::StreamWriterBuilder builder; builder["indentation"] = "  "; file << Json::writeString(builder, value);
}
torch::Tensor readTensor(torch::serialize::InputArchive& archive, const std::string& name) {
  torch::Tensor tensor; archive.read(name, tensor); return tensor.to(torch::kCUDA).detach();
}
template<class M> void load(M& model, const fs::path& path) {
  torch::serialize::InputArchive archive; archive.load_from(path.string(), torch::kCUDA); model->load(archive);
}
template<class M> M student(const M& teacher) {
  M model(teacher->cloneFrozen()); for (auto p : model->parameters()) p.requires_grad_(true); model->train(); return model;
}
torch::optim::AdamOptions adamOptions(const Json::Value& value) {
  auto options = torch::optim::AdamOptions(value["lr"].asDouble());
  options.eps(value["eps"].asDouble()); options.betas(std::make_tuple(value["beta1"].asDouble(), value["beta2"].asDouble()));
  options.amsgrad(value["amsgrad"].asBool()); options.weight_decay(value["weight_decay"].asDouble()); return options;
}
struct Snapshot {
  Json::Value meta;
  GaussianAttrState before, after;
  torch::Tensor mature;
  AABB3D old_aabb, new_aabb;
  std::vector<CorrectionGroup> groups;
  TransformFieldConfig tf_config;
  Selector selector = nullptr;
  TransformField tf = nullptr;
  ChangeDistillConfig config;
  std::vector<float> ratios;
  explicit Snapshot(const fs::path& path) : meta(readJson(path / "event.json")) {
    torch::serialize::InputArchive archive; archive.load_from((path / "snapshot.pt").string(), torch::kCUDA);
    before = {readTensor(archive, "old_xyz"), readTensor(archive, "old_log_scale"), readTensor(archive, "old_rotation")};
    after = {readTensor(archive, "new_xyz"), readTensor(archive, "new_log_scale"), readTensor(archive, "new_rotation")};
    mature = readTensor(archive, "mature");
    old_aabb = {readTensor(archive, "old_aabb_min"), readTensor(archive, "old_aabb_max")};
    new_aabb = {readTensor(archive, "new_aabb_min"), readTensor(archive, "new_aabb_max")};
    for (Json::ArrayIndex i=0; i<meta["groups"].size(); ++i) {
      const auto name = "group_"+std::to_string(i); const auto& j=meta["groups"][i]; CorrectionGroup group;
      group.indices=readTensor(archive,name+"_indices"); group.transform_4x4=readTensor(archive,name+"_matrix");
      group.rotation_wxyz=readTensor(archive,name+"_quaternion"); group.scale=j["scale"].asFloat();
      group.changed_attributes=j["attributes"].asUInt(); group.keyframe_id=j["keyframe_id"].asUInt64(); groups.push_back(group);
    }
    const auto& t=meta["tf"]; tf_config.backend=t["backend"].asString();
    tf_config.hexplane.spatial_resolution=t["spatial_resolution"].asInt();
    tf_config.hexplane.ratio_resolution=t["ratio_resolution"].asInt(); tf_config.hexplane.feature_dim=t["feature_dim"].asInt();
    tf_config.hidden_dim=t["hidden_dim"].asInt(); tf_config.hexplane.multires.clear();
    for (const auto& x:t["multires"]) tf_config.hexplane.multires.push_back(x.asInt());
    tf_config.outside_identity=t["outside_identity"].asBool();
    selector=Selector(); selector->to(torch::kCUDA); selector->setBackend(meta["selector_backend"].asString()); load(selector,path/"selector.pt");
    tf=TransformField(old_aabb.min_xyz,old_aabb.max_xyz,tf_config); tf->to(torch::kCUDA); load(tf,path/"transform_field.pt");
    for(auto p:selector->parameters()) p.requires_grad_(false);
    for(auto p:tf->parameters()) p.requires_grad_(false);
    const auto& c=meta["cd"];
#define FLOAT_FIELD(k) config.k=c[#k].asFloat()
    FLOAT_FIELD(position_weight); FLOAT_FIELD(scale_weight); FLOAT_FIELD(rotation_weight); FLOAT_FIELD(selector_weight);
    FLOAT_FIELD(selector_learning_rate); FLOAT_FIELD(tf_grid_learning_rate); FLOAT_FIELD(tf_mlp_learning_rate);
    FLOAT_FIELD(anchor_ratio_to_corrected); FLOAT_FIELD(anchor_loss_weight);
#undef FLOAT_FIELD
    config.batch_size=c["batch_size"].asInt64(); config.anchor_max_count=c["anchor_max_count"].asInt64();
    config.enable_unaffected_anchors=c["enable_unaffected_anchors"].asBool();
    for(const auto& r:meta["ratios"]) ratios.push_back(r.asFloat());
  }
};
bool identical(const torch::nn::Module& a, const torch::nn::Module& b) {
  auto x=a.parameters(), y=b.parameters(); if(x.size()!=y.size()) return false;
  for(size_t i=0;i<x.size();++i) if(!torch::equal(x[i],y[i])) return false; return true;
}
Json::Value run(Snapshot& snapshot, const fs::path& output, const std::string& branch,
                int epochs, int batch, int ratios, int mode, const std::string& label) {
  fs::create_directories(output);
  cdperf::mode=0; cdperf::branch.clear();
  auto selector=student(snapshot.selector); auto tf=student(snapshot.tf);
  TORCH_CHECK(identical(*selector,*snapshot.selector)&&identical(*tf,*snapshot.tf), "Model restore failed");
  torch::manual_seed(0);
  auto selector_optimizer=torch::optim::Adam(selector->parameters(),adamOptions(snapshot.meta["selector_adam"][0]));
  std::vector<torch::optim::OptimizerParamGroup> tf_groups;
  tf_groups.emplace_back(tf->mlpParameters(), std::make_unique<torch::optim::AdamOptions>(adamOptions(snapshot.meta["tf_adam"][0])));
  tf_groups.emplace_back(tf->gridParameters(), std::make_unique<torch::optim::AdamOptions>(adamOptions(snapshot.meta["tf_adam"][1])));
  auto tf_optimizer=torch::optim::Adam(tf_groups);
  auto config=snapshot.config; config.epochs=epochs; config.selector_epochs=epochs; config.tf_epochs=epochs; config.batch_size=batch; config.seed=0;
  // Match the previous paired ablation's fixed TF branch RNG. Selector stays seed 0.
  // This is a microbenchmark input choice, outside production implementation.
  if(branch=="tf") config.seed=0x54465f5041495245ULL;
  config.distill_selector_on_geometry_change=branch!="tf";
  config.distill_tf_on_geometry_change=branch!="selector";
  config.distill_tf_on_aabb_change=branch!="selector";
  ChangeDistillManager manager(config); DynamicAABBManager domain; ChangeCoordinator coordinator(&domain,&manager);
  ChangeTrainingContext context; context.current=snapshot.after; context.selector=selector; context.tf=tf;
  context.normal_selector_optimizer=&selector_optimizer; context.normal_tf_optimizer=&tf_optimizer;
  context.ratios=std::vector<float>(snapshot.ratios.begin(),snapshot.ratios.begin()+ratios); context.temperature=snapshot.meta["temperature"].asDouble();
  AABBUpdatePlan plan; plan.should_expand=snapshot.meta["aabb_changed"].asBool(); plan.old_aabb=snapshot.old_aabb; plan.new_aabb=snapshot.new_aabb;
  const auto& planes=snapshot.meta["planes"];
  plan.affected_planes={planes["xy"].asBool(),planes["xz"].asBool(),planes["yz"].asBool(),planes["xe"].asBool(),planes["ye"].asBool(),planes["ze"].asBool()};
  cdperf::check(cudaDeviceSynchronize());
  cdperf::mode=mode; cdperf::branch="event";
  cdperf::tensor("event/old_xyz",snapshot.before.xyz);
  cdperf::tensor("event/current_xyz",snapshot.after.xyz);
  cdperf::tensor("event/current_scale",snapshot.after.log_scaling);
  cdperf::tensor("event/current_rotation",snapshot.after.rotation);
  cdperf::tensor("event/mature",snapshot.mature);
  cdperf::tensor("event/aabb_min",snapshot.old_aabb.min_xyz);
  const auto prep_start=cdperf::Clock::now();
  ChangeCapture capture;
  {
    cdperf::Scope scope("snapshot_and_teacher_preparation");
    capture=coordinator.captureBeforeGeometryChange(static_cast<ChangeFlag>(snapshot.meta["flags"].asUInt()),
        snapshot.mature,snapshot.before,snapshot.old_aabb,1,1,snapshot.meta["mapping_iter"].asUInt64());
    if(branch!="tf") { cdperf::Scope scope("clone_selector_teacher"); capture.teacher_selector=Selector(selector->cloneFrozen()); }
    if(branch!="selector") { cdperf::Scope scope("clone_tf_teacher"); capture.teacher_tf=TransformField(tf->cloneFrozen()); }
    capture.corrections=snapshot.groups;
  }
  cdperf::check(cudaDeviceSynchronize());
  const auto prep_ms=std::chrono::duration<double,std::milli>(cdperf::Clock::now()-prep_start).count();
  torch::profiler::impl::ProfilerConfig profiler_config(torch::profiler::impl::ProfilerState::KINETO,true,true,false,true,false);
  std::set<torch::profiler::impl::ActivityType> activities{torch::profiler::impl::ActivityType::CPU,torch::profiler::impl::ActivityType::CUDA};
  if(mode==2) { torch::autograd::profiler::prepareProfiler(profiler_config,activities); torch::autograd::profiler::enableProfiler(profiler_config,activities); }
  const auto xyz_before=snapshot.after.xyz.clone(), scale_before=snapshot.after.log_scaling.clone(), rotation_before=snapshot.after.rotation.clone();
  cdperf::check(cudaDeviceSynchronize()); c10::cuda::CUDACachingAllocator::resetPeakStats(0);
  const auto allocated_before=c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes[0].current;
  cudaEvent_t begin,end; cdperf::check(cudaEventCreate(&begin)); cdperf::check(cudaEventCreate(&end));
  if(mode) cudaProfilerStart();
  cdperf::check(cudaEventRecord(begin,c10::cuda::getCurrentCUDAStream())); const auto start=cdperf::Clock::now();
  ChangeDistillReport report;
  { cdperf::branch=branch; cdperf::Scope scope("coordinator_finalize");
    report=coordinator.finalizeGeometryChange(std::move(capture),context,1,&plan); }
  cdperf::check(cudaEventRecord(end,c10::cuda::getCurrentCUDAStream())); cdperf::check(cudaEventSynchronize(end));
  const auto cpu_ms=std::chrono::duration<double,std::milli>(cdperf::Clock::now()-start).count();
  float cuda_ms=0; cdperf::check(cudaEventElapsedTime(&cuda_ms,begin,end)); cudaEventDestroy(begin); cudaEventDestroy(end);
  if(mode) cudaProfilerStop();
  auto stats=c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  if(mode==2) { auto result=torch::autograd::profiler::disableProfiler(); result->save((output/"trace.json").string()); }
  if(mode) { cdperf::flush((output/"components.csv").string()); writeJson(output/"tensor_metadata.json",cdperf::tensor_metadata);
    writeJson(output/"optimizer_metadata.json",cdperf::optimizer_metadata); }
  cdperf::mode=0;
  TORCH_CHECK(torch::equal(xyz_before,snapshot.after.xyz)&&torch::equal(scale_before,snapshot.after.log_scaling)&&
              torch::equal(rotation_before,snapshot.after.rotation), "Gaussian changed");
  TORCH_CHECK(selector_optimizer.state().empty()&&tf_optimizer.state().empty(), "Normal optimizer changed");
  if(branch=="selector") {
    const auto expected=(std::max(report.selector_count,report.anchor_count)+batch-1)/batch*epochs;
    TORCH_CHECK(report.selector_steps==expected,"Unexpected Selector step count: ",report.selector_steps," expected ",expected);
  }
  if(branch=="tf") {
    const auto expected=(report.tf_count+batch-1)/batch*epochs;
    TORCH_CHECK(report.tf_steps==expected,"Unexpected TF step count: ",report.tf_steps," expected ",expected);
  }
  Json::Value row; row["label"]=label; row["branch"]=branch; row["epochs"]=epochs; row["batch_size"]=batch; row["ratio_count"]=ratios;
  row["instrumentation_mode"]=mode; row["cpu_wall_ms"]=cpu_ms; row["cuda_elapsed_ms"]=cuda_ms; row["event_preparation_ms"]=prep_ms;
  row["sampler_seed"]=Json::UInt64(config.seed);
  row["libtorch_version"]=TORCH_VERSION;
  row["cxx11_abi"]=_GLIBCXX_USE_CXX11_ABI;
  row["allow_tf32_cublas"]=at::globalContext().allowTF32CuBLAS();
  row["selector_steps"]=Json::Int64(report.selector_steps); row["tf_steps"]=Json::Int64(report.tf_steps);
  row["selector_count"]=Json::Int64(report.selector_count); row["anchor_count"]=Json::Int64(report.anchor_count);
  row["tf_count"]=Json::Int64(report.tf_count); row["skipped_outside"]=Json::Int64(report.skipped_outside);
  row["peak_allocated_bytes"]=Json::Int64(stats.allocated_bytes[0].peak); row["peak_reserved_bytes"]=Json::Int64(stats.reserved_bytes[0].peak);
  row["additional_allocated_bytes"]=Json::Int64(stats.allocated_bytes[0].peak-allocated_before);
  row["identical_initial_parameters"]=true; row["gaussian_unchanged"]=true; row["normal_optimizer_fresh_and_unchanged"]=true;
  row["selector_pre"]=report.selector_pre; row["selector_post"]=report.selector_post; row["tf_pre"]=report.tf_pre; row["tf_post"]=report.tf_post;
  if(label=="warm_0" || label=="profile") {
    torch::serialize::OutputArchive archive;
    size_t index=0; for(const auto& p:(branch=="selector"?selector->parameters():tf->parameters())) archive.write("p"+std::to_string(index++),p.detach());
    archive.save_to((output/"final_parameters.pt").string());
  }
  writeJson(output/"runtime.json",row); std::cout<<"[PERF RESULT] "<<label<<" cpu_ms="<<cpu_ms<<" cuda_ms="<<cuda_ms<<std::endl;
  return row;
}
int main(int argc,char** argv) {
  try {
    TORCH_CHECK(argc==10,"EVENT OUTPUT BRANCH EPOCHS BATCH RATIOS REPEATS MODE WARMUPS");
    torch::set_num_threads(4);
    Snapshot snapshot(argv[1]); const fs::path output(argv[2]); const std::string branch(argv[3]);
    const int epochs=std::stoi(argv[4]),batch=std::stoi(argv[5]),ratios=std::stoi(argv[6]),repeats=std::stoi(argv[7]),mode=std::stoi(argv[8]),warmups=std::stoi(argv[9]);
    TORCH_CHECK((branch=="selector"||branch=="tf"||branch=="both")&&epochs>0&&batch>0&&ratios>0&&ratios<=int(snapshot.ratios.size()),"Invalid benchmark case");
    fs::create_directories(output); Json::Value rows;
    rows.append(run(snapshot,output/"cold",branch,epochs,batch,ratios,0,"cold"));
    for(int i=0;i<warmups;++i) run(snapshot,output/("warmup_"+std::to_string(i)),branch,1,batch,ratios,0,"warmup");
    for(int i=0;i<repeats;++i) rows.append(run(snapshot,output/("warm_"+std::to_string(i)),branch,epochs,batch,ratios,mode,"warm_"+std::to_string(i)));
    if(mode>=2 && repeats==0) rows.append(run(snapshot,output/"profile",branch,epochs,batch,ratios,mode,"profile"));
    writeJson(output/"runs.json",rows); return 0;
  } catch(const std::exception& error) { std::cerr<<error.what()<<std::endl; return 1; }
}
