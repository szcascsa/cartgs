#include "include/gaussian_mapper.h"
#include <cstdlib>
#include <iostream>

class FullTailGateTest : public GaussianMapper {
 public:
  FullTailGateTest(const std::filesystem::path& mapping,const std::filesystem::path& selector,
                   const std::filesystem::path& out)
      : GaussianMapper(nullptr,mapping,out,0,torch::kCUDA,true,selector) {}
  void verify(int expected_prune) {
    TORCH_CHECK(real_lod_.enabled&&!ultrafast_rotation_.enabled,"Full-only online schedule");
    TORCH_CHECK(selector_anchor_only_pruning_==expected_prune,"Pruning config mismatch");
    TORCH_CHECK(!record_rendered_image_&&!record_ground_truth_image_&&!record_loss_image_,"Images must not be saved");
    TORCH_CHECK(!tail_optimization_schedule_.enabled&&selector_max_mapping_iterations_==0,"Natural Full tail");
    TORCH_CHECK(selector_anchor_growth_threshold_==0 && selector_anchor_growth_alpha_==1.5 &&
                selector_anchor_growth_quantile_==.8 && selector_anchor_growth_strict_upper_,"Configured policy mismatch");
    TORCH_CHECK(gaussians_->anchorGrowthThreshold()==0 && gaussians_->anchorGrowthAlpha()==1.5 &&
                !gaussians_->anchorGrowthManual() && gaussians_->anchorGrowthQuantile()==.8 &&
                gaussians_->anchorGrowthStrictUpper(),"Configured policy did not reach actual GaussianModel");
    real_lod_.online_steps=7;real_lod_.tail_steps=3;real_lod_.phase=RealLodSchedule::Phase::TailFull;
    runRealLodOptimization();
    TORCH_CHECK(real_lod_.phase==RealLodSchedule::Phase::TailFull && real_lod_.lod_steps==0 &&
        real_lod_.target_steps==0 && real_lod_full_renders_==0 && real_lod_full_backwards_==0 &&
        flexgs_lod_master_updates_==0 && flexgs_lod_selector_updates_==0 && selector_step_==0 &&
        flexgs_importance_version_==0,"Skip path performed unexpected Fine work");
    TORCH_CHECK(gaussians_->anchorGrowthThreshold()==0,"No-training gate must not calibrate threshold");
    std::cout<<"PASS config/Full-tail-only gate; quantile=0.8 alpha=1.5 strict_upper=1; train=0 render=0 backward=0 optimizer=0\n";
  }
};

int main(int argc,char** argv) {
  TORCH_CHECK(argc==5,"Usage: gate mapping selector output prune");
  const char* enabled=std::getenv("CARTGS_BATCH_FULL_TAIL_ONLY");
  TORCH_CHECK(enabled&&std::string(enabled)=="1","No-train gate must be explicitly enabled");
  if(!torch::cuda::is_available())return 77;
  torch::set_num_threads(2);
  FullTailGateTest mapper(argv[1],argv[2],argv[3]);mapper.verify(std::stoi(argv[4]));
}
