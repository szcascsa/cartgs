"""在独立副本接入q80校准和alpha=1.5；默认仍为q90、alpha=2。"""
from pathlib import Path
import difflib
import hashlib
import json

BASE_THRESHOLD=0.0
QUANTILE=0.8
ALPHA=1.5
MODEL=Path('include/gaussian_model.h')
MAPPER=Path('include/gaussian_mapper.h')
GROWTH=Path('improvements/selector/explicit_anchor_blocks.h')
BANK=Path('improvements/selector/anchor_bank.h')
CPP=Path('src/gaussian_model.cpp')
MAPPER_CPP=Path('src/gaussian_mapper.cpp')
CONFIG_APPEND=('\n# 首个有效L0窗口q80校准一次后固定，严格超过上界细化。\n'
 'Selector.anchor_growth_threshold: 0.0\nSelector.anchor_growth_alpha: 1.5\n'
 'Selector.anchor_growth_quantile: 0.8\nSelector.anchor_growth_strict_upper: 1\n')


def replace(text,before,after,count=1):
    if text.count(before)!=count:
        raise ValueError('源码与核查版本不符或补丁已应用: '+before[:100])
    return text.replace(before,after)


def patch_files(files):
    changed=dict(files)
    changed[MODEL]=replace(files[MODEL],
      '  double anchorGrowthThreshold() const { return anchor_growth_threshold_; }',
      '''  double anchorGrowthThreshold() const { return anchor_growth_threshold_; }
  double anchorGrowthAlpha() const { return anchor_growth_alpha_; }
  bool anchorGrowthManual() const { return anchor_growth_manual_; }
  double anchorGrowthQuantile() const { return anchor_growth_quantile_; }
  bool anchorGrowthStrictUpper() const { return anchor_growth_strict_upper_; }
  void setAnchorGrowthPolicy(double threshold, double alpha, double quantile=.9, bool strict_upper=false);''')
    changed[MODEL]=replace(changed[MODEL],'  double anchor_growth_threshold_ = 0.0;',
      '''  double anchor_growth_threshold_ = 0.0;
  double anchor_growth_alpha_ = 2.0;
  bool anchor_growth_manual_ = false;
  double anchor_growth_quantile_ = .9;
  bool anchor_growth_strict_upper_ = false;''')
    changed[MAPPER]=replace(files[MAPPER],'  int selector_anchor_only_pruning_ = 0;',
      '''  double selector_anchor_growth_threshold_ = 0.0;
  double selector_anchor_growth_alpha_ = 2.0;
  double selector_anchor_growth_quantile_ = .9;
  bool selector_anchor_growth_strict_upper_ = false;
  int selector_anchor_only_pruning_ = 0;''')
    changed[BANK]=replace(files[BANK],
      '      const std::vector<torch::optim::Adam*>& optimizers = {}, bool refine_requires_full = false);',
      '      const std::vector<torch::optim::Adam*>& optimizers = {}, bool refine_requires_full = false, bool strict_upper = false);')
    changed[GROWTH]=replace(files[GROWTH],
      '    const std::vector<torch::optim::Adam*>& optimizers, bool refine_requires_full) {',
      '    const std::vector<torch::optim::Adam*>& optimizers, bool refine_requires_full, bool strict_upper) {')
    changed[GROWTH]=replace(changed[GROWTH],'    if(g[i]>=high) {',
      '    if(strict_upper ? g[i]>high : g[i]>=high) {')
    changed[GROWTH]=replace(changed[GROWTH],
      '           <<" same_candidates="<<same<<" fine_candidates="<<fine<<" max_level_limited="<<limited',
      '           <<" strict_upper="<<strict_upper<<" same_candidates="<<same<<" fine_candidates="<<fine<<" max_level_limited="<<limited')
    changed[CPP]=replace(files[CPP],'void GaussianModel::densifyAndPrune(float max_grad,',
      '''void GaussianModel::setAnchorGrowthPolicy(double threshold, double alpha, double quantile, bool strict_upper) {
  TORCH_CHECK(std::isfinite(threshold) && threshold>=0 && std::isfinite(alpha) && alpha>1 &&
              std::isfinite(quantile) && quantile>=0 && quantile<=1,
              "Invalid configured Anchor growth thresholds");
  TORCH_CHECK(xyz_.numel()==0,"Anchor growth policy must precede map initialization");
  anchor_growth_threshold_=threshold;anchor_growth_alpha_=alpha;anchor_growth_manual_=threshold>0;
  anchor_growth_quantile_=quantile;anchor_growth_strict_upper_=strict_upper || anchor_growth_manual_;
  std::cout<<"[Anchor Growth Policy] mode="<<(anchor_growth_manual_?"manual":"quantile")
           <<" tau0="<<threshold<<" quantile="<<quantile<<" alpha="<<alpha
           <<" strict_upper="<<anchor_growth_strict_upper_<<'\\n';
}

void GaussianModel::densifyAndPrune(float max_grad,''')
    changed[CPP]=replace(changed[CPP],
      'anchor_growth_denom_,anchor_growth_threshold_,2,\n            {optimizer_.get(),anchor_feature_optimizer_},anchor_refine_requires_full_)',
      'anchor_growth_denom_,anchor_growth_threshold_,anchor_growth_alpha_,\n            {optimizer_.get(),anchor_feature_optimizer_},anchor_refine_requires_full_,anchor_growth_strict_upper_)')
    changed[CPP]=replace(changed[CPP],
      'grads,denom_,anchor_growth_threshold_,2,\n          {optimizer_.get(),anchor_feature_optimizer_})',
      'grads,denom_,anchor_growth_threshold_,anchor_growth_alpha_,\n          {optimizer_.get(),anchor_feature_optimizer_},false,anchor_growth_strict_upper_)')
    changed[CPP]=replace(changed[CPP],'torch::quantile(values,.9)',
      'torch::quantile(values,anchor_growth_quantile_)',count=2)
    changed[CPP]=replace(changed[CPP],
      '        if(values.numel())anchor_growth_threshold_=torch::quantile(values,anchor_growth_quantile_).item<double>();',
      '''        if(values.numel()) {
          anchor_growth_threshold_=torch::quantile(values,anchor_growth_quantile_).item<double>();
          std::cout<<"[Explicit Anchor Calibration] quantile="<<anchor_growth_quantile_
                   <<" tau0="<<anchor_growth_threshold_<<" visible_positive_L0="<<values.numel()
                   <<" alpha="<<anchor_growth_alpha_<<'\\n';
        }''')
    changed[CPP]=replace(changed[CPP],
      '"[Explicit Anchor Calibration] Full-only native gradient q90 tau0="',
      '"[Explicit Anchor Calibration] Full-only native gradient quantile="<<anchor_growth_quantile_<<" tau0="')
    changed[CPP]=replace(changed[CPP],'<<" visible_positive_L0="<<values.numel()<<" alpha=2\\n";',
      '<<" visible_positive_L0="<<values.numel()<<" alpha="<<anchor_growth_alpha_<<\'\\n\';')
    changed[MAPPER_CPP]=replace(files[MAPPER_CPP],
      '      gaussians_->setAnchorTopologyPolicy(selector_anchor_only_pruning_,selector_anchor_refine_requires_full_);',
      '''      gaussians_->setAnchorTopologyPolicy(selector_anchor_only_pruning_,selector_anchor_refine_requires_full_);
      gaussians_->setAnchorGrowthPolicy(selector_anchor_growth_threshold_,selector_anchor_growth_alpha_,
          selector_anchor_growth_quantile_,selector_anchor_growth_strict_upper_);''')
    changed[MAPPER_CPP]=replace(changed[MAPPER_CPP],
      '  int anchor_only_pruning=real_lod_.enabled?1:0,refine_requires_full=0;',
      '''  const auto growth_threshold=settings_file["Selector.anchor_growth_threshold"];
  const auto growth_alpha=settings_file["Selector.anchor_growth_alpha"];
  const auto growth_quantile=settings_file["Selector.anchor_growth_quantile"];
  int growth_strict_upper=0;readInt("Selector.anchor_growth_strict_upper",growth_strict_upper);
  if(growth_strict_upper!=0 && growth_strict_upper!=1)
    throw std::invalid_argument("Anchor growth strict_upper must be 0 or 1");
  selector_anchor_growth_strict_upper_=growth_strict_upper!=0;
  if(!growth_threshold.empty())growth_threshold >> selector_anchor_growth_threshold_;
  if(!growth_alpha.empty())growth_alpha >> selector_anchor_growth_alpha_;
  if(!growth_quantile.empty())growth_quantile >> selector_anchor_growth_quantile_;
  if(!std::isfinite(selector_anchor_growth_threshold_) || selector_anchor_growth_threshold_<0 ||
      !std::isfinite(selector_anchor_growth_alpha_) || selector_anchor_growth_alpha_<=1 ||
      !std::isfinite(selector_anchor_growth_quantile_) || selector_anchor_growth_quantile_<0 || selector_anchor_growth_quantile_>1)
    throw std::invalid_argument("Anchor growth threshold must be >=0, alpha >1 and quantile in [0,1]; all finite");
  int anchor_only_pruning=real_lod_.enabled?1:0,refine_requires_full=0;''')
    changed[MAPPER_CPP]=replace(changed[MAPPER_CPP],
      '        anchor_state["growth"]= "Full-gradient q90 first-window tau0; alpha=2; stable UID collision buckets";',
      '''        anchor_state["growth"]=gaussians_->anchorGrowthManual()
            ? "Full-gradient fixed tau0; geometric level thresholds; stable UID collision buckets"
            : "Full-gradient first valid L0 window quantile; frozen tau0; geometric level thresholds; stable UID collision buckets";
        anchor_state["growth_threshold_mode"]=gaussians_->anchorGrowthManual()?"manual":"quantile";
        anchor_state["growth_quantile"]=gaussians_->anchorGrowthQuantile();
        anchor_state["growth_strict_upper"]=gaussians_->anchorGrowthStrictUpper();''')
    changed[MAPPER_CPP]=replace(changed[MAPPER_CPP],
      '        anchor_state["growth_alpha"] = 2.0;',
      '        anchor_state["growth_alpha"] = gaussians_->anchorGrowthAlpha();')
    before='  settings_file << "Selector.anchor_refine_requires_full: " << int(selector_anchor_refine_requires_full_) << \'\\n\';'
    changed[MAPPER_CPP]=replace(changed[MAPPER_CPP],before,before+'''
  settings_file << "Selector.anchor_growth_threshold: " << selector_anchor_growth_threshold_ << '\\n';
  settings_file << "Selector.anchor_growth_alpha: " << selector_anchor_growth_alpha_ << '\\n';''')
    last='  settings_file << "Selector.anchor_growth_alpha: " << selector_anchor_growth_alpha_ << \'\\n\';'
    changed[MAPPER_CPP]=replace(changed[MAPPER_CPP],last,last+'''
  settings_file << "Selector.anchor_growth_quantile: " << selector_anchor_growth_quantile_ << '\\n';
  settings_file << "Selector.anchor_growth_strict_upper: " << int(selector_anchor_growth_strict_upper_) << '\\n';''')
    return changed


FILES=[MODEL,MAPPER,BANK,GROWTH,CPP,MAPPER_CPP]


def apply(source,reports):
    source,reports=Path(source),Path(reports)
    before={p:(source/p).read_text(encoding='utf-8') for p in FILES}
    after=patch_files(before)
    reports.mkdir(parents=True,exist_ok=True)
    diff=[];hashes={}
    for p in FILES:
        (source/p).write_text(after[p],encoding='utf-8')
        diff.extend(difflib.unified_diff(before[p].splitlines(True),after[p].splitlines(True),
             fromfile='baseline/'+p.as_posix(),tofile='experiment/'+p.as_posix()))
        hashes[p.as_posix()]={k:hashlib.sha256(v.encode()).hexdigest() for k,v in [('before',before[p]),('after',after[p])]}
    receipt=dict(changed_files=hashes,manual_threshold=BASE_THRESHOLD,quantile=QUANTILE,alpha=ALPHA,strict_upper=True,
         automatic_default_threshold=0,automatic_default_alpha=2,automatic_default_strict_upper=False,
         calibration='first valid L0 finite positive Full-gradient window; quantile=0.8; frozen afterwards',
         gradient_statistic='per-Gaussian temporal mean of Full screen-space XY gradient norm',
         spatial_level_scale_unchanged=True,topology_schedule_unchanged=True)
    (reports/'GROWTH_THRESHOLD.diff').write_text(''.join(diff),encoding='utf-8')
    (reports/'GROWTH_THRESHOLD_CHANGE.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return receipt
