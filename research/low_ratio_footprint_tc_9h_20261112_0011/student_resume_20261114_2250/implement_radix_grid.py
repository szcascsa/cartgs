"""精确调度候选：有界CTA数降低cooperative grid barrier成本，保持严格K和稳定顺序。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    root = HERE / 'source_radix_grid'
    shutil.copytree(HERE / 'source_joint_final', root)
    (HERE / 'build_radix_grid').mkdir()
    path = root / 'deployment_radix.cu'
    text = path.read_text()
    text = replace_once(text, 'bool small_finish=false) {', 'bool small_finish=false,int64_t cta_limit=0) {')
    text = replace_once(text, 'const int blocks=std::min((n+255)/256,std::min(sm_count,blocks_per_sm*sm_count));',
        'const int blocks=std::min((n+255)/256,std::min(cta_limit?int(cta_limit):sm_count,blocks_per_sm*sm_count));')
    text = replace_once(text, 'bool saturated_fast,bool fused,bool small_finish) {',
        'bool saturated_fast,bool fused,bool small_finish,int64_t cta_limit) {')
    text = replace_once(text, 'saturated_fast,fused,small_finish);', 'saturated_fast,fused,small_finish,cta_limit);')
    path.write_text(text)
    path = root / 'deployment_runtime_kernels.cu'
    text = path.read_text()
    text = replace_once(text, 'const std::vector<torch::Tensor>&,bool,bool,bool);',
        'const std::vector<torch::Tensor>&,bool,bool,bool,int64_t);')
    text = replace_once(text, 'int64_t tensor_comp_warps,bool small_finish) {',
        'int64_t tensor_comp_warps,bool small_finish,int64_t radix_ctas) {')
    text = replace_once(text, 'fused_selection && n>=16384,small_finish);',
        'fused_selection && n>=16384,small_finish,radix_ctas);')
    path.write_text(text)
    path = root / 'deployment_prepare.cpp'
    text = path.read_text()
    text = replace_once(text, 'int64_t,int64_t,int64_t,int64_t,bool);',
        'int64_t,int64_t,int64_t,int64_t,bool,int64_t);')
    text = replace_once(text, '  bool small_radix_finish_=false;',
        '  bool small_radix_finish_=false;\n  int64_t radix_ctas_=0;')
    assert text.count('tensor_comp_warps_,small_radix_finish_);') == 2
    text = text.replace('tensor_comp_warps_,small_radix_finish_);', 'tensor_comp_warps_,small_radix_finish_,radix_ctas_);')
    anchor = '  void small_radix_finish(bool enabled){graphs_.clear();small_radix_finish_=enabled;}'
    text = replace_once(text, anchor, anchor + '\n  void radix_ctas(int64_t count){TORCH_CHECK(count>=0&&count<=128,"Radix CTA capacity");graphs_.clear();radix_ctas_=count;}')
    anchor = '.def("_small_radix_finish",&TensorDeploymentPipeline::small_radix_finish);'
    text = replace_once(text, anchor,
        '.def("_small_radix_finish",&TensorDeploymentPipeline::small_radix_finish)\n    .def("_radix_ctas",&TensorDeploymentPipeline::radix_ctas);')
    path.write_text(text)
    for name in ['deployment_prepare.cpp', 'raster/graph_main.cu']:
        path = root / name
        path.write_text(path.read_text().replace('mycartgs.student-joint-final.shared.v1',
            'mycartgs.student-radix-grid.shared.v1').replace('StudentJointFinalRasterizer', 'StudentRadixGridRasterizer'))
    print('RADIX_GRID_SOURCE_READY', flush=True)


if __name__ == '__main__':
    main()
