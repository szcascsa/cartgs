"""精确候选：按实际有界发布容量决定Graph可用性，消除原32768上限造成的执行断点。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source_wide_graph'
    shutil.copytree(HERE / 'source_joint_final', root)
    (HERE / 'build_wide_graph').mkdir()
    path = root / 'deployment_prepare.cpp'
    text = path.read_text()
    anchor = '  bool small_radix_finish_=false;'
    text = text.replace(anchor, anchor + '\n  bool wide_graph_=false;')
    anchor = '''    const bool graph_capacity=!compensation || (use_ratio?
        (std::isfinite(ratio) && ratio*double(total_)<32769.):std::min(budget,total_)<=32768);'''
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''    const int64_t capacity_limit=wide_graph_?std::min<int64_t>(total_,(128ll<<20)/(width_*4+8)):32768;
    const bool graph_capacity=!compensation || (use_ratio?
        (std::isfinite(ratio) && ratio*double(total_)<double(capacity_limit)+1.):std::min(budget,total_)<=capacity_limit);''')
    anchor = '  void small_radix_finish(bool enabled){graphs_.clear();small_radix_finish_=enabled;}'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + '\n  void wide_graph(bool enabled){TORCH_CHECK(!pending_inflight_,"Finish pending frame before Graph limit change");graphs_.clear();wide_graph_=enabled;}')
    anchor = '.def("_small_radix_finish",&TensorDeploymentPipeline::small_radix_finish);'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '.def("_small_radix_finish",&TensorDeploymentPipeline::small_radix_finish)\n    .def("_wide_graph",&TensorDeploymentPipeline::wide_graph);')
    path.write_text(text)
    for name in ['deployment_prepare.cpp', 'raster/graph_main.cu']:
        path = root / name
        path.write_text(path.read_text().replace('mycartgs.student-joint-final.shared.v1',
            'mycartgs.student-wide-graph.shared.v1').replace('StudentJointFinalRasterizer', 'StudentWideGraphRasterizer'))
    print('WIDE_GRAPH_SOURCE_READY', flush=True)


if __name__ == '__main__':
    main()
