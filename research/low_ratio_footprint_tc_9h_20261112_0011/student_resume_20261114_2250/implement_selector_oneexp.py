"""Selector二分类softmax只计算非零差的exp；保留概率、finite检查和严格K。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'source_selector_oneexp';shutil.copytree(HERE/'source_joint_final',root);(HERE/'build_selector_oneexp').mkdir()
    path=root/'tensor_register.cuh';text=path.read_text()
    old='float mx=fmaxf(x,y),ea=expf(x-mx),eb=expf(y-mx);\n      scores[row]=eb/(ea+eb);'
    assert text.count(old)==1
    text=text.replace(old,'float e=expf(-fabsf(x-y));\n      scores[row]=y>=x?1.f/(1.f+e):e/(1.f+e);');path.write_text(text)
    for name in ['raster/graph_main.cu','deployment_prepare.cpp']:
        path=root/name;text=path.read_text().replace('mycartgs.student-joint-final.shared.v1','mycartgs.student-selector-oneexp.shared.v1')
        text=text.replace('StudentJointFinalRasterizer','StudentSelectorOneExpRasterizer');path.write_text(text)
    path=root/'deployment_selection.cu';text=path.read_text()+'''
__global__ void selectorScoreParityKernel(int64_t n,const float* a,const float* b,float* old_score,float* new_score){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
  float x=a[i],y=b[i],mx=fmaxf(x,y),ea=expf(x-mx),eb=expf(y-mx);
  old_score[i]=eb/(ea+eb);float e=expf(-fabsf(x-y));new_score[i]=y>=x?1.f/(1.f+e):e/(1.f+e);
}
std::vector<torch::Tensor> selector_score_parity(const torch::Tensor& a,const torch::Tensor& b){
  TORCH_CHECK(a.is_cuda()&&b.is_cuda()&&a.scalar_type()==at::kFloat&&b.scalar_type()==at::kFloat&&a.is_contiguous()&&b.is_contiguous()&&a.numel()==b.numel(),"Selector parity FP32 CUDA");
  auto x=torch::empty_like(a),y=torch::empty_like(a);
  selectorScoreParityKernel<<<(a.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(a.numel(),a.data_ptr<float>(),b.data_ptr<float>(),x.data_ptr<float>(),y.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {x,y};
}
''';path.write_text(text)
    path=root/'deployment_runtime_bindings.cpp';text=path.read_text()
    anchor='PYBIND11_MODULE(TORCH_EXTENSION_NAME'
    assert anchor in text;text=text.replace(anchor,'std::vector<torch::Tensor> selector_score_parity(const torch::Tensor&,const torch::Tensor&);\n\n'+anchor)
    position=text.index('{',text.index(anchor));binding=text[text.index(anchor):position].split(',')[-1].strip().rstrip(') ').strip()
    text=text[:position+1]+f'\n  {binding}.def("_selector_score_parity",&selector_score_parity);'+text[position+1:];path.write_text(text)
    print('SELECTOR_ONEEXP_SOURCE_READY')


if __name__=='__main__':main()
