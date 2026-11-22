"""Student发布近似独立开关：省略Scale log-exp往返及冗余归一化；网络和权重不变。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'source_fast_publish';shutil.copytree(HERE/'source_joint_final',root);(HERE/'build_fast_publish').mkdir()
    path=root/'deployment_runtime_kernels.cu';text=path.read_text()
    text=text.replace('template<bool Direct=false,bool Rest=true> __device__ void publishPoint',
        'template<bool Direct=false,bool Rest=true,int Fast=0> __device__ void publishPoint')
    text=text.replace('  normalize(rotation);normalize(rotation);','  normalize(rotation);if constexpr(Fast<2)normalize(rotation);')
    text=text.replace('value=expf(logf(value));dest=n*4+', 'if constexpr(Fast==0)value=expf(logf(value));dest=n*4+')
    # 仅第三种发布候选使用快速exp，保留所有finite检查。
    text=text.replace('value=1.f/(1.f+expf(-(master.p[1][id]+d[10])));',
        'value=1.f/(1.f+(Fast>=3?__expf(-(master.p[1][id]+d[10])):expf(-(master.p[1][id]+d[10]))));')
    text=text.replace('log1pf(expf(z))','log1pf(Fast>=3?__expf(z):expf(z))')
    text=text.replace('compensation_mode!=9','compensation_mode<9').replace('compensation_mode==9','compensation_mode>=9')
    anchor='            compensationTensor<Mode,Warps,Direct,Projected><<<'
    line=next(line for line in text.splitlines() if line.startswith(anchor))
    extra=[]
    for mode,fast in [(10,1),(11,2),(12,3)]:
        extra.append('            '+('if' if mode==10 else 'else if')+f'(compensation_mode=={mode})'+line.strip().replace('compensationTensor<Mode,Warps,Direct,Projected>','compensationTensor<Mode,Warps,Direct,Projected,'+str(fast)+'>'))
    text=text.replace(line,'\n'.join(extra)+'\n            else '+line.strip())
    path.write_text(text)
    path=root/'tensor_networks.cuh';text=path.read_text()
    text=text.replace('template<int Mode,int Warps,bool Direct=false,bool Projected=false>\n__global__ void compensationTensor',
        'template<int Mode,int Warps,bool Direct=false,bool Projected=false,int Fast=0>\n__global__ void compensationTensor')
    text=text.replace('publishPoint<Direct,false>','publishPoint<Direct,false,Fast>');path.write_text(text)
    for name in ['deployment_prepare.cpp','raster/graph_main.cu']:
        path=root/name;text=path.read_text().replace('mycartgs.student-joint-final.shared.v1','mycartgs.student-fast-publish.shared.v1')
        text=text.replace('StudentJointFinalRasterizer','StudentFastPublishRasterizer').replace('compensator_mode<=9','compensator_mode<=12');path.write_text(text)
    print('FAST_PUBLISH_SOURCE_READY')


if __name__=='__main__':main()
