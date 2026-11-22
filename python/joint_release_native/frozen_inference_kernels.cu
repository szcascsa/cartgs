#include <torch/extension.h>
#include <cub/cub.cuh>
#include <math_constants.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <climits>

namespace {
void check(const torch::Tensor& tensor, const torch::Tensor& reference, at::ScalarType type=at::kFloat) {
  TORCH_CHECK(tensor.is_cuda() && tensor.device()==reference.device() && tensor.scalar_type()==type &&
              tensor.is_contiguous(), "Frozen inference needs contiguous same-device tensors");
}
struct Tables { const float* table[6]; int count; };
struct Heads { const float* part[4]; };
struct Render { const float* part[6]; int width[6]; };
struct VisibleRow {
  const bool* visible;const int64_t* owners;
  __device__ bool operator()(int64_t row) const {return visible[owners[row]];}
};

__global__ void visibleCounts(int64_t n,const bool* visible,const int64_t* eligible,int64_t* counts) {
  __shared__ int64_t total[256],available[256];
  int64_t a=0,b=0;
  for(int64_t i=threadIdx.x;i<n;i+=256)if(visible[i]){++a;b+=eligible[i];}
  total[threadIdx.x]=a;available[threadIdx.x]=b;__syncthreads();
  for(int stride=128;stride;stride/=2) {
    if(threadIdx.x<stride){total[threadIdx.x]+=total[threadIdx.x+stride];available[threadIdx.x]+=available[threadIdx.x+stride];}
    __syncthreads();
  }
  if(!threadIdx.x){counts[1]=total[0];counts[2]=available[0];}
}

__global__ void packSelector(int64_t n,const float* physical,const int64_t* ids,const float* ratio,float* output) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=n*11)return;
  auto row=i/11,col=i%11;
  output[i]=col<10 ? physical[ids[row]*10+col] : logf(ratio[0]);
}
__global__ void packSelectorView(int64_t n,const float* physical,const int64_t* ids,const int64_t* owners,
    const float* ratio,const float* code,bool local,bool real_view,float* output) {
  const auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=n*19)return;
  const auto row=i/19,col=i%19,id=ids[row];
  output[i]=col<10?physical[id*10+col]:col==10?logf(ratio[0]):
      col==18?(local?1.f:0.f):(local&&real_view?code[owners[id]*7+col-11]:0.f);
}
__global__ void packCondition(int64_t n,const int64_t* ids,const int64_t* owners,const float* code,
    bool real_view,float* output) {
  const auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=n*8)return;
  const auto row=i/8,col=i%8;
  output[i]=col==7?1.f:(real_view?code[owners[ids[row]]*7+col]:0.f);
}
__device__ float safeW(float value,float near_plane) {
  return fabsf(value)<near_plane?(value<0.f?-near_plane:near_plane):value;
}
__device__ float exactClamp(float value,float lo,float hi) {
  return isnan(value)?value:fminf(fmaxf(value,lo),hi);
}
__global__ void packViewCode(int64_t n,const float* direction,const float* camera,const float* clip,
    const float* corners,const float* depth,const float* scene_scale,float aspect,float near_plane,float* out) {
  const auto row=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(row>=n)return;
  for(int j=0;j<3;++j)out[row*7+j]=direction[row*3+j];
  out[row*7+3]=asinhf(__fdiv_rn(camera[row*4+2],scene_scale[0]));
  for(int j=0;j<2;++j)out[row*7+4+j]=exactClamp(__fdiv_rn(clip[row*4+j],safeW(clip[row*4+3],near_plane)),-4.f,4.f);
  float minimum[2],maximum[2],min_depth=depth[row*32+2],max_depth=min_depth;
  for(int j=0;j<2;++j)minimum[j]=maximum[j]=__fdiv_rn(corners[row*32+j],safeW(corners[row*32+3],near_plane));
  for(int k=1;k<8;++k) {
    float d=depth[row*32+k*4+2];
    if(d<min_depth || isnan(d))min_depth=d;
    if(d>max_depth || isnan(d))max_depth=d;
    for(int j=0;j<2;++j) {
      float v=__fdiv_rn(corners[row*32+k*4+j],safeW(corners[row*32+k*4+3],near_plane));
      if(v<minimum[j] || isnan(v))minimum[j]=v;
      if(v>maximum[j] || isnan(v))maximum[j]=v;
    }
  }
  float x=__fmul_rn(__fmul_rn(__fsub_rn(maximum[0],minimum[0]),aspect),.5f);
  float y=__fmul_rn(__fsub_rn(maximum[1],minimum[1]),.5f);
  float longest=exactClamp(isnan(x)||x>=y?x:y,0.f,1e4f);
  if(min_depth<=near_plane)longest=1e4f;
  if(max_depth<=near_plane)longest=0.f;
  out[row*7+6]=log1pf(longest);
}
__global__ void packCompensation(int64_t n,const float* offset,const int64_t* ids,const int64_t* owners,
    Tables features,const float* weights,float* output) {
  const int64_t row=int64_t(blockIdx.x)*4+(threadIdx.x>>6);
  const int col=threadIdx.x&63;
  if(row>=n || col>=51)return;
  const auto id=ids[row];
  if(col<3)output[row*51+col]=offset[id*3+col];
  else {
    const auto address=owners[id]*48+col-3;
    float value=features.table[0][address];
    for(int j=1;j<features.count;++j) {
      if(weights[j]<0.f)break; // -1 means inactive, preserves original <=knot+1e-8 rule.
      value=__fadd_rn(value,__fmul_rn(weights[j],features.table[j][address]));
    }
    output[row*51+col]=value;
  }
}
__global__ void addRelu(int64_t n,const float* a,const float* b,float* output) {
  const auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=n)return;
  float value=__fadd_rn(a[i],b[i%64]);output[i]=value<=0.f?0.f:value;
}
__global__ void addViewRelu(int64_t n,const float* a,const float* b,const float* view,float* output) {
  const auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=n)return;
  const float value=__fadd_rn(__fadd_rn(a[i],b[i%64]),view[i]);
  output[i]=value<=0.f?0.f:value;
}
__global__ void packHeads(int64_t n,Heads p,const float* reference,float* output) {
  const auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=n*14)return;
  const auto row=i/14,col=i%14;
  if(col<3)output[i]=__fmul_rn(p.part[0][row*3+col],reference[row]);
  else if(col<6)output[i]=p.part[1][row*3+col-3];
  else if(col<10)output[i]=p.part[2][row*4+col-6];
  else output[i]=p.part[3][row*4+col-10];
}
__global__ void publish(int64_t n,int64_t total,Render p,float* output) {
  auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=total)return;
  int64_t start=0;
  for(int j=0;j<6;++j) {
    int64_t length=n*p.width[j];
    if(i<start+length){output[i]=p.part[j][i-start];return;}
    start+=length;
  }
  output[i]=0.f; // Renderer means2D
}
__global__ void publishSelected(int64_t n,int64_t total,Render p,const int64_t* rows,
    const float* residual,const float* rotation,bool softplus,float* output) {
  auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=total)return;
  int64_t start=0;
  for(int j=0;j<6;++j) {
    int64_t length=n*p.width[j];
    if(i<start+length) {
      const auto local=i-start,row=local/p.width[j],col=local%p.width[j];
      float value=p.part[j][rows[row]*p.width[j]+col];
      if(residual) {
        if(j==0)value=__fadd_rn(value,residual[row*14+col]);
        else if(j==1) {
          float logit=__fadd_rn(value,residual[row*14+10]);
          value=__fdiv_rn(1.f,__fadd_rn(1.f,expf(-logit)));
        } else if(j==2) {
          float delta=residual[row*14+3+col];
          if(softplus) {
            float z=__fadd_rn(delta,0.5413248546129181f);
            float factor=z>20.f?z:log1pf(expf(z));
            value=__fmul_rn(value,factor);
            if(value<1.1754943508222875e-38f)value=1.1754943508222875e-38f;
          } else value=__fadd_rn(value,delta);
          value=expf(logf(value));
        } else if(j==3)value=rotation[row*4+col];
        else if(j==4)value=__fadd_rn(value,residual[row*14+11+col]);
      }
      output[i]=value;return;
    }
    start+=length;
  }
  output[i]=0.f;
}
__global__ void maskedScores(int n,const float* scores,const bool* eligible,float* keys,int64_t* values) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){keys[i]=eligible[i]?scores[i]:-CUDART_INF_F;values[i]=i;}
}
__global__ void packDisplay(int64_t pixels,int64_t count,const float* image,const float* reference,float* out) {
  const auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=pixels*3*count)return;
  const auto local=i%(pixels*3),pixel=local/3,channel=local%3;
  const float value=(i<pixels*3?image:reference)[channel*pixels+pixel];
  out[i]=exactClamp(value,0.f,1.f);
}
}

torch::Tensor selector_pack(const torch::Tensor& physical,const torch::Tensor& ids,const torch::Tensor& ratio) {
  TORCH_CHECK(!torch::GradMode::is_enabled(),"Frozen inference only");
  check(physical,physical);check(ids,physical,at::kLong);check(ratio,physical);
  TORCH_CHECK(physical.dim()==2 && physical.size(1)==10 && ratio.numel()==1,"Selector pack layout");
  c10::cuda::CUDAGuard guard(physical.device());
  auto output=torch::empty({ids.numel(),11},physical.options());
  if(ids.numel())packSelector<<<(output.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
    ids.numel(),physical.data_ptr<float>(),ids.data_ptr<int64_t>(),ratio.data_ptr<float>(),output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}

void display_pack(const torch::Tensor& image,const torch::Tensor& reference,const torch::Tensor& out) {
  TORCH_CHECK(!torch::GradMode::is_enabled(),"Frozen display packing only");
  check(image,image);check(out,image);
  TORCH_CHECK(image.dim()==3 && image.size(0)==3,"Display image must be contiguous FP32 CHW");
  const bool second=reference.defined() && reference.numel();
  if(second){check(reference,image);TORCH_CHECK(reference.sizes()==image.sizes(),"Display reference shape");}
  TORCH_CHECK(out.numel()==image.numel()*(second?2:1),"Display workspace shape");
  c10::cuda::CUDAGuard guard(image.device());
  if(out.numel())packDisplay<<<(out.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      image.size(1)*image.size(2),second?2:1,image.data_ptr<float>(),second?reference.data_ptr<float>():nullptr,out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor selector_view_pack(const torch::Tensor& physical,const torch::Tensor& ids,const torch::Tensor& owners,
    const torch::Tensor& ratio,const torch::Tensor& code,bool local,bool real_view) {
  TORCH_CHECK(!torch::GradMode::is_enabled(),"Frozen view inference only");
  check(physical,physical);check(ids,physical,at::kLong);check(owners,physical,at::kLong);check(ratio,physical);check(code,physical);
  TORCH_CHECK(physical.dim()==2 && physical.size(1)==10 && owners.numel()==physical.size(0) && ratio.numel()==1 &&
      (!local || !real_view || (code.dim()==2 && code.size(1)==7)),"Selector view pack layout");
  c10::cuda::CUDAGuard guard(physical.device());auto output=torch::empty({ids.numel(),19},physical.options());
  if(output.numel())packSelectorView<<<(output.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      ids.numel(),physical.data_ptr<float>(),ids.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),ratio.data_ptr<float>(),
      code.data_ptr<float>(),local,real_view,output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}

torch::Tensor condition_pack(const torch::Tensor& ids,const torch::Tensor& owners,const torch::Tensor& code,bool real_view) {
  TORCH_CHECK(!torch::GradMode::is_enabled(),"Frozen view inference only");
  check(code,code);check(ids,code,at::kLong);check(owners,code,at::kLong);
  TORCH_CHECK(!real_view || (code.dim()==2 && code.size(1)==7),"View condition pack layout");
  c10::cuda::CUDAGuard guard(code.device());auto output=torch::empty({ids.numel(),8},code.options());
  if(output.numel())packCondition<<<(output.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      ids.numel(),ids.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),code.data_ptr<float>(),real_view,output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}

torch::Tensor view_code_pack(const torch::Tensor& direction,const torch::Tensor& camera,const torch::Tensor& clip,
    const torch::Tensor& corners,const torch::Tensor& depth,const torch::Tensor& scene_scale,double aspect,double near_plane) {
  TORCH_CHECK(!torch::GradMode::is_enabled(),"Frozen view code only");
  for(const auto& t:{direction,camera,clip,corners,depth,scene_scale})check(t,direction);
  const auto n=direction.size(0);
  TORCH_CHECK(direction.sizes()==at::IntArrayRef({n,3}) && camera.sizes()==at::IntArrayRef({n,4}) && clip.sizes()==camera.sizes() &&
      corners.sizes()==at::IntArrayRef({n,8,4}) && depth.sizes()==corners.sizes() && scene_scale.numel()==1 && aspect>0 && near_plane>0,
      "View code layout");
  c10::cuda::CUDAGuard guard(direction.device());auto output=torch::empty({n,7},direction.options());
  if(n)packViewCode<<<(n+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(n,direction.data_ptr<float>(),camera.data_ptr<float>(),
      clip.data_ptr<float>(),corners.data_ptr<float>(),depth.data_ptr<float>(),scene_scale.data_ptr<float>(),float(aspect),float(near_plane),output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
torch::Tensor compensation_pack(const torch::Tensor& offset,const torch::Tensor& ids,const torch::Tensor& owners,
    const std::vector<torch::Tensor>& tables,const torch::Tensor& weights) {
  TORCH_CHECK(!torch::GradMode::is_enabled(),"Frozen inference only");
  check(offset,offset);check(ids,offset,at::kLong);check(owners,offset,at::kLong);check(weights,offset);
  TORCH_CHECK(offset.dim()==2 && offset.size(1)==3 && owners.numel()==offset.size(0) &&
      tables.size()>=1 && tables.size()<=6 && weights.numel()==int64_t(tables.size()),"Compensation pack layout");
  c10::cuda::CUDAGuard guard(offset.device());Tables features{};features.count=tables.size();
  for(int j=0;j<features.count;++j){check(tables[j],offset);TORCH_CHECK(tables[j].dim()==2 && tables[j].size(1)==48,"Feature48 layout");features.table[j]=tables[j].data_ptr<float>();}
  auto output=torch::empty({ids.numel(),51},offset.options());
  if(ids.numel())packCompensation<<<(ids.numel()+3)/4,256,0,c10::cuda::getCurrentCUDAStream()>>>(ids.numel(),
    offset.data_ptr<float>(),ids.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),features,weights.data_ptr<float>(),output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
torch::Tensor add_relu(const torch::Tensor& input,const torch::Tensor& ratio) {
  auto output=torch::empty_like(input);
  if(input.numel())addRelu<<<(input.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
    input.numel(),input.data_ptr<float>(),ratio.data_ptr<float>(),output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}

torch::Tensor add_view_relu(const torch::Tensor& input,const torch::Tensor& ratio,const torch::Tensor& view) {
  auto output=torch::empty_like(input);
  if(input.numel())addViewRelu<<<(input.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      input.numel(),input.data_ptr<float>(),ratio.data_ptr<float>(),view.data_ptr<float>(),output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
torch::Tensor head_pack(const std::vector<torch::Tensor>& parts,const torch::Tensor& reference) {
  Heads pointers{};for(int j=0;j<4;++j)pointers.part[j]=parts[j].data_ptr<float>();
  auto output=torch::empty({reference.numel(),14},reference.options());
  if(reference.numel())packHeads<<<(output.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
    reference.numel(),pointers,reference.data_ptr<float>(),output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
torch::Tensor publish_render(const std::vector<torch::Tensor>& parts) {
  TORCH_CHECK(parts.size()==6 && !torch::GradMode::is_enabled(),"Frozen render packet needs six attributes");
  c10::cuda::CUDAGuard guard(parts[0].device());
  auto n=parts[0].size(0);Render data{};int width=3;
  for(int j=0;j<6;++j){check(parts[j],parts[0]);TORCH_CHECK(parts[j].size(0)==n,"Render packet rows");
    data.part[j]=parts[j].data_ptr<float>();data.width[j]=n?parts[j].numel()/n:(j==0?3:0);width+=data.width[j];}
  auto output=torch::empty({n*width},parts[0].options());
  if(output.numel())publish<<<(output.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
    n,output.numel(),data,output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}

torch::Tensor publish_selected(const std::vector<torch::Tensor>& parts,const torch::Tensor& rows,
    const torch::Tensor& residual,const torch::Tensor& rotation,bool softplus) {
  TORCH_CHECK(parts.size()==6 && !torch::GradMode::is_enabled(),"Frozen selected packet requires six attributes");
  c10::cuda::CUDAGuard guard(parts[0].device());
  check(rows,parts[0],at::kLong);Render data{};int width=3;
  for(int j=0;j<6;++j) {
    check(parts[j],parts[0]);TORCH_CHECK(parts[j].size(0)==parts[0].size(0),"Selected Master rows");
    data.part[j]=parts[j].data_ptr<float>();
    int w=1;for(int d=1;d<parts[j].dim();++d)w*=parts[j].size(d);
    data.width[j]=w;width+=w;
  }
  bool compensate=residual.numel()!=0;
  if(compensate) {
    check(residual,parts[0]);check(rotation,parts[0]);
    TORCH_CHECK(residual.numel()==rows.numel()*14 && rotation.numel()==rows.numel()*4,"Selected residual layout");
  }
  auto output=torch::empty({rows.numel()*width},parts[0].options());
  if(output.numel())publishSelected<<<(output.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
    rows.numel(),output.numel(),data,rows.data_ptr<int64_t>(),compensate?residual.data_ptr<float>():nullptr,
    compensate?rotation.data_ptr<float>():nullptr,softplus,output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
torch::Tensor exact_select(const torch::Tensor& scores,const torch::Tensor& eligible,int64_t k,
    const std::vector<torch::Tensor>& workspace, bool narrow_indices) {
  TORCH_CHECK(!torch::GradMode::is_enabled(),"Frozen inference only");
  check(scores,scores);check(eligible,scores,at::kBool);
  const auto n=scores.numel();TORCH_CHECK(n<=INT_MAX && eligible.numel()==n && k>=0 && k<=n,"Selection layout");
  c10::cuda::CUDAGuard guard(scores.device());auto output=torch::empty({k},scores.options().dtype(at::kLong));
  if(!k)return output;
  TORCH_CHECK(workspace.empty() || workspace.size()==5,"Sort workspace layout");
  if(!workspace.empty()) {
    for(int j=0;j<5;++j)check(workspace[j],scores,j<2?at::kFloat:(j<4?at::kLong:at::kByte));
    for(int j=0;j<4;++j)TORCH_CHECK(workspace[j].numel()>=n,"Sort workspace capacity");
  }
  auto keys=workspace.empty()?torch::empty_like(scores):workspace[0];
  auto sorted_keys=workspace.empty()?torch::empty_like(scores):workspace[1];
  auto indices=workspace.empty()?torch::empty({n},output.options()):workspace[2];
  auto sorted_indices=workspace.empty()?torch::empty_like(indices):workspace[3];
  const auto stream=c10::cuda::getCurrentCUDAStream();
  maskedScores<<<(n+255)/256,256,0,stream>>>(n,scores.data_ptr<float>(),eligible.data_ptr<bool>(),keys.data_ptr<float>(),indices.data_ptr<int64_t>());
  int bits=64;
  if(narrow_indices){bits=1;for(auto v=n-1;v>>=1;)++bits;}
  size_t bytes=0,bytes2=0;
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(nullptr,bytes,keys.data_ptr<float>(),sorted_keys.data_ptr<float>(),indices.data_ptr<int64_t>(),sorted_indices.data_ptr<int64_t>(),int(n),0,32,stream));
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr,bytes2,sorted_indices.data_ptr<int64_t>(),output.data_ptr<int64_t>(),int(k),0,bits,stream));
  auto temp=workspace.empty()?torch::empty({int64_t(std::max(bytes,bytes2))},scores.options().dtype(at::kByte)):workspace[4];
  TORCH_CHECK(temp.numel()>=int64_t(std::max(bytes,bytes2)),"Sort temporary workspace capacity");
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(temp.data_ptr(),bytes,keys.data_ptr<float>(),sorted_keys.data_ptr<float>(),indices.data_ptr<int64_t>(),sorted_indices.data_ptr<int64_t>(),int(n),0,32,stream));
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(temp.data_ptr(),bytes2,sorted_indices.data_ptr<int64_t>(),output.data_ptr<int64_t>(),int(k),0,bits,stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}

std::vector<torch::Tensor> visible_candidates(const torch::Tensor& visible,const torch::Tensor& owners,
    const torch::Tensor& eligible_counts,const torch::Tensor& storage,const torch::Tensor& counts,
    const torch::Tensor& workspace) {
  TORCH_CHECK(!torch::GradMode::is_enabled(),"Frozen inference only");
  check(visible,visible,at::kBool);check(owners,visible,at::kLong);check(eligible_counts,visible,at::kLong);
  check(storage,visible,at::kLong);check(counts,visible,at::kLong);check(workspace,visible,at::kByte);
  TORCH_CHECK(visible.numel()==eligible_counts.numel() && storage.numel()>=owners.numel() &&
      counts.numel()==3 && owners.numel()<=INT_MAX,"Candidate workspace layout");
  c10::cuda::CUDAGuard guard(visible.device());const auto stream=c10::cuda::getCurrentCUDAStream();
  visibleCounts<<<1,256,0,stream>>>(visible.numel(),visible.data_ptr<bool>(),eligible_counts.data_ptr<int64_t>(),counts.data_ptr<int64_t>());
  auto temporary=workspace;
  if(owners.numel()) {
    cub::CountingInputIterator<int64_t> input(0);VisibleRow predicate{visible.data_ptr<bool>(),owners.data_ptr<int64_t>()};
    size_t bytes=0;
    C10_CUDA_CHECK(cub::DeviceSelect::If(nullptr,bytes,input,storage.data_ptr<int64_t>(),counts.data_ptr<int64_t>(),int(owners.numel()),predicate,stream));
    if(temporary.numel()<int64_t(bytes))temporary=torch::empty({int64_t(bytes)},workspace.options());
    C10_CUDA_CHECK(cub::DeviceSelect::If(temporary.data_ptr(),bytes,input,storage.data_ptr<int64_t>(),counts.data_ptr<int64_t>(),int(owners.numel()),predicate,stream));
  } else C10_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr<int64_t>(),0,sizeof(int64_t),stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {storage,counts,temporary};
}

namespace {
__global__ void makePlanes(const float* m,int64_t s0,int64_t s1,float* out) {
  const int i=threadIdx.x;
  if(i>=24)return;
  const int p=i/4,c=i%4;
  const float a=m[c*s0+3*s1],b=m[c*s0+(p/2)*s1];
  out[i]=p==4?m[c*s0+2*s1]:(p%2?__fsub_rn(a,b):__fadd_rn(a,b));
}
__global__ void testFrustum(int64_t n,const float* d,const float* planes,
    const float* scales,const bool* occupied,bool* out) {
  const auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=n)return;
  bool visible=occupied[i];
  for(int p=0;p<6;++p) {
    float norm=__fadd_rn(__fadd_rn(fabsf(planes[p*4]),fabsf(planes[p*4+2])),fabsf(planes[p*4+1]));
    float radius=__fmul_rn(__fmul_rn(scales[i],.5f),norm);
    float distance=__fadd_rn(d[i*6+p],planes[p*4+3]);
    visible=visible && __fadd_rn(distance,radius)>=0.f;
  }
  out[i]=visible;
}
__global__ void selectionScores(int64_t n,const float* logits,const float* noise,
    const int64_t* rows,float* output) {
  const auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=n)return;
  const auto r=rows[i];
  float a=__fadd_rn(logits[2*i],noise[2*r]),b=__fadd_rn(logits[2*i+1],noise[2*r+1]);
  float maximum=fmaxf(a,b);
  float ea=expf(__fsub_rn(a,maximum)),eb=expf(__fsub_rn(b,maximum));
  output[i]=__fdiv_rn(eb,__fadd_rn(ea,eb));
}
}
torch::Tensor frustum_planes(const torch::Tensor& projection) {
  TORCH_CHECK(projection.is_cuda() && projection.scalar_type()==at::kFloat && projection.sizes()==at::IntArrayRef({4,4}),"Frustum projection layout");
  c10::cuda::CUDAGuard guard(projection.device());
  auto out=torch::empty({6,4},projection.options());
  makePlanes<<<1,32,0,c10::cuda::getCurrentCUDAStream()>>>(projection.data_ptr<float>(),projection.stride(0),projection.stride(1),out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor frustum_test(const torch::Tensor& distance,const torch::Tensor& planes,
    const torch::Tensor& scales,const torch::Tensor& occupied) {
  check(distance,distance);check(planes,distance);check(scales,distance);check(occupied,distance,at::kBool);
  const auto n=scales.numel();
  TORCH_CHECK(distance.sizes()==at::IntArrayRef({n,6}) && occupied.numel()==n,"Frustum row layout");
  c10::cuda::CUDAGuard guard(distance.device());auto out=torch::empty({n},occupied.options());
  if(n)testFrustum<<<(n+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(n,distance.data_ptr<float>(),planes.data_ptr<float>(),scales.data_ptr<float>(),occupied.data_ptr<bool>(),out.data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor selection_scores(const torch::Tensor& logits,const torch::Tensor& noise,const torch::Tensor& rows) {
  TORCH_CHECK(!torch::GradMode::is_enabled(),"Frozen scores only");
  check(logits,logits);check(noise,logits);check(rows,logits,at::kLong);
  const auto n=rows.numel();TORCH_CHECK(logits.sizes()==at::IntArrayRef({n,2}) && noise.dim()==2 && noise.size(1)==2,"Selection score layout");
  c10::cuda::CUDAGuard guard(logits.device());auto out=torch::empty({n},logits.options());
  if(n)selectionScores<<<(n+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,logits.data_ptr<float>(),noise.data_ptr<float>(),rows.data_ptr<int64_t>(),out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

namespace {
__global__ void quaternionIdentity(int64_t n,const float* residual,float* out) {
  const auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n*4)out[i]=__fadd_rn(i%4==0?1.f:0.f,residual[(i/4)*14+6+i%4]);
}
__global__ void quaternionDivide(int64_t n,const float* q,const float* norm,float* out) {
  const auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n*4){float v=norm[i/4];out[i]=__fdiv_rn(q[i],isnan(v)?v:fmaxf(v,1e-12f));}
}
__global__ void quaternionMultiply(int64_t n,const float* q,const int64_t* rows,const float* master,float* out) {
  const auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=n)return;
  const auto a=q+i*4,b=master+rows[i]*4;
  float sum=__fadd_rn(__fadd_rn(__fmul_rn(a[1],b[1]),__fmul_rn(a[3],b[3])),__fmul_rn(a[2],b[2]));
  out[i*4]=__fsub_rn(__fmul_rn(a[0],b[0]),sum);
  for(int j=1;j<4;++j) {
    int k=j%3+1,l=(j+1)%3+1;
    float cross=__fmaf_rn(a[k],b[l],-__fmul_rn(a[l],b[k]));
    out[i*4+j]=__fadd_rn(__fadd_rn(__fmul_rn(a[0],b[j]),__fmul_rn(b[0],a[j])),cross);
  }
}
}
torch::Tensor quaternion_identity(const torch::Tensor& residual) {
  check(residual,residual);TORCH_CHECK(residual.dim()==2 && residual.size(1)==14,"Quaternion residual layout");
  auto out=torch::empty({residual.size(0),4},residual.options());
  if(out.numel())quaternionIdentity<<<(out.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(residual.size(0),residual.data_ptr<float>(),out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor quaternion_divide(const torch::Tensor& q,const torch::Tensor& norm) {
  auto out=torch::empty_like(q);
  if(out.numel())quaternionDivide<<<(out.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(q.size(0),q.data_ptr<float>(),norm.data_ptr<float>(),out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor quaternion_multiply(const torch::Tensor& q,const torch::Tensor& rows,const torch::Tensor& master) {
  check(q,q);check(rows,q,at::kLong);check(master,q);
  TORCH_CHECK(q.sizes()==at::IntArrayRef({rows.numel(),4}) && master.dim()==2 && master.size(1)==4,"Quaternion master layout");
  auto out=torch::empty_like(q);
  if(q.size(0))quaternionMultiply<<<(q.size(0)+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(q.size(0),q.data_ptr<float>(),rows.data_ptr<int64_t>(),master.data_ptr<float>(),out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

namespace {
__global__ void countRows(int64_t n,const int64_t* rows,const int64_t* owners,const int* radii,
    int64_t anchors,unsigned long long* counts) {
  const auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=n)return;
  const auto owner=owners[rows[i]];
  atomicAdd(counts+owner,1ULL);
  if(radii[i]>0)atomicAdd(counts+anchors+owner,1ULL);
}
__global__ void countAnchors(int64_t n,const bool* visible,const int64_t* members,const int64_t* levels,
    const bool* enabled,const unsigned long long* rows,unsigned long long* output) {
  const auto a=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(a>=n)return;
  const int level=levels[a];
  const bool active=members[a]>0 && (!enabled || enabled[a]);
  const bool seen=visible[a];
  if(active){atomicAdd(output+level,1ULL);atomicAdd(output+3+level,static_cast<unsigned long long>(members[a]));}
  if(seen){atomicAdd(output+6+level,1ULL);atomicAdd(output+9+level,static_cast<unsigned long long>(members[a]));}
  if(active&&seen){atomicAdd(output+12+level,1ULL);atomicAdd(output+15+level,static_cast<unsigned long long>(members[a]));}
  const auto submitted=rows[a],projected=rows[n+a];
  if(submitted){atomicAdd(output+18+level,1ULL);atomicAdd(output+24+level,submitted);}
  if(projected){atomicAdd(output+21+level,1ULL);atomicAdd(output+27+level,projected);}
}
}
torch::Tensor frame_counts(const torch::Tensor& visible,const torch::Tensor& members,const torch::Tensor& levels,
    const torch::Tensor& owners,const torch::Tensor& ids,const torch::Tensor& radii,const torch::Tensor& enabled) {
  TORCH_CHECK(!torch::GradMode::is_enabled(),"Frozen integer summary only");
  check(visible,visible,at::kBool);check(members,visible,at::kLong);check(levels,visible,at::kLong);
  check(owners,visible,at::kLong);check(ids,visible,at::kLong);check(radii,visible,at::kInt);
  const auto a=visible.numel(),n=ids.numel();
  TORCH_CHECK(members.numel()==a && levels.numel()==a && radii.numel()==n,"Frame summary layout");
  if(enabled.numel()){check(enabled,visible,at::kBool);TORCH_CHECK(enabled.numel()==a,"Summary mask layout");}
  c10::cuda::CUDAGuard guard(visible.device());auto scratch=torch::zeros({2*a+30},members.options());
  auto counts=reinterpret_cast<unsigned long long*>(scratch.data_ptr<int64_t>());
  if(n)countRows<<<(n+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(n,ids.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),radii.data_ptr<int>(),a,counts);
  if(a)countAnchors<<<(a+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(a,visible.data_ptr<bool>(),members.data_ptr<int64_t>(),levels.data_ptr<int64_t>(),enabled.numel()?enabled.data_ptr<bool>():nullptr,counts,counts+2*a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return scratch.slice(0,2*a).view({10,3});
}
