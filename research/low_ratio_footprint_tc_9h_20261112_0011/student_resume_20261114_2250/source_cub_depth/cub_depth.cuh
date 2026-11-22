__global__ void cubDepthInverse(int n,const uint32_t* sorted,uint32_t* inverse){
  const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)inverse[sorted[i]]=i;
}
