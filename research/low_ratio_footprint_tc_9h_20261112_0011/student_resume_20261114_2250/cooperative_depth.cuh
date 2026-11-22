// GPU实际P决定排序及归并次数；协作grid驻留上限由Occupancy查询确定。
__global__ void cooperativeDepthSort(int capacity,const uint32_t* input,uint64_t* first,uint64_t* second,
    uint32_t* output,uint32_t* ids,uint32_t* inverse,const GraphInputs* dynamic){
  const int n=min(capacity,dynamic->P);
  if(!n){if(!blockIdx.x&&!threadIdx.x)output[0]=0xffffffff;return;}
  using Sort=cub::BlockRadixSort<uint32_t,128,4,uint32_t>;
  __shared__ typename Sort::TempStorage storage;
  for(int block=blockIdx.x;block*512<n;block+=gridDim.x){
    uint32_t keys[4],values[4];
    #pragma unroll
    for(int j=0;j<4;++j){const unsigned i=block*512+threadIdx.x*4+j;keys[j]=i<unsigned(n)?input[i]:0xffffffff;values[j]=i;}
    Sort(storage).Sort(keys,values);
    #pragma unroll
    for(int j=0;j<4;++j){const unsigned i=block*512+threadIdx.x*4+j;if(i<unsigned(n)){
      if(n<=512){output[i]=keys[j];ids[i]=values[j];if(inverse)inverse[values[j]]=i;}
      else first[i]=(uint64_t(keys[j])<<32)|values[j];
    }}
    __syncthreads();
  }
  if(n<=512)return;
  auto grid=cooperative_groups::this_grid();grid.sync();
  uint64_t* src=first;uint64_t* dst=second;
  for(int width=512;width<n;width*=2){
    const bool final=width*2>=n;
    for(int index=blockIdx.x*blockDim.x+threadIdx.x;index<n;index+=gridDim.x*blockDim.x){
      const int base=(index/(width*2))*(width*2),diagonal=index-base;
      const int left=min(width,n-base),right=min(width,max(0,n-base-width));
      const uint64_t* A=src+base;const uint64_t* B=A+left;
      int lo=max(0,diagonal-right),hi=min(diagonal,left);
      while(lo<hi){const int i=(lo+hi)/2,j=diagonal-i;if(i<left&&j>0&&B[j-1]>=A[i])lo=i+1;else hi=i;}
      const int i=lo,j=diagonal-i;
      const uint64_t value=(i<left&&(j>=right||A[i]<B[j]))?A[i]:B[j];
      if(final){output[index]=value>>32;ids[index]=uint32_t(value);if(inverse)inverse[uint32_t(value)]=index;}
      else dst[index]=value;
    }
    if(!final)grid.sync();
    auto temporary=src;src=dst;dst=temporary;
  }
}
