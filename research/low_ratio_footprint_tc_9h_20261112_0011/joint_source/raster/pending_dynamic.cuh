__global__ void blockDepthSort512Pending(int n,const uint32_t* input,uint64_t* temporary,
    uint32_t* output,uint32_t* ids,bool final,uint32_t* inverse,const GraphInputs* dynamic) {
  n=min(n,dynamic->P);
  if(!n){if(!blockIdx.x&&!threadIdx.x)output[0]=0xffffffff;return;}
  if(blockIdx.x*512>=unsigned(n))return;
  final=n<=512;

  using Sort=cub::BlockRadixSort<uint32_t,128,4,uint32_t>;
  __shared__ typename Sort::TempStorage storage;
  uint32_t keys[4],values[4];
  for(int j=0;j<4;++j) {
    unsigned i=blockIdx.x*512+threadIdx.x*4+j;
    keys[j]=i<unsigned(n)?input[i]:0xffffffff;values[j]=i;
  }
  // CUB BlockRadixSort文档保证稳定；padding位于所有真实输入之后。
  Sort(storage).Sort(keys,values);
  for(int j=0;j<4;++j) {
    unsigned i=blockIdx.x*512+threadIdx.x*4+j;
    if(i<unsigned(n)) {
      if(final){output[i]=keys[j];ids[i]=values[j];if(inverse)inverse[values[j]]=i;}
      else temporary[i]=(uint64_t(keys[j])<<32)|values[j];
    }
  }
}

__global__ void mergeDepthSortPending(int n,int width,const uint64_t* input,uint64_t* temporary,
    uint32_t* output,uint32_t* ids,bool final,uint32_t* inverse,const GraphInputs* dynamic) {
  n=min(n,dynamic->P);if(width>=n)return;final=width*2>=n;

  int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=n)return;
  int base=(index/(width*2))*(width*2),diagonal=index-base;
  int left=min(width,n-base),right=min(width,max(0,n-base-width));
  const uint64_t* A=input+base;const uint64_t* B=A+left;
  int lo=max(0,diagonal-right),hi=min(diagonal,left);
  while(lo<hi) {
    int i=(lo+hi)/2,j=diagonal-i;
    if(i<left&&j>0&&B[j-1]>=A[i])lo=i+1;else hi=i;
  }
  int i=lo,j=diagonal-i;
  uint64_t value=(i<left&&(j>=right||A[i]<B[j]))?A[i]:B[j];
  if(final){output[index]=value>>32;ids[index]=uint32_t(value);if(inverse)inverse[uint32_t(value)]=index;}else temporary[index]=value;
}

__device__ bool copyPendingBackground(const uint32_t* bits,int words,int tiles,int n,const uint32_t* sorted,
    const uint32_t* keys,const uint32_t* touched,int lane,const GraphInputs* dynamic=nullptr) {
  if(dynamic) {
    // 只有非敏感背景可跳过全图空成员证明：原C=0,T=1的输出与直接copy完全相同。
    // 不接受负零、denormal、NaN/Inf；这些仍走原检查。
    bool ordinary=true;
    #pragma unroll
    for(int ch=0;ch<3;++ch){
      const uint32_t raw=__float_as_uint(dynamic->background[ch]),mag=raw&0x7fffffffu;
      ordinary=ordinary&&(raw==0||(mag>=0x00800000u&&mag<0x7f800000u));
    }
    if(ordinary)return false;
  }
  n=min(n,dynamic->P);if(keys[0]==0xffffffff)return true;
  unsigned total=0;const int limit=min(n,257);
  for(int base=0;base<limit;base+=32) {
    int rank=base+lane;unsigned value=rank<limit&&keys[rank]!=0xffffffff?touched[sorted[rank]]:0;
    for(int offset=16;offset;offset/=2)value+=__shfl_down_sync(0xffffffff,value,offset);
    total+=__shfl_sync(0xffffffff,value,0);
    if(total>256)return false;
  }
  if(total==0)return true;
  const int visible_words=min((n+31)/32,8);
  for(int base=0;base<tiles*visible_words;base+=32) {
    int i=base+lane;bool nonempty=i<tiles*visible_words&&bits[(i/visible_words)*words+i%visible_words]!=0;
    if(__any_sync(0xffffffff,nonempty))return false;
  }
  return true;
}

__global__ void compactPendingBits(uint32_t* bits,int words,int tiles,const uint32_t* sorted,
    uint32_t* directory,int groups,uint32_t* lists,uint32_t* counts,int capacity,
    int n,const uint32_t* depth_keys,const uint32_t* touched,const GraphInputs* dynamic=nullptr) {
  const int active_words=(min(n,dynamic->P)+31)/32;
  int lane=threadIdx.x&31,tile=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
  if(tile>=tiles)return;
  unsigned carry=0;
  for(int group=0;group<groups;++group) {
    int w=group*32+lane;unsigned mask=w<active_words?bits[tile*words+w]:0;
    unsigned nonempty=__ballot_sync(0xffffffff,mask!=0);
    if(!lane)directory[tile*groups+group]=nonempty;
    unsigned count=__popc(mask),sum=count;
    for(int offset=1;offset<32;offset*=2){unsigned v=__shfl_up_sync(0xffffffff,sum,offset);if(lane>=offset)sum+=v;}
    unsigned position=carry+sum-count;
    for(unsigned remaining=mask;remaining;remaining&=remaining-1) {
      int rank=w*32+__ffs(remaining)-1;
      if(position<unsigned(capacity))lists[tile*capacity+position]=sorted[rank];
      ++position;
    }
    carry+=__shfl_sync(0xffffffff,sum,31);
  }
  if(!lane)counts[tile]=carry;
  if(tile==0) {
    bool copy=copyPendingBackground(bits,words,tiles,n,sorted,depth_keys,touched,lane,dynamic);
    if(!lane)bits[int64_t(tiles)*words+1]=copy;
  }
}

__global__ void clearPendingBits(uint32_t* bits,int words,int tiles,const GraphInputs* dynamic){
  const int i=blockIdx.x*blockDim.x+threadIdx.x,active_words=(dynamic->P+31)/32;
  if(!i){bits[int64_t(tiles)*words]=0;bits[int64_t(tiles)*words+1]=0;}
  if(i<int64_t(tiles)*active_words)bits[int64_t(i/active_words)*words+i%active_words]=0;
}