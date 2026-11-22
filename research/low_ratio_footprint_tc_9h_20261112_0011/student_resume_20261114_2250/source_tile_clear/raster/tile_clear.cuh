template<int Threads> __global__ void clearTilePendingBits(uint32_t* bits,int words,int tiles,const GraphInputs* input){
  const int active=(input->P+31)/32,tile=blockIdx.x;
  if(!tile&&!threadIdx.x){bits[int64_t(tiles)*words]=0;bits[int64_t(tiles)*words+1]=0;}
  for(int word=threadIdx.x;word<active;word+=Threads)bits[int64_t(tile)*words+word]=0;
}
