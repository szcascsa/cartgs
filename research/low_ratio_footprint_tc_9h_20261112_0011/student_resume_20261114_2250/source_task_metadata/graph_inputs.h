#pragma once
#include <cuda_runtime.h>

// 每次调用重新提交指针和标量；图计划只缓存调度，不缓存任何视角或Gaussian结果。
struct GraphInputs {
  int P,D,M,W,H;
  float focal_x,focal_y,tanx,tany,modifier;
  const float *xyz,*scales,*rotations,*opacity,*dc,*sh,*colors,*covariance;
  const float *view,*projection,*camera,*background;
  int v0,v1,p0,p1,c0;
  float* image;
  int* radii;
};

void graph_preprocess(int capacity,const GraphInputs* inputs,int* radii,float2* xy,float* depths,
    float* covariance,float* rgb,float4* conic,bool* clamped,uint32_t* touched,
    uint32_t* depth_keys,uint32_t* depth_ids,dim3 grid,cudaStream_t stream,int threads=256,uint2* tasks=nullptr,uint32_t* task_count=nullptr,uint4* footprint_rects=nullptr,float* footprint_limits=nullptr);
