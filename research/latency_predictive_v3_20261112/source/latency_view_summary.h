#pragma once
#include "latency_predictive_core.h"
#include <vector>
#include <immintrin.h>

namespace latency_predictive {
struct Anchor {float x,y,z,half,count,radius2;};
class ViewSummary {
  struct Node {
    float lo[3],hi[3];double count=0.,center[3]{},radius2=0.;
    int begin=0,end=0,left=-1,right=-1;
  };
  std::vector<Anchor> anchors_;
  std::vector<Node> nodes_;
  std::array<std::vector<float>,6> soa_;
  std::array<float,16> projection_{};
  double fx_=0.,fy_=0.;int width_=0,height_=0;
  int build(int begin,int end){
    Node node;node.begin=begin;node.end=end;
    for(int c=0;c<3;++c){node.lo[c]=INFINITY;node.hi[c]=-INFINITY;}
    for(int i=begin;i<end;++i){const auto& a=anchors_[i];const float p[3]={a.x,a.y,a.z};
      node.count+=a.count;node.radius2+=a.count*a.radius2;
      for(int c=0;c<3;++c){node.lo[c]=std::min(node.lo[c],p[c]-a.half);node.hi[c]=std::max(node.hi[c],p[c]+a.half);
        node.center[c]+=a.count*p[c];}}
    for(int c=0;c<3;++c)node.center[c]/=std::max(1.,node.count);
    node.radius2/=std::max(1.,node.count);
    const int id=nodes_.size();nodes_.push_back(node);
    if(end-begin>16){int axis=0;for(int c=1;c<3;++c)if(node.hi[c]-node.lo[c]>node.hi[axis]-node.lo[axis])axis=c;
      int mid=(begin+end)/2;
      std::nth_element(anchors_.begin()+begin,anchors_.begin()+mid,anchors_.begin()+end,[axis](const Anchor& a,const Anchor& b){
        return (axis==0?a.x:axis==1?a.y:a.z)<(axis==0?b.x:axis==1?b.y:b.z);});
      nodes_[id].left=build(begin,mid);nodes_[id].right=build(mid,end);
    }
    return id;
  }
public:
  bool configured=false;
  size_t bytes()const{size_t total=anchors_.capacity()*sizeof(Anchor)+nodes_.capacity()*sizeof(Node);
    for(const auto& a:soa_)total+=a.capacity()*sizeof(float);return total;}
  void configure(std::vector<Anchor> anchors,const std::array<float,16>& projection,int width,int height){
    if(width<=0||height<=0)throw std::invalid_argument("Invalid view summary size");
    anchors_=std::move(anchors);projection_=projection;width_=width;height_=height;
    fx_=width*.5*projection[0];fy_=height*.5*projection[5];configured=true;
    anchors_.erase(std::remove_if(anchors_.begin(),anchors_.end(),[](const Anchor& a){return a.count<=0;}),anchors_.end());
    nodes_.reserve(anchors_.size()/4+1);if(!anchors_.empty())build(0,anchors_.size());
    const size_t padded=(anchors_.size()+7)/8*8;
    for(auto& a:soa_)a.resize(padded,0.f);
    for(size_t i=0;i<anchors_.size();++i){const auto& a=anchors_[i];
      soa_[0][i]=a.x;soa_[1][i]=a.y;soa_[2][i]=a.z;soa_[3][i]=a.half;soa_[4][i]=a.count;soa_[5][i]=a.radius2;}
  }
  View evaluate(const std::vector<double>& pose)const{
    if(!configured||pose.size()!=16)throw std::invalid_argument("Configure immutable view summary and pass host pose");
    double a[4][8]{};
    for(int r=0;r<4;++r)for(int c=0;c<4;++c){a[r][c]=pose[r*4+c];a[r][c+4]=r==c?1.:0.;}
    for(int c=0;c<4;++c){int pivot=c;
      for(int r=c+1;r<4;++r)if(std::abs(a[r][c])>std::abs(a[pivot][c]))pivot=r;
      if(!std::isfinite(a[pivot][c])||a[pivot][c]==0.)throw std::invalid_argument("Singular predictive pose");
      if(pivot!=c)for(int j=0;j<8;++j)std::swap(a[pivot][j],a[c][j]);
      const double d=a[c][c];for(int j=0;j<8;++j)a[c][j]/=d;
      for(int r=0;r<4;++r)if(r!=c){const double v=a[r][c];for(int j=0;j<8;++j)a[r][j]-=v*a[c][j];}}
    float vm[16],pm[16],planes[24],norm[6];
    for(int r=0;r<4;++r)for(int c=0;c<4;++c)vm[r*4+c]=float(a[c][r+4]);
    for(int r=0;r<4;++r)for(int c=0;c<4;++c){float v=0.f;
      for(int k=0;k<4;++k)v=std::fma(vm[r*4+k],projection_[k*4+c],v);pm[r*4+c]=v;}
    for(int p=0;p<6;++p){for(int c=0;c<4;++c){const float x=pm[c*4+3],y=pm[c*4+p/2];
      planes[p*4+c]=p==4?pm[c*4+2]:(p%2?x-y:x+y);}
      norm[p]=std::abs(planes[p*4])+std::abs(planes[p*4+2])+std::abs(planes[p*4+1]);}
    double count=0.,area=0.;
    auto add=[&](double x,double y,double z,double weight,double radius2){
      const double depth=std::max(.01,double(std::fma(float(z),vm[10],std::fma(float(y),vm[6],std::fma(float(x),vm[2],vm[14])))));
      const double footprint=std::min(double(width_)*height_,(fx_*fx_+fy_*fy_)*radius2/(depth*depth));
      // 仅控制器的有界几何代理，避免每个Anchor调用log；不改变渲染数学。
      count+=weight;area+=weight*footprint/(footprint+64.);
    };
    alignas(32) float pp[4][8]{},guard[8]{};
    for(int p=0;p<6;++p){for(int c=0;c<4;++c)pp[c][p]=planes[p*4+c];guard[p]=.0001f*(norm[p]+1.f);}
    const auto sign=_mm256_set1_ps(-0.f);
    __m256 pl[4],ab[3];for(int c=0;c<4;++c)pl[c]=_mm256_load_ps(pp[c]);
    for(int c=0;c<3;++c)ab[c]=_mm256_andnot_ps(sign,pl[c]);
    const auto margin=_mm256_load_ps(guard);
    std::array<int,128> stack;int pending=0;if(!nodes_.empty())stack[pending++]=0;
    while(pending){const auto& node=nodes_[stack[--pending]];bool outside=false,inside=true;
      const float center[3]={(node.lo[0]+node.hi[0])*.5f,(node.lo[1]+node.hi[1])*.5f,(node.lo[2]+node.hi[2])*.5f};
      const float half[3]={(node.hi[0]-node.lo[0])*.5f,(node.hi[1]-node.lo[1])*.5f,(node.hi[2]-node.lo[2])*.5f};
      auto distance=_mm256_fmadd_ps(_mm256_set1_ps(center[0]),pl[0],_mm256_setzero_ps());
      distance=_mm256_fmadd_ps(_mm256_set1_ps(center[1]),pl[1],distance);
      distance=_mm256_add_ps(_mm256_fmadd_ps(_mm256_set1_ps(center[2]),pl[2],distance),pl[3]);
      auto radius=_mm256_mul_ps(_mm256_set1_ps(half[0]),ab[0]);
      radius=_mm256_add_ps(radius,_mm256_mul_ps(_mm256_set1_ps(half[1]),ab[1]));
      radius=_mm256_add_ps(radius,_mm256_mul_ps(_mm256_set1_ps(half[2]),ab[2]));
      outside=bool(_mm256_movemask_ps(_mm256_cmp_ps(_mm256_add_ps(distance,radius),_mm256_sub_ps(_mm256_setzero_ps(),margin),_CMP_LT_OQ))&63);
      inside=!(_mm256_movemask_ps(_mm256_cmp_ps(_mm256_sub_ps(distance,radius),margin,_CMP_LT_OQ))&63);
      if(outside)continue;
      if(inside){add(node.center[0],node.center[1],node.center[2],node.count,node.radius2);continue;}
      if(node.left>=0){stack[pending++]=node.left;stack[pending++]=node.right;continue;}
      double weight=0.,cx=0.,cy=0.,cz=0.,rr=0.;
      for(int i=node.begin;i<node.end;i+=8){
        const auto x=_mm256_loadu_ps(soa_[0].data()+i),y=_mm256_loadu_ps(soa_[1].data()+i),z=_mm256_loadu_ps(soa_[2].data()+i);
        const auto h=_mm256_loadu_ps(soa_[3].data()+i);int visible=255;
        for(int p=0;p<6;++p){const float* plane=planes+p*4;
          auto d=_mm256_fmadd_ps(x,_mm256_set1_ps(plane[0]),_mm256_setzero_ps());
          d=_mm256_fmadd_ps(y,_mm256_set1_ps(plane[1]),d);d=_mm256_fmadd_ps(z,_mm256_set1_ps(plane[2]),d);
          d=_mm256_add_ps(_mm256_add_ps(d,_mm256_set1_ps(plane[3])),_mm256_mul_ps(h,_mm256_set1_ps(norm[p])));
          visible&=_mm256_movemask_ps(_mm256_cmp_ps(d,_mm256_setzero_ps(),_CMP_GE_OQ));
        }
        for(int lane=0;lane<8&&i+lane<node.end;++lane)if(visible&(1<<lane)){
          const int j=i+lane;double w=soa_[4][j];weight+=w;cx+=w*soa_[0][j];cy+=w*soa_[1][j];cz+=w*soa_[2][j];rr+=w*soa_[5][j];}
      }
      if(weight)add(cx/weight,cy/weight,cz/weight,weight,rr/weight);
    }
    return {int64_t(std::llround(count)),count?area/count:1.};
  }
};
}
