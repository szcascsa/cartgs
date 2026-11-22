#pragma once
#include "latency_predictive_core.h"
#include <vector>

namespace latency_predictive {
struct Anchor {float x,y,z,half,count,radius2;};
class ViewSummary {
  std::vector<Anchor> anchors_;
  std::array<float,16> projection_{};
  double fx_=0.,fy_=0.;int width_=0,height_=0;
public:
  bool configured=false;
  void configure(std::vector<Anchor> anchors,const std::array<float,16>& projection,int width,int height){
    if(width<=0||height<=0)throw std::invalid_argument("Invalid view summary size");
    anchors_=std::move(anchors);projection_=projection;width_=width;height_=height;
    fx_=width*.5*projection[0];fy_=height*.5*projection[5];configured=true;
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
    for(const auto& anchor:anchors_){if(anchor.count<=0)continue;bool visible=true;
      for(int p=0;p<6;++p){const float* plane=planes+p*4;
        float d=std::fma(anchor.x,plane[0],0.f);d=std::fma(anchor.y,plane[1],d);d=std::fma(anchor.z,plane[2],d);
        if((d+plane[3])+anchor.half*norm[p]<0.f){visible=false;break;}}
      if(!visible)continue;
      const double depth=std::max(.01,double(std::fma(anchor.z,vm[10],std::fma(anchor.y,vm[6],std::fma(anchor.x,vm[2],vm[14])))));
      const double footprint=std::min(double(width_)*height_,(fx_*fx_+fy_*fy_)*anchor.radius2/(depth*depth));
      count+=anchor.count;area+=anchor.count*std::log1p(footprint)/8.;
    }
    return {int64_t(std::llround(count)),count?area/count:1.};
  }
};
}
