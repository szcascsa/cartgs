#include "decode.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mycartgs {
namespace {
std::array<float,4> Norm(std::array<float,4> q) {
    float n=0; for(float x:q)n+=x*x; n=std::max(std::sqrt(n),1e-12f); for(float&x:q)x/=n; return q;
}
std::array<float,4> Mul(const std::array<float,4>& a,const std::array<float,4>& b) {
    return {a[0]*b[0]-((a[1]*b[1]+a[2]*b[2])+a[3]*b[3]),
            a[0]*b[1]+b[0]*a[1]+(a[2]*b[3]-a[3]*b[2]),
            a[0]*b[2]+b[0]*a[2]+(a[3]*b[1]-a[1]*b[3]),
            a[0]*b[3]+b[0]*a[3]+(a[1]*b[2]-a[2]*b[1])};
}
}
DecodedAttribute DecodeResidual(const MasterAttribute& m,const std::array<float,14>& d,bool softplus) {
    for(float value:d)if(!std::isfinite(value))throw std::runtime_error("Nonfinite Student residual");
    DecodedAttribute out; for(int j=0;j<3;++j){out.xyz[j]=m.xyz[j]+d[j];out.dc[j]=m.dc[j]+d[11+j];
        const float v=d[3+j]+.5413248546129181f;
        out.scale[j]=softplus?std::max(m.scale[j]*(v>20?v:std::log1p(std::exp(v))),std::numeric_limits<float>::min()):m.scale[j]+d[3+j];
        if(!(out.scale[j]>0)||!std::isfinite(out.scale[j]))throw std::runtime_error("Invalid decoded scale");}
    out.rotation=Norm(Mul(Norm({1+d[6],d[7],d[8],d[9]}),m.rotation));out.rotation=Norm(out.rotation);
    out.opacity=1.f/(1.f+std::exp(-(m.opacity_logit+d[10])));
    return out;
}
}
