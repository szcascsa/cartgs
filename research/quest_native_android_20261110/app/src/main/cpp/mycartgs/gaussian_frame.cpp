#include "gaussian_frame.h"
#include "point_benchmark.h"
#include "residual48.h"
#include "strict_topk.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace mycartgs {
namespace {
using Clock=std::chrono::steady_clock;
double Millis(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
float Clamp(float v,float lo,float hi){return std::max(lo,std::min(v,hi));}
std::array<float,4> Transform(const std::array<float,16>& m,const float* p) {
    std::array<float,4> out{};
    for(uint32_t r=0;r<4;++r)out[r]=((m[r*4]*p[0]+m[r*4+1]*p[1])+m[r*4+2]*p[2])+m[r*4+3];
    return out;
}
float Safe(float w,float near){return std::abs(w)<near?(w<0?-near:near):w;}
const float* F(const SceneAsset& scene,const char* name){return reinterpret_cast<const float*>(scene.Section(name).data);}
std::array<float,3> Color(const DecodedAttribute& a,const float* rest,const std::array<float,3>& camera) {
    float x=a.xyz[0]-camera[0],y=a.xyz[1]-camera[1],z=a.xyz[2]-camera[2];
    const float length=std::max(1e-12f,std::sqrt(x*x+y*y+z*z));x/=length;y/=length;z/=length;
    const float xx=x*x,yy=y*y,zz=z*z,xy=x*y,yz=y*z,xz=x*z;
    constexpr float c0=.28209479177387814f,c1=.4886025119029199f;
    const float b[15]={-c1*y,c1*z,-c1*x,1.0925484305920792f*xy,-1.0925484305920792f*yz,
        .31539156525252005f*(2*zz-xx-yy),-1.0925484305920792f*xz,.5462742152960396f*(xx-yy),
        -.5900435899266435f*y*(3*xx-yy),2.890611442640554f*xy*z,-.4570457994644658f*y*(4*zz-xx-yy),
        .3731763325901154f*z*(2*zz-3*xx-3*yy),-.4570457994644658f*x*(4*zz-xx-yy),
        1.445305721320277f*z*(xx-yy),-.5900435899266435f*x*(xx-3*yy)};
    std::array<float,3> out{};
    for(uint32_t c=0;c<3;++c){float v=c0*a.dc[c];for(uint32_t j=0;j<15;++j)v+=b[j]*rest[j*3+c];out[c]=std::max(0.f,v+.5f);}
    return out;
}
}

bool AnchorVisible(const float* center,float scale,const FrameCamera& camera) {
    const auto& m=camera.projection;
    float planes[6][4];
    for(uint32_t c=0;c<4;++c){planes[0][c]=m[12+c]+m[c];planes[1][c]=m[12+c]-m[c];
        planes[2][c]=m[12+c]+m[4+c];planes[3][c]=m[12+c]-m[4+c];planes[4][c]=m[8+c];planes[5][c]=m[12+c]-m[8+c];}
    for(const auto& p:planes){const float d=((center[0]*p[0]+center[1]*p[1])+center[2]*p[2])+p[3];
        if(d+scale*.5f*(std::abs(p[0])+std::abs(p[1])+std::abs(p[2]))<0)return false;}
    return true;
}

std::array<float,7> AnchorViewCode(const float* center,float scale,const FrameCamera& camera,float sceneScale) {
    std::array<float,7> out{};
    float length=0;for(uint32_t c=0;c<3;++c){out[c]=camera.center[c]-center[c];length+=out[c]*out[c];}
    length=std::max(std::sqrt(length),1e-6f);for(uint32_t c=0;c<3;++c)out[c]/=length;
    const auto cam=Transform(camera.view,center),clip=Transform(camera.projection,center);
    out[3]=std::asinh(cam[2]/sceneScale);
    out[4]=Clamp(clip[0]/Safe(clip[3],camera.nearPlane),-4,4);out[5]=Clamp(clip[1]/Safe(clip[3],camera.nearPlane),-4,4);
    float minX=INFINITY,maxX=-INFINITY,minY=INFINITY,maxY=-INFINITY,minZ=INFINITY,maxZ=-INFINITY;
    for(uint32_t j=0;j<8;++j){float p[3];for(uint32_t c=0;c<3;++c)p[c]=center[c]+scale*.5f*((j&(1u<<c))?1.f:-1.f);
        const auto q=Transform(camera.projection,p),v=Transform(camera.view,p);const float w=Safe(q[3],camera.nearPlane);
        minX=std::min(minX,q[0]/w);maxX=std::max(maxX,q[0]/w);minY=std::min(minY,q[1]/w);maxY=std::max(maxY,q[1]/w);
        minZ=std::min(minZ,v[2]);maxZ=std::max(maxZ,v[2]);}
    float extent=Clamp(std::max((maxX-minX)*float(camera.width)/camera.height*.5f,(maxY-minY)*.5f),0,1e4f);
    if(minZ<=camera.nearPlane)extent=1e4f;if(maxZ<=camera.nearPlane)extent=0;out[6]=std::log1p(extent);
    return out;
}

void GaussianFrame::Bind(const SceneAsset& scene) {
    scene_=&scene;const uint32_t n=scene.Identity().gaussianCount;
    master_.resize(n);
    const float *xyz=F(scene,"xyz"),*scale=F(scene,"log_scale"),*q=F(scene,"rotation"),*dc=F(scene,"dc"),
                *opacity=F(scene,"opacity_logits"),*reference=F(scene,"anchor_scales");
    const int32_t* owners=reinterpret_cast<const int32_t*>(scene.Section("owners").data);
    for(uint32_t i=0;i<n;++i){auto& m=master_[i];for(uint32_t c=0;c<3;++c){m.xyz[c]=xyz[i*3+c];m.scale[c]=std::exp(scale[i*3+c]);m.dc[c]=dc[i*3+c];}
        std::copy_n(q+i*4,4,m.rotation.data());m.opacity_logit=opacity[i];m.reference=reference[owners[i]];}
    candidates_.reserve(n);selected_.reserve(n);decoded_.reserve(n);views_.resize(scene.Identity().anchorCount*7);
    benchmarkOrder_.clear();
}

void GaussianFrame::PreparePointBenchmark(uint32_t k) {
    if(!scene_ || k>master_.size())throw std::runtime_error("Invalid point benchmark budget");
    const auto start=Clock::now();
    if(benchmarkOrder_.empty()) {
        const auto* uid=reinterpret_cast<const uint64_t*>(scene_->Section("uids").data);
        benchmarkOrder_.resize(master_.size());std::iota(benchmarkOrder_.begin(),benchmarkOrder_.end(),0u);
        std::sort(benchmarkOrder_.begin(),benchmarkOrder_.end(),[&](uint32_t a,uint32_t b){
            const auto x=BenchmarkUidHash(uid[a]),y=BenchmarkUidHash(uid[b]);return x==y?uid[a]<uid[b]:x<y;});
    }
    selected_.assign(benchmarkOrder_.begin(),benchmarkOrder_.begin()+k);
    std::sort(selected_.begin(),selected_.end());decoded_.resize(k);
    for(uint32_t i=0;i<k;++i){const auto& m=master_[selected_[i]];
        decoded_[i]={m.xyz,m.scale,m.dc,m.rotation,1.f/(1.f+std::exp(-m.opacity_logit))};}
    stats_={};stats_.candidates=stats_.eligible=uint32_t(master_.size());stats_.k=k;stats_.prepareMs=Millis(start);
}

void GaussianFrame::Prepare(const FrameCamera& camera,InferenceBackend& backend,float ratio,uint32_t fixedK,bool full) {
    if(!scene_)throw std::runtime_error("GaussianFrame has no bound scene");
    const auto& scene=*scene_;const uint32_t n=scene.Identity().gaussianCount,a=scene.Identity().anchorCount;
    stats_={};selected_.clear();decoded_.clear();const auto start=Clock::now();
    if(full) {
        selected_.resize(n);std::iota(selected_.begin(),selected_.end(),0u);decoded_.resize(n);
        for(uint32_t i=0;i<n;++i){const auto& m=master_[i];decoded_[i]={m.xyz,m.scale,m.dc,m.rotation,1.f/(1.f+std::exp(-m.opacity_logit))};}
        stats_.candidates=stats_.eligible=stats_.k=n;stats_.prepareMs=Millis(start);return;
    }
    if(!backend.Ready())throw std::runtime_error(backend.Status().message);
    const int32_t* owner=reinterpret_cast<const int32_t*>(scene.Section("owners").data);
    const auto* eligibility=scene.Section("eligible").data;const auto* uid=reinterpret_cast<const uint64_t*>(scene.Section("uids").data);
    const float *center=F(scene,"anchor_centers"),*scale=F(scene,"anchor_scales"),*static10=F(scene,"selector_static10"),
                *noise=F(scene,"noise2"),*offset=F(scene,"offset3");
    std::vector<uint8_t> visible(a);
    for(uint32_t anchor=0;anchor<a;++anchor){visible[anchor]=AnchorVisible(center+anchor*3,scale[anchor],camera);
        if(visible[anchor]){const auto code=AnchorViewCode(center+anchor*3,scale[anchor],camera,scene.Identity().viewSceneScale);
            std::copy_n(code.data(),7,views_.data()+anchor*7);}}
    candidates_.clear();for(uint32_t i=0;i<n;++i)if(visible[owner[i]])candidates_.push_back(i);
    const uint32_t count=uint32_t(candidates_.size());stats_.candidates=count;
    if(count==0){stats_.prepareMs=Millis(start);return;}
    if(fixedK)ratio=float(std::min(1.,double(fixedK)/count));
    selector_.resize(count*19);noise_.resize(count*2);uids_.resize(count);eligible_.resize(count);probability_.resize(count*2);scores_.resize(count);
    uint32_t available=0;
    for(uint32_t i=0;i<count;++i){const uint32_t row=candidates_[i];float* x=selector_.data()+i*19;
        std::copy_n(static10+row*10,10,x);x[10]=std::log(ratio);
        for(uint32_t j=0;j<7;++j)x[11+j]=(scene.Identity().viewSchema==2||scene.Identity().viewSchema==3)?views_[owner[row]*7+j]:0.f;
        x[18]=1;std::copy_n(noise+row*2,2,noise_.data()+i*2);uids_[i]=uid[row];available+=eligible_[i]=eligibility[row];}
    stats_.eligible=available;stats_.prepareMs=Millis(start);
    auto timer=Clock::now();backend.RunSelector(count,selector_.data(),noise_.data(),probability_.data());stats_.selectorMs=Millis(timer);
    timer=Clock::now();for(uint32_t i=0;i<count;++i)scores_[i]=probability_[i*2+1];
    const uint32_t k=std::min(available,fixedK?fixedK:uint32_t(std::floor(double(count)*ratio)));
    const auto compact=StrictTopK(scores_,uids_,eligible_,k);for(uint32_t row:compact)selected_.push_back(candidates_[row]);
    stats_.topkMs=Millis(timer);stats_.k=uint32_t(selected_.size());
    if(stats_.k!=k)throw std::runtime_error("Strict K violated in Android selection");
    timer=Clock::now();student_.resize(k*62);residual_.resize(k*14);
    const auto weights=Residual48Weights(ratio,scene.Identity().ratioKnots);
    const auto code=ContinuousRatioCode(ratio,scene.Identity().ratioBounds[0],scene.Identity().ratioBounds[1]);
    const float* tables[6];for(uint32_t level=0;level<6;++level)tables[level]=F(scene,("residual48_"+std::to_string(level)).c_str());
    for(uint32_t i=0;i<k;++i){const uint32_t row=selected_[i],anchor=uint32_t(owner[row]);float* x=student_.data()+i*62;
        std::copy_n(offset+row*3,3,x);GatherResidual48(tables,anchor,weights,x+3);std::copy_n(code.data(),3,x+51);
        for(uint32_t j=0;j<7;++j)x[54+j]=(scene.Identity().viewSchema==2||scene.Identity().viewSchema==4)?views_[anchor*7+j]:0.f;x[61]=1;}
    stats_.prepareMs+=Millis(timer);timer=Clock::now();backend.RunStudent(k,student_.data(),residual_.data());stats_.studentMs=Millis(timer);
    timer=Clock::now();decoded_.resize(k);
    for(uint32_t i=0;i<k;++i){std::array<float,14> residual;std::copy_n(residual_.data()+i*14,14,residual.data());const auto& m=master_[selected_[i]];
        float xyz=0;for(uint32_t j=0;j<3;++j){residual[j]*=m.reference;xyz+=residual[j]*residual[j];}
        decoded_[i]=DecodeResidual(m,residual,scene.Identity().softplusScale);stats_.maxXyz=std::max(stats_.maxXyz,std::sqrt(xyz));
        for(uint32_t j=0;j<3;++j)stats_.maxScaleRatio=std::max(stats_.maxScaleRatio,decoded_[i].scale[j]/m.scale[j]);}
    stats_.decodeMs=Millis(timer);
}

void GaussianFrame::Project(const FrameCamera& camera,GaussianEye& out) const {
    const uint32_t tx=(camera.width+15)/16,ty=(camera.height+15)/16,tiles=tx*ty;
    out.width=camera.width;out.height=camera.height;out.splats.clear();out.members.clear();out.ranges.assign(tiles,{});out.projected=0;
    struct Bounds{uint32_t x0,y0,x1,y1;float depth;};std::vector<Bounds> bounds;
    const float* rest=F(*scene_,"sh_rest");
    for(uint32_t i=0;i<selected_.size();++i){const auto& a=decoded_[i];const auto p=Transform(camera.view,a.xyz.data());
        if(p[2]<=.2f)continue;
        const float px=camera.fx*p[0]/p[2]+camera.cx,py=camera.fy*p[1]/p[2]+camera.cy;
        const auto& q=a.rotation;const float w=q[0],x=q[1],y=q[2],z=q[3];
        const float rotation[9]={1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)};
        float covariance[9]={};
        for(uint32_t r=0;r<3;++r)for(uint32_t c=0;c<3;++c)for(uint32_t j=0;j<3;++j)
            covariance[r*3+c]+=rotation[r*3+j]*rotation[c*3+j]*a.scale[j]*a.scale[j];
        const float limX=1.3f*camera.width/(2*camera.fx),limY=1.3f*camera.height/(2*camera.fy);
        const float cx=Clamp(p[0]/p[2],-limX,limX)*p[2],cy=Clamp(p[1]/p[2],-limY,limY)*p[2];
        const float j[6]={camera.fx/p[2],0,-camera.fx*cx/(p[2]*p[2]),0,camera.fy/p[2],-camera.fy*cy/(p[2]*p[2])};
        float t[6]={};for(uint32_t r=0;r<2;++r)for(uint32_t c=0;c<3;++c)for(uint32_t v=0;v<3;++v)t[r*3+c]+=j[r*3+v]*camera.view[v*4+c];
        float c00=.3f,c01=0,c11=.3f;
        for(uint32_t r=0;r<3;++r)for(uint32_t c=0;c<3;++c){c00+=t[r]*covariance[r*3+c]*t[c];c01+=t[r]*covariance[r*3+c]*t[3+c];c11+=t[3+r]*covariance[r*3+c]*t[3+c];}
        const float det=c00*c11-c01*c01;if(!(det>0))continue;
        const float mid=.5f*(c00+c11),radius=std::ceil(3.f*std::sqrt(mid+std::sqrt(std::max(.1f,mid*mid-det))));
        if(!std::isfinite(radius)||radius>1e7f)throw std::runtime_error("Nonfinite/extreme projected Gaussian footprint");
        Bounds b{uint32_t(Clamp(float(int((px-radius)/16)),0,float(tx))),uint32_t(Clamp(float(int((py-radius)/16)),0,float(ty))),
                 uint32_t(Clamp(float(int((px+radius+15)/16)),0,float(tx))),uint32_t(Clamp(float(int((py+radius+15)/16)),0,float(ty))),p[2]};
        if(b.x0>=b.x1||b.y0>=b.y1)continue;
        const auto color=Color(a,rest+selected_[i]*45,camera.center);
        out.splats.push_back({{px,py,p[2],radius},{c11/det,-c01/det,c00/det,a.opacity},{color[0],color[1],color[2],0}});bounds.push_back(b);
        for(uint32_t y0=b.y0;y0<b.y1;++y0)for(uint32_t x0=b.x0;x0<b.x1;++x0)++out.ranges[y0*tx+x0].end;
    }
    uint64_t total=0;for(auto& r:out.ranges){const uint32_t size=r.end;r.begin=uint32_t(total);total+=size;r.end=uint32_t(total);}
    if(total>32ull*1024*1024)throw std::runtime_error("Gaussian tile instance workspace exceeds 128MiB cap");
    out.members.resize(size_t(total));std::vector<uint32_t> cursor(tiles);for(uint32_t i=0;i<tiles;++i)cursor[i]=out.ranges[i].begin;
    std::vector<uint32_t> order(bounds.size());std::iota(order.begin(),order.end(),0u);
    std::stable_sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b){return bounds[a].depth<bounds[b].depth;});
    for(uint32_t index:order){const auto& b=bounds[index];for(uint32_t y=b.y0;y<b.y1;++y)for(uint32_t x=b.x0;x<b.x1;++x)out.members[cursor[y*tx+x]++]=index;}
    out.projected=uint32_t(out.splats.size());
}
}  // namespace mycartgs
