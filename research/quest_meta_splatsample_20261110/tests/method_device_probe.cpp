#include "method_engine.h"
#include "mycartgs/strict_topk.h"
#include <fstream>
#include <iostream>
#include <cmath>

namespace {
mycartgs::FrameCamera ReferenceCamera(){
    mycartgs::FrameCamera c;c.width=1200;c.height=680;c.fx=c.fy=600;c.cx=599.5;c.cy=339.5;
    c.center={-.8294860125f,-.8426836729f,1.7652033567f};
    const float r[9]={-.7053378820f,.4040099382f,-.5824726224f,-.1863143742f,.6871432066f,.7022258043f,.6839483976f,.6038295031f,-.4093953669f};
    for(int i=0;i<3;++i){for(int j=0;j<3;++j)c.view[i*4+j]=r[j*3+i];for(int j=0;j<3;++j)c.view[i*4+3]-=c.view[i*4+j]*c.center[j];}c.view[15]=1;
    std::array<float,16> p{};p[0]=2*c.fx/c.width;p[5]=2*c.fy/c.height;p[10]=c.farPlane/(c.farPlane-c.nearPlane);p[11]=-c.farPlane*c.nearPlane/(c.farPlane-c.nearPlane);p[14]=1;
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int k=0;k<4;++k)c.projection[i*4+j]+=p[i*4+k]*c.view[k*4+j];return c;
}
void Require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
}
int main(int argc,char** argv){
    try{
        if(argc!=3)throw std::runtime_error("usage: method_device_probe scene.mcgs output-directory");
        mycartgs::SceneDescriptor d;d.slug="Replica__office2__rgbd";d.gaussianCount=43197;d.anchorCount=16723;
        d.assetSha256="1d735b0871ff94cc6acbf039ad014e0d1490ca4e209caef92648a0b9e53a6685";
        d.teacherFingerprint="9e2d413fa7864127465394fa22bb3da4e05d912d4c9e0822a4e928933aa36804";
        d.studentCheckpointSha256="fd85285c6bd5871f4b41c85690597ccadc965d5a5ec6f474a983e4e09f547a75";
        mycartgs::MethodEngine engine;engine.LoadFile(argv[1],d);const auto camera=ReferenceCamera();
        Json::Value results(Json::arrayValue);const std::string folder=argv[2];
        for(bool global:{false,true})for(float ratio:{.01f,.1f,.137f,.3f,.5f,1.f}){
            const std::string prefix=folder+"/"+(global?"global":"local")+std::to_string(int(ratio*1000));
            auto result=engine.Run(camera,global,ratio,-1,prefix+".ply",1,prefix);
            const auto expected=std::min(result["eligible"].asUInt(),uint32_t(std::floor(double(result["candidates"].asUInt())*ratio)));
            Require(result["k"].asUInt()==expected,"Ratio strict K mismatch");
            if(global)Require(result["candidates"].asUInt()==43197,"Global domain not full map");
            auto repeat=engine.Run(camera,global,ratio,-1,"");Require(repeat["selected_uid_sha256"]==result["selected_uid_sha256"],"Repeated UID set changed");
            result["case"]=prefix;results.append(result);std::cout<<"RATIO_PASS "<<prefix<<" K="<<expected<<std::endl;
        }
        for(bool global:{false,true})for(int fixed:{0,1,1000,5000,100000}){
            const std::string prefix=folder+"/"+(global?"global_k":"local_k")+std::to_string(fixed);
            auto result=engine.Run(camera,global,.3f,fixed,prefix+".ply",1,prefix);
            Require(result["k"].asUInt()==std::min(uint32_t(fixed),result["eligible"].asUInt()),"Fixed strict K mismatch");
            result["case"]=prefix;results.append(result);
        }
        auto empty=camera;empty.projection={};empty.projection[15]=-1;
        auto zero=engine.Run(empty,false,.3f,-1,folder+"/empty.ply");Require(zero["k"].asUInt()==0&&zero["candidates"].asUInt()==0,"Empty frustum failed");results.append(zero);
        float head[]={0,0,0,1,0,0,0};float views[]={1200,680,-.032f,0,0,1,0,0,0,-.7f,.7f,.6f,-.6f,.032f,0,0,1,0,0,0,-.7f,.7f,.6f,-.6f};
        auto stereo=mycartgs::MethodCamera(head,views);Require(stereo.cullingProjections.size()==2,"Stereo union missing");
        Require(stereo.view[0]==1&&stereo.view[5]==-1&&stereo.view[10]==-1,"OpenXR to CV axes wrong");
        for(int i=0;i<2;++i){const float x=views[2+i*11];const float p[]={x,0,-2};const auto& m=stereo.cullingProjections[i];
            float clipx=m[0]*p[0]+m[2]*p[2]+m[3],clipw=m[12]*p[0]+m[14]*p[2]+m[15];Require(std::abs(clipx/clipw)<1e-6f,"Eye center projection wrong");}
        auto unionCamera=camera;unionCamera.cullingProjections={camera.projection,empty.projection};
        auto unionCase=engine.Run(unionCamera,false,.3f,-1,"");auto single=engine.Run(camera,false,.3f,-1,"");
        Require(unionCase["selected_uid_sha256"]==single["selected_uid_sha256"],"Stereo union changed identical candidate domain");
        auto tied=mycartgs::StrictTopK({.5f,.5f,.5f},{3,1,2},{1,1,1},2);Require(tied==std::vector<uint32_t>({1,2}),"UID tie policy failed");
        auto bench=engine.Run(camera,true,.3f,1000,folder+"/benchmark.ply",11);Require(bench["cpu_measurements"].size()==9,"Benchmark warmup accounting failed");results.append(bench);
        Json::StreamWriterBuilder builder;builder["indentation"]="  ";std::ofstream(folder+"/probe.json")<<Json::writeString(builder,results);
        std::cout<<"METHOD_PROBE_PASS ratios, fixedK, zero, saturation, repeat, stereo, camera, UID ties, Student32 binding"<<std::endl;
    }catch(const std::exception& e){std::cerr<<"METHOD_PROBE_FAIL "<<e.what()<<std::endl;return 1;}
}
