#include "xnnpack_backend.h"
#include "scene_asset.h"
#include "residual48.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using Clock=std::chrono::steady_clock;
double Ms(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
void Write(const std::string& path,const std::vector<float>& data) {
    std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(float));
    if(!out)throw std::runtime_error("Probe write failed: "+path);
}
int main(int argc,char** argv) {
    try {
        if(argc!=4)throw std::runtime_error("usage: xnnpack_device_probe scene.mcgs output-prefix threads");
        mycartgs::SceneDescriptor identity;
        identity.slug="Replica__office2__rgbd";identity.gaussianCount=43197;identity.anchorCount=16723;
        identity.assetSha256="1d735b0871ff94cc6acbf039ad014e0d1490ca4e209caef92648a0b9e53a6685";
        identity.teacherFingerprint="9e2d413fa7864127465394fa22bb3da4e05d912d4c9e0822a4e928933aa36804";
        identity.studentCheckpointSha256="fd85285c6bd5871f4b41c85690597ccadc965d5a5ec6f474a983e4e09f547a75";
        mycartgs::SceneAsset scene;scene.LoadFile(argv[1],identity);
        const auto start=Clock::now();mycartgs::XnnpackBackend backend;backend.Initialize(uint32_t(std::stoul(argv[3])));backend.Bind(scene);
        std::cout<<"LOAD_PASS ms="<<Ms(start)<<" workspace="<<backend.Status().workspaceBytes<<std::endl;
        const auto f=[&](const char* name){return reinterpret_cast<const float*>(scene.Section(name).data);};
        const float *physical=f("selector_static10"),*fixedNoise=f("noise2"),*offset=f("offset3");
        const auto* owner=reinterpret_cast<const int32_t*>(scene.Section("owners").data);
        const float* tables[6];for(uint32_t j=0;j<6;++j)tables[j]=f(("residual48_"+std::to_string(j)).c_str());
        const auto weights=mycartgs::Residual48Weights(.3f,scene.Identity().ratioKnots);
        const auto code=mycartgs::ContinuousRatioCode(.3f,scene.Identity().ratioBounds[0],scene.Identity().ratioBounds[1]);
        std::ofstream csv(std::string(argv[2])+"_timings.csv");csv<<"rows,selector_median_ms,student_median_ms,threads\n";
        for(uint32_t rows:{0u,1u,17u,1023u,1024u,1027u,4096u,4097u,43197u}) {
            std::vector<float> selector(rows*19),noise(rows*2),student(rows*62),score(rows*2),residual(rows*14);
            for(uint32_t i=0;i<rows;++i) {
                std::copy_n(physical+i*10,10,selector.data()+i*19);selector[i*19+10]=std::log(.3f);selector[i*19+18]=1;
                std::copy_n(fixedNoise+i*2,2,noise.data()+i*2);
                std::copy_n(offset+i*3,3,student.data()+i*62);
                mycartgs::GatherResidual48(tables,uint32_t(owner[i]),weights,student.data()+i*62+3);
                std::copy_n(code.data(),3,student.data()+i*62+51);student[i*62+61]=1;
            }
            std::vector<double> selTime,studentTime;
            for(uint32_t repeat=0;repeat<11;++repeat) {
                auto timer=Clock::now();backend.RunSelector(rows,selector.data(),noise.data(),score.data());double s=Ms(timer);
                timer=Clock::now();backend.RunStudent(rows,student.data(),residual.data());double c=Ms(timer);
                if(repeat>=2){selTime.push_back(s);studentTime.push_back(c);}
            }
            std::sort(selTime.begin(),selTime.end());std::sort(studentTime.begin(),studentTime.end());
            csv<<rows<<','<<selTime[4]<<','<<studentTime[4]<<','<<backend.Status().threads<<'\n';
            if(rows==1027 || rows==4097 || rows==43197) {
                const std::string prefix=std::string(argv[2])+"_"+std::to_string(rows);
                Write(prefix+"_selector_input.bin",selector);Write(prefix+"_noise.bin",noise);Write(prefix+"_student_input.bin",student);
                Write(prefix+"_selector_output.bin",score);Write(prefix+"_student_output.bin",residual);
            }
            std::cout<<"CASE_PASS rows="<<rows<<" selector_ms="<<selTime[4]<<" student_ms="<<studentTime[4]<<std::endl;
        }
        if(backend.Status().selectorCalls!=88 || backend.Status().studentCalls!=88)throw std::runtime_error("Empty execution counter semantics changed");
        backend.Bind(scene);backend.RunSelector(0,nullptr,nullptr,nullptr);backend.RunStudent(0,nullptr,nullptr);
        if(backend.Status().execution!="NOT_EXECUTED")throw std::runtime_error("Rebind or empty execution state incorrect");
        std::cout<<"PROBE_PASS ARM64 XNNPACK FP32, tails, dynamic rows, empty and rebind"<<std::endl;
    }catch(const std::exception& error){std::cerr<<"PROBE_FAIL "<<error.what()<<std::endl;return 1;}
}
