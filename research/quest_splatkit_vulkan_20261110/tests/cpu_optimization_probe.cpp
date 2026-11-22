#include "gaussian_frame.h"
#include "xnnpack_backend.h"
#include "gaussian_frame_reference.h"
#include "xnnpack_backend_reference.h"
#include "cpu_recorded_cameras.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace mycartgs {
struct CpuProbe {
    static void Compare(const GaussianFrame& a,const ReferenceGaussianFrame& b) {
        auto same=[](const auto& x,const auto& y,const char* name){
            if(x.size()!=y.size()||(!x.empty()&&std::memcmp(x.data(),y.data(),x.size()*sizeof(x[0]))))
                throw std::runtime_error(std::string("BITWISE_MISMATCH ")+name);
        };
        same(a.selected_,b.selected_,"selected_rows");same(a.decoded_,b.decoded_,"decoded_attributes");
        if(a.stats_.candidates!=b.stats_.candidates||a.stats_.eligible!=b.stats_.eligible||a.stats_.k!=b.stats_.k)
            throw std::runtime_error("COUNTS_MISMATCH");
        if(a.stats_.candidates){same(a.candidates_,b.candidates_,"candidate_rows");same(a.probability_,b.probability_,"selector_probabilities");
            same(a.scores_,b.scores_,"selector_scores");same(a.residual_,b.residual_,"student_residuals");}
        if(std::memcmp(&a.stats_.maxXyz,&b.stats_.maxXyz,sizeof(float))||std::memcmp(&a.stats_.maxScaleRatio,&b.stats_.maxScaleRatio,sizeof(float)))
            throw std::runtime_error("DIAGNOSTIC_MISMATCH");
    }
};
}
namespace {
using namespace mycartgs;
using Clock=std::chrono::steady_clock;
void Require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
FrameCamera Camera(unsigned index) {
    auto c=RecordedCamera(index%2);if(index<2)return c;
    if(index>=8){const float angle=std::sin(float(index)*1.7f)*.35f,co=std::cos(angle),si=std::sin(angle);
        for(auto* m:{&c.view,&c.projection}){const auto old=*m;for(unsigned r=0;r<4;++r){
            (*m)[r*4]=old[r*4]*co-old[r*4+2]*si;(*m)[r*4+2]=old[r*4]*si+old[r*4+2]*co;}}
        const auto old=c.center;c.center[0]=old[0]*co-old[2]*si;c.center[2]=old[0]*si+old[2]*co;
    }
    // Fresh poses for continuous and jumping camera tests; no prior-frame masks reused.
    const float dx=index==7?100000.f:std::sin(float(index)*1.1f)*.3f;
    const float dy=index==7?100000.f:std::cos(float(index)*.7f)*.2f;
    const float dz=index==7?100000.f:std::sin(float(index)*.3f)*.2f;
    const float delta[]={dx,dy,dz};
    for(unsigned j=0;j<3;++j)c.center[j]+=delta[j];
    for(unsigned r=0;r<4;++r)for(unsigned j=0;j<3;++j){c.view[r*4+3]-=c.view[r*4+j]*delta[j];c.projection[r*4+3]-=c.projection[r*4+j]*delta[j];}
    return c;
}
SceneDescriptor Descriptor(){SceneDescriptor d;d.slug="Replica__office2__rgbd";d.gaussianCount=43197;d.anchorCount=16723;
    d.assetSha256="1d735b0871ff94cc6acbf039ad014e0d1490ca4e209caef92648a0b9e53a6685";
    d.teacherFingerprint="9e2d413fa7864127465394fa22bb3da4e05d912d4c9e0822a4e928933aa36804";
    d.studentCheckpointSha256="fd85285c6bd5871f4b41c85690597ccadc965d5a5ec6f474a983e4e09f547a75";return d;}
void NetworkChecks(const SceneAsset& scene,XnnpackBackend& backend) {
    ReferenceXnnpackBackend reference;reference.Initialize(2);reference.Bind(scene);
    for(uint32_t rows:{0u,1u,7u,31u,32u,63u,255u,4095u,4096u,4097u,8192u,8193u,12243u,16383u,16384u,16385u,43197u}){
        for(unsigned network=0;network<2;++network){const uint32_t width=network?62:19,outputWidth=network?14:2;
            std::vector<float> input(size_t(rows)*width),noise(size_t(rows)*2),a(size_t(rows)*outputWidth),b(a.size());
            for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i)*.001f)*2;
            for(size_t i=0;i<noise.size();++i)noise[i]=std::cos(float(i)*.01f);
            if(network)reference.RunStudent(rows,input.data(),a.data());else reference.RunSelector(rows,input.data(),noise.data(),a.data());
            for(uint32_t flags:{0u,uint32_t(CpuDirectInput),uint32_t(CpuStableBatchPlans),uint32_t(CpuLargeBatch|CpuDirectInput),uint32_t(CpuOptimizationsAll)}){
                backend.SetOptimizationFlags(flags);float* destination=network?backend.StudentInput(rows):backend.SelectorInput(rows);
                if(destination)std::copy(input.begin(),input.end(),destination);else destination=input.data();
                if(network)backend.RunStudent(rows,destination,b.data());else backend.RunSelector(rows,destination,noise.data(),b.data());
                Require(a.empty()||!std::memcmp(a.data(),b.data(),a.size()*sizeof(float)),"NETWORK_BITWISE_MISMATCH");
            }
        }
    }
}
void Checks(const SceneAsset& scene,XnnpackBackend& backend,GaussianFrame& frame,ReferenceGaussianFrame& reference) {
    NetworkChecks(scene,backend);uint32_t cases=0;
    for(uint32_t flags:{0u,1u,2u,4u,8u,16u,32u,64u,93u,127u,221u,255u})for(unsigned camera=0;camera<16;++camera)
        for(float ratio:{.01f,.05f,.1f,.15f,.2f,.3f,.4f,.45f,.5f,.7f,1.f}){
            frame.SetOptimizationFlags(flags);backend.SetOptimizationFlags(flags);const auto c=Camera(camera);
            // Same pool and same legacy FC operations isolate preparation, layout and scheduling.
            backend.SetOptimizationFlags(0);reference.Prepare(c,backend,ratio,0,false);backend.SetOptimizationFlags(flags);
            frame.Prepare(c,backend,ratio,0,false);
            try{CpuProbe::Compare(frame,reference);}catch(...){std::cerr<<"flags="<<flags<<" camera="<<camera<<" ratio="<<ratio<<std::endl;throw;}++cases;
        }
    for(unsigned camera=0;camera<16;++camera)for(uint32_t k:{1u,31u,32u,255u,1000u,2000u,4095u,4096u,4097u,5000u,10000u,100000u}){
        const auto c=Camera(camera);backend.SetOptimizationFlags(0);reference.Prepare(c,backend,.3f,k,false);
        frame.SetOptimizationFlags(CpuOptimizationsAll);backend.SetOptimizationFlags(CpuOptimizationsAll);frame.Prepare(c,backend,.3f,k,false);CpuProbe::Compare(frame,reference);++cases;
    }
    const auto c=Camera(0);reference.Prepare(c,backend,.3f,0,true);frame.Prepare(c,backend,.3f,0,true);CpuProbe::Compare(frame,reference);++cases;
    bool rejected=false;try{CpuParallelFor(backend,4096,128,[](size_t begin,size_t){if(begin==0)throw std::runtime_error("worker finite sentinel");});}
    catch(const std::runtime_error& e){rejected=std::string(e.what())=="worker finite sentinel";}Require(rejected,"Worker exception was lost");
    for(unsigned pattern=0;pattern<5;++pattern){std::vector<float> scores(10000);std::vector<uint64_t> uids(10000);std::vector<uint8_t> eligible(10000);
        for(uint32_t i=0;i<scores.size();++i){scores[i]=pattern==0?1.f:pattern==1?.5f:pattern==2?float(i%100)/100:pattern==3?std::sin(float(i)):(i%2?-1.f:2.f);uids[i]=(i*1777)%5000;eligible[i]=(i%7)!=0;}
        for(uint32_t k:{0u,1u,255u,1000u,5000u,10000u}){std::vector<ScoreKey> keys;std::vector<uint32_t> rows;
            StrictTopKHistogram(scores,uids,eligible,k,keys,rows);Require(rows==StrictTopK(scores,uids,eligible,k),"HISTOGRAM_TIES_MISMATCH");}
    }
    std::vector<DecodedAttribute> scalar(128),parallel(128);std::vector<std::string> scalarError(128),parallelError(128);
    const auto decodeExtreme=[&](unsigned i){MasterAttribute m{};m.scale={1e-30f,.01f,100.f};m.rotation={1,0,0,0};m.reference=.05f;
        std::array<float,14> d{};const float values[]={-1000,-100,-30,-20,-1,0,20,30,100,1000};d[3]=values[i%10];d[4]=values[(i+1)%10];d[5]=values[(i+2)%10];d[10]=values[(i+3)%10];d[6]=values[(i+4)%10];return DecodeResidual(m,d,true);};
    for(unsigned i=0;i<scalar.size();++i)try{scalar[i]=decodeExtreme(i);}catch(const std::exception& e){scalarError[i]=e.what();}
    CpuParallelFor(backend,parallel.size(),8,[&](size_t begin,size_t size){for(size_t i=begin;i<begin+size;++i)
        try{parallel[i]=decodeExtreme(unsigned(i));}catch(const std::exception& e){parallelError[i]=e.what();}});
    for(unsigned i=0;i<scalar.size();++i){Require(scalarError[i]==parallelError[i],"EXTREME_DECODE_ERROR_MISMATCH");
        if(scalarError[i].empty())Require(!std::memcmp(&scalar[i],&parallel[i],sizeof(DecodedAttribute)),"EXTREME_DECODE_BITWISE_MISMATCH");}
    frame.Prepare(c,backend,.3f,0,false);std::cout<<"CPU_CHECK_PASS bitwise_frames="<<cases<<" network_shapes=17 flags=12 histogram_ties_and_fallback30 extreme_decode128 empty_full_fixed_dynamic worker_exception_recovery"<<std::endl;
}
}
int main(int argc,char** argv){try{
    using namespace mycartgs;if(argc<3)throw std::runtime_error("usage: cpu_optimization_probe scene.mcgs check | bench output.csv flags threads rounds samples");
    SceneAsset scene;scene.LoadFile(argv[1],Descriptor());const unsigned threads=std::string(argv[2])=="check"?(argc>3?std::stoul(argv[3]):2):(argc>5?std::stoul(argv[5]):2);
    XnnpackBackend backend;backend.Initialize(threads);backend.Bind(scene);GaussianFrame frame;frame.Bind(scene);ReferenceGaussianFrame reference;reference.Bind(scene);
    if(std::string(argv[2])=="check"){Checks(scene,backend,frame,reference);return 0;}
    Require(argc==8,"Invalid bench arguments");const uint32_t flags=std::stoul(argv[4]),rounds=std::stoul(argv[6]),samples=std::stoul(argv[7]);
    const std::string mode=argv[2];const bool moving=mode.find("motion")!=std::string::npos;
    const unsigned firstCamera=mode.find("holdout")!=std::string::npos?8:0;
    std::unique_ptr<GaussianFrame> control;uint32_t controlFlags=0;
    if(mode.rfind("bench_control_",0)==0){controlFlags=std::stoul(mode.substr(14));control=std::make_unique<GaussianFrame>();control->Bind(scene);control->SetOptimizationFlags(controlFlags);}
    std::ofstream csv(argv[3]);Require(bool(csv),"Cannot write benchmark CSV");csv.precision(10);
    csv<<"round,camera,ratio,fixed_k,variant,flags,threads,sample,N,eligible,K,prepare_ms,selector_ms,topk_ms,student_ms,decode_ms,cpu_total_ms\n";
    for(uint32_t round=0;round<rounds;++round)for(unsigned path=0;path<6;++path)for(unsigned condition=0;condition<9;++condition){
        const unsigned camera=firstCamera+(round%2?5-path:path);
        const float ratios[]={.1f,.2f,.3f,.4f,.5f,.1f,.1f,.1f,.1f};const uint32_t budgets[]={0,0,0,0,0,1000,2000,5000,10000};
        const float ratio=ratios[condition];const uint32_t k=budgets[condition];const auto c=Camera(camera);
        backend.SetOptimizationFlags(0);reference.Prepare(c,backend,ratio,k,false);backend.SetOptimizationFlags(flags);frame.SetOptimizationFlags(flags);frame.Prepare(c,backend,ratio,k,false);CpuProbe::Compare(frame,reference);
        for(unsigned block=0;block<4;++block){const bool optimized=((block==0||block==3)^(round%2==0));backend.SetThreadCount(optimized||control?threads:2);backend.SetOptimizationFlags(optimized?flags:controlFlags);
            const auto prepare=[&](const FrameCamera& camera){if(optimized)frame.Prepare(camera,backend,ratio,k,false);else if(control)control->Prepare(camera,backend,ratio,k,false);else reference.Prepare(camera,backend,ratio,k,false);};
            for(unsigned warm=0;warm<3;++warm)prepare(c);
            for(uint32_t sample=0;sample<samples;++sample){const unsigned pose=moving?firstCamera+((camera-firstCamera+sample)%6):camera;const auto current=moving?Camera(pose):c;
                const auto start=Clock::now();prepare(current);
                const double total=std::chrono::duration<double,std::milli>(Clock::now()-start).count();const auto& s=optimized?frame.Statistics():control?control->Statistics():reference.Statistics();
                const std::string label=optimized?"optimized":control?"control_"+std::to_string(controlFlags):"original_022";
                csv<<round<<','<<pose<<','<<ratio<<','<<k<<','<<label<<','<<(optimized?flags:controlFlags)<<','<<backend.Status().threads<<','<<block*samples+sample<<','<<s.candidates<<','<<s.eligible<<','<<s.k<<','<<s.prepareMs<<','<<s.selectorMs<<','<<s.topkMs<<','<<s.studentMs<<','<<s.decodeMs<<','<<total<<'\n';
            }
        }
    }
    csv.close();Require(bool(csv),"CPU benchmark write failed");std::cout<<"CPU_BENCH_COMPLETE flags="<<flags<<" threads="<<threads<<" rounds="<<rounds<<" samples_per_block="<<samples<<std::endl;
}catch(const std::exception& e){std::cerr<<"CPU_OPTIMIZATION_FAIL "<<e.what()<<std::endl;return 1;}}
