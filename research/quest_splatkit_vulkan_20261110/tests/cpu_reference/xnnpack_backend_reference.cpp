#include "xnnpack_backend_reference.h"
#include "scene_asset.h"
#include <xnnpack.h>
#include <cpuinfo.h>
#include <android/log.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mycartgs {
namespace {
constexpr uint32_t kBatch = 4096;
constexpr size_t kPadding = (XNN_EXTRA_BYTES+sizeof(float)-1)/sizeof(float);
void Check(xnn_status value,const char* stage) {
    if(value!=xnn_status_success)throw std::runtime_error(std::string(stage)+" XNNPACK error="+std::to_string(value));
}
}
struct ReferenceXnnpackBackend::Impl {
    pthreadpool_t pool = nullptr;
    uint32_t threads = 0;
    struct Graph {
        std::array<xnn_operator_t,3> layers{};
        xnn_operator_t softmax = nullptr;
        uint32_t inWidth=0,outWidth=0,lastRows=0;
        std::vector<float> input,h0,h1,output,probability;
        ~Graph(){for(auto op:layers)if(op)xnn_delete_operator(op);if(softmax)xnn_delete_operator(softmax);}
    };
    std::unique_ptr<Graph> selector,student;
    ~Impl(){selector.reset();student.reset();if(pool)pthreadpool_destroy(pool);}
    std::unique_ptr<Graph> Build(const SceneAsset& scene,bool isSelector) {
        auto graph=std::make_unique<Graph>();graph->inWidth=isSelector?19:62;graph->outWidth=isSelector?2:14;
        const uint32_t widths[]={graph->inWidth,32,32,graph->outWidth};
        const std::string prefix=isSelector?"selector":"student";
        for(uint32_t j=0;j<3;++j) {
            const auto& w=scene.Section(prefix+"_w"+std::to_string(j)+(!isSelector&&j==0?"_fused":""));
            const auto& b=scene.Section(prefix+"_b"+std::to_string(j));
            if(w.shape!=std::vector<uint32_t>{widths[j+1],widths[j]} || b.shape!=std::vector<uint32_t>{widths[j+1]})
                throw std::runtime_error("XNNPACK weight shape mismatch: "+prefix);
            Check(xnn_create_fully_connected_nc_f32(widths[j],widths[j+1],widths[j],widths[j+1],
                reinterpret_cast<const float*>(w.data),reinterpret_cast<const float*>(b.data),
                j<2?0.f:-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity(),
                0,nullptr,&graph->layers[j]),"create FC FP32");
        }
        if(isSelector)Check(xnn_create_softmax_nc_f32(0,&graph->softmax),"create Softmax FP32");
        graph->input.resize(size_t(kBatch)*graph->inWidth+kPadding);
        graph->h0.resize(size_t(kBatch)*32+kPadding);graph->h1.resize(size_t(kBatch)*32+kPadding);
        graph->output.resize(size_t(kBatch)*graph->outWidth+kPadding);
        if(isSelector)graph->probability.resize(size_t(kBatch)*2+kPadding);
        return graph;
    }
    void Run(Graph& g,uint32_t rows,const float* input,const float* noise,float* output) {
        if(rows && (!input||!output))throw std::runtime_error("Null XNNPACK IO");
        for(uint32_t begin=0;begin<rows;begin+=kBatch) {
            const uint32_t valid=std::min(kBatch,rows-begin);
            // Own the required XNN_EXTRA_BYTES padding; external caller buffers need no padding contract.
            std::copy_n(input+size_t(begin)*g.inWidth,size_t(valid)*g.inWidth,g.input.data());
            if(g.lastRows!=valid) {
                for(auto op:g.layers)Check(xnn_reshape_fully_connected_nc_f32(op,valid,pool),"reshape FC");
                if(g.softmax)Check(xnn_reshape_softmax_nc_f32(g.softmax,2,2,2,valid,pool),"reshape Softmax");
                g.lastRows=valid;
            }
            const float* in[]={g.input.data(),g.h0.data(),g.h1.data()};
            float* out[]={g.h0.data(),g.h1.data(),g.output.data()};
            for(uint32_t j=0;j<3;++j){Check(xnn_setup_fully_connected_nc_f32(g.layers[j],in[j],out[j]),"setup FC");
                Check(xnn_run_operator(g.layers[j],pool),"run FC");}
            const float* result=g.output.data();
            if(g.softmax) {
                if(!noise)throw std::runtime_error("Selector fixed noise is required");
                for(size_t j=0;j<size_t(valid)*2;++j)g.output[j]+=noise[size_t(begin)*2+j];
                Check(xnn_setup_softmax_nc_f32(g.softmax,g.output.data(),g.probability.data()),"setup Softmax");
                Check(xnn_run_operator(g.softmax,pool),"run Softmax");result=g.probability.data();
            }
            for(size_t j=0;j<size_t(valid)*g.outWidth;++j) {
                if(!std::isfinite(result[j]))throw std::runtime_error("XNNPACK returned nonfinite output");
                output[size_t(begin)*g.outWidth+j]=result[j];
            }
        }
    }
};
ReferenceXnnpackBackend::ReferenceXnnpackBackend() = default;
ReferenceXnnpackBackend::~ReferenceXnnpackBackend() = default;
void ReferenceXnnpackBackend::Initialize(uint32_t threads) {
    impl_.reset();status_={};
    if(threads<1||threads>4)throw std::runtime_error("XNNPACK thread count must be 1..4");
    Check(xnn_initialize(nullptr),"initialize");
#if defined(__aarch64__)
    if(!cpuinfo_has_arm_neon())throw std::runtime_error("ARM NEON unavailable");
#else
    throw std::runtime_error("Quest deployment requires ARM64");
#endif
    impl_=std::make_unique<Impl>();impl_->threads=threads;status_.threads=threads;
    if(threads>1){impl_->pool=pthreadpool_create(threads);if(!impl_->pool)throw std::runtime_error("XNNPACK thread pool allocation failed");}
    status_.state=InferenceState::GraphError;status_.message="XNNPACK ARM CPU initialized; load Student32 weights";
}
void ReferenceXnnpackBackend::Bind(const SceneAsset& scene) {
    if(!impl_)throw std::runtime_error("XNNPACK has not been initialized");
    status_.state=InferenceState::GraphError;status_.execution="NOT_EXECUTED";
    impl_->selector=impl_->Build(scene,true);impl_->student=impl_->Build(scene,false);
    status_.selectorCalls=status_.studentCalls=0;status_.workspaceBytes=0;
    for(const auto* g:{impl_->selector.get(),impl_->student.get()})
        status_.workspaceBytes+=(g->input.size()+g->h0.size()+g->h1.size()+g->output.size()+g->probability.size())*sizeof(float);
    status_.state=InferenceState::Ready;
    status_.message="XNNPACK / ARM CPU / FP32 / "+std::to_string(impl_->threads)+" THREADS";
    __android_log_print(ANDROID_LOG_INFO,"MyCartGS-XNNPACK","READY selector=19/32/32/2 student=62/32/32/14 threads=%u workspace=%llu",
        impl_->threads,(unsigned long long)status_.workspaceBytes);
}
void ReferenceXnnpackBackend::RunSelector(uint32_t rows,const float* input,const float* noise,float* output) {
    if(!Ready())throw std::runtime_error("Selector XNNPACK not ready");
    impl_->Run(*impl_->selector,rows,input,noise,output);
    if(rows){++status_.selectorCalls;status_.execution="XNNPACK_ARM_CPU_EXECUTED";}
}
void ReferenceXnnpackBackend::RunStudent(uint32_t rows,const float* input,float* output) {
    if(!Ready())throw std::runtime_error("Student XNNPACK not ready");
    impl_->Run(*impl_->student,rows,input,nullptr,output);
    if(rows){++status_.studentCalls;status_.execution="XNNPACK_ARM_CPU_EXECUTED";}
}
}  // namespace mycartgs
