#include "xnnpack_backend.h"
#include "scene_asset.h"
#include <xnnpack.h>
#include <cpuinfo.h>
#include <android/log.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace mycartgs {
namespace {
constexpr uint32_t kBatch = 4096;
constexpr uint32_t kLargeBatch = 16384;
constexpr size_t kPadding = (XNN_EXTRA_BYTES+sizeof(float)-1)/sizeof(float);
void Check(xnn_status value,const char* stage) {
    if(value!=xnn_status_success)throw std::runtime_error(std::string(stage)+" XNNPACK error="+std::to_string(value));
}
}
struct XnnpackBackend::Impl {
    pthreadpool_t pool = nullptr;
    uint32_t threads = 0;
    const SceneAsset* scene=nullptr;
    struct Graph {
        std::array<xnn_operator_t,3> layers{};
        xnn_operator_t softmax = nullptr;
        uint32_t inWidth=0,outWidth=0,lastRows=0,capacity=0;
        bool fixedSetup=false;
        std::vector<float> input,h0,h1,output,probability;
        ~Graph(){for(auto op:layers)if(op)xnn_delete_operator(op);if(softmax)xnn_delete_operator(softmax);}
    };
    std::unique_ptr<Graph> selector,student,selectorTail,studentTail;
    ~Impl(){selector.reset();student.reset();selectorTail.reset();studentTail.reset();if(pool)pthreadpool_destroy(pool);}
    std::unique_ptr<Graph> Build(const SceneAsset& scene,bool isSelector,uint32_t capacity=0) {
        auto graph=std::make_unique<Graph>();graph->inWidth=isSelector?19:62;graph->outWidth=isSelector?2:14;
        graph->capacity=capacity?capacity:std::max(kLargeBatch,scene.Identity().gaussianCount);
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
        graph->input.resize(size_t(graph->capacity)*graph->inWidth+kPadding);
        graph->h0.resize(size_t(kLargeBatch)*32+kPadding);graph->h1.resize(size_t(kLargeBatch)*32+kPadding);
        graph->output.resize(size_t(kLargeBatch)*graph->outWidth+kPadding);
        if(isSelector)graph->probability.resize(size_t(kLargeBatch)*2+kPadding);
        return graph;
    }
    void Run(Graph& g,uint32_t rows,const float* input,const float* noise,float* output,bool direct=false,bool fixedSetup=false,uint32_t batch=kBatch) {
        if(rows && (!input||!output))throw std::runtime_error("Null XNNPACK IO");
        for(uint32_t begin=0;begin<rows;begin+=batch) {
            const uint32_t valid=std::min(batch,rows-begin);
            // Own the required XNN_EXTRA_BYTES padding; external caller buffers need no padding contract.
            if(!direct)std::copy_n(input+size_t(begin)*g.inWidth,size_t(valid)*g.inWidth,g.input.data());
            if(g.lastRows!=valid) {
                for(auto op:g.layers)Check(xnn_reshape_fully_connected_nc_f32(op,valid,pool),"reshape FC");
                if(g.softmax)Check(xnn_reshape_softmax_nc_f32(g.softmax,2,2,2,valid,pool),"reshape Softmax");
                g.lastRows=valid;
                g.fixedSetup=false;
            }
            const float* in[]={direct?input+size_t(begin)*g.inWidth:g.input.data(),g.h0.data(),g.h1.data()};
            float* out[]={g.h0.data(),g.h1.data(),g.output.data()};
            for(uint32_t j=0;j<3;++j){if(!fixedSetup||!g.fixedSetup||j==0)Check(xnn_setup_fully_connected_nc_f32(g.layers[j],in[j],out[j]),"setup FC");
                Check(xnn_run_operator(g.layers[j],pool),"run FC");}
            const float* result=g.output.data();
            if(g.softmax) {
                if(!noise)throw std::runtime_error("Selector fixed noise is required");
                for(size_t j=0;j<size_t(valid)*2;++j)g.output[j]+=noise[size_t(begin)*2+j];
                if(!fixedSetup||!g.fixedSetup)Check(xnn_setup_softmax_nc_f32(g.softmax,g.output.data(),g.probability.data()),"setup Softmax");
                Check(xnn_run_operator(g.softmax,pool),"run Softmax");result=g.probability.data();
            }
            for(size_t j=0;j<size_t(valid)*g.outWidth;++j) {
                if(!std::isfinite(result[j]))throw std::runtime_error("XNNPACK returned nonfinite output");
                output[size_t(begin)*g.outWidth+j]=result[j];
            }
            g.fixedSetup=fixedSetup;
        }
    }
    void Dispatch(Graph& graph,Graph* tail,uint32_t rows,const float* input,const float* noise,float* output,uint32_t flags) {
        const bool direct=(flags&CpuDirectInput)&&input==graph.input.data();
        const uint32_t batch=(flags&CpuLargeBatch)?kLargeBatch:kBatch;
        if(!(flags&CpuStableBatchPlans)){Run(graph,rows,input,noise,output,direct,false,batch);return;}
        for(uint32_t begin=0;begin<rows;begin+=batch){
            const uint32_t valid=std::min(batch,rows-begin);
            if(!tail)throw std::runtime_error("Missing tail batch plan");
            Run(valid==batch?graph:*tail,valid,input+size_t(begin)*graph.inWidth,
                noise?noise+size_t(begin)*2:nullptr,output+size_t(begin)*graph.outWidth,direct,true,batch);
        }
    }
};
XnnpackBackend::XnnpackBackend() = default;
XnnpackBackend::~XnnpackBackend() = default;
void XnnpackBackend::SetOptimizationFlags(uint32_t flags) {
    flags_=flags;
    if(impl_&&impl_->scene&&(flags&CpuStableBatchPlans)&&!impl_->selectorTail){
        impl_->selectorTail=impl_->Build(*impl_->scene,true,kLargeBatch);impl_->studentTail=impl_->Build(*impl_->scene,false,kLargeBatch);
        for(const auto* g:{impl_->selectorTail.get(),impl_->studentTail.get()})
            status_.workspaceBytes+=(g->input.size()+g->h0.size()+g->h1.size()+g->output.size()+g->probability.size())*sizeof(float);
    }
}
void XnnpackBackend::SetThreadCount(uint32_t threads) {
    if(threads<1||threads>4)throw std::runtime_error("XNNPACK thread count must be 1..4");
    if(!impl_)throw std::runtime_error("XNNPACK has not been initialized");
    if(impl_->threads==threads)return;
    if(impl_->pool)pthreadpool_destroy(impl_->pool);
    impl_->pool=threads>1?pthreadpool_create(threads):nullptr;
    if(threads>1&&!impl_->pool)throw std::runtime_error("XNNPACK thread pool allocation failed");
    impl_->threads=threads;status_.threads=threads;
    for(auto* graph:{impl_->selector.get(),impl_->student.get(),impl_->selectorTail.get(),impl_->studentTail.get()})
        if(graph){graph->lastRows=0;graph->fixedSetup=false;}
}
void XnnpackBackend::Initialize(uint32_t threads) {
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
void XnnpackBackend::Bind(const SceneAsset& scene) {
    if(!impl_)throw std::runtime_error("XNNPACK has not been initialized");
    status_.state=InferenceState::GraphError;status_.execution="NOT_EXECUTED";
    impl_->scene=&scene;impl_->selectorTail.reset();impl_->studentTail.reset();
    impl_->selector=impl_->Build(scene,true);impl_->student=impl_->Build(scene,false);
    status_.selectorCalls=status_.studentCalls=0;status_.workspaceBytes=0;
    for(const auto* g:{impl_->selector.get(),impl_->student.get(),impl_->selectorTail.get(),impl_->studentTail.get()})
        if(g)status_.workspaceBytes+=(g->input.size()+g->h0.size()+g->h1.size()+g->output.size()+g->probability.size())*sizeof(float);
    SetOptimizationFlags(flags_);
    status_.state=InferenceState::Ready;
    status_.message="XNNPACK / ARM CPU / FP32 / "+std::to_string(impl_->threads)+" THREADS";
    __android_log_print(ANDROID_LOG_INFO,"MyCartGS-XNNPACK","READY selector=19/32/32/2 student=62/32/32/14 threads=%u workspace=%llu",
        impl_->threads,(unsigned long long)status_.workspaceBytes);
}
void XnnpackBackend::RunSelector(uint32_t rows,const float* input,const float* noise,float* output) {
    if(!Ready())throw std::runtime_error("Selector XNNPACK not ready");
    impl_->Dispatch(*impl_->selector,impl_->selectorTail.get(),rows,input,noise,output,flags_);
    if(rows){++status_.selectorCalls;status_.execution="XNNPACK_ARM_CPU_EXECUTED";}
}
void XnnpackBackend::RunStudent(uint32_t rows,const float* input,float* output) {
    if(!Ready())throw std::runtime_error("Student XNNPACK not ready");
    impl_->Dispatch(*impl_->student,impl_->studentTail.get(),rows,input,nullptr,output,flags_);
    if(rows){++status_.studentCalls;status_.execution="XNNPACK_ARM_CPU_EXECUTED";}
}
float* XnnpackBackend::SelectorInput(uint32_t rows) {
    if(!(flags_&CpuDirectInput))return nullptr;
    if(!Ready()||rows>impl_->selector->capacity)throw std::runtime_error("Selector input capacity exceeded");
    return impl_->selector->input.data();
}
float* XnnpackBackend::StudentInput(uint32_t rows) {
    if(!(flags_&CpuDirectInput))return nullptr;
    if(!Ready()||rows>impl_->student->capacity)throw std::runtime_error("Student input capacity exceeded");
    return impl_->student->input.data();
}
void XnnpackBackend::ParallelFor(size_t range,size_t tile,RangeTask task,void* context) const {
    if(!range)return;
    if(!impl_||!impl_->pool||range<=tile){task(context,0,range);return;}
    struct Work { RangeTask task;void* context;std::mutex mutex;std::exception_ptr error;size_t errorBegin=SIZE_MAX; } work{task,context};
    pthreadpool_parallelize_1d_tile_1d(impl_->pool,[](void* opaque,size_t begin,size_t count){
        auto& work=*static_cast<Work*>(opaque);
        try{work.task(work.context,begin,count);}catch(...){std::lock_guard<std::mutex> guard(work.mutex);if(begin<work.errorBegin){work.error=std::current_exception();work.errorBegin=begin;}}
    },&work,range,tile,0);
    if(work.error)std::rethrow_exception(work.error);
}
}  // namespace mycartgs
