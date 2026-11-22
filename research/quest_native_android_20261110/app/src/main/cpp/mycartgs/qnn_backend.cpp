#include "qnn_backend.h"
#include "scene_asset.h"
#include <QnnInterface.h>
#include <QnnOpDef.h>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpGraph.h>
#include <android/log.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <list>
#include <stdexcept>
#include <vector>
#include <unistd.h>

namespace mycartgs {
namespace {
constexpr uint32_t kBatch = 1024;
void Check(Qnn_ErrorHandle_t error, const std::string& stage) {
    if (error != QNN_SUCCESS) throw std::runtime_error(stage + " QNN error=" + std::to_string(error));
}
void Log(const char* format, QnnLog_Level_t level, uint64_t, va_list args) {
    char message[2048]; std::vsnprintf(message, sizeof(message), format, args);
    __android_log_print(level <= QNN_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR : ANDROID_LOG_INFO, "MyCartGS-QNN", "%s", message);
}
}

struct QnnBackend::Impl {
    QNN_INTERFACE_VER_TYPE api{};
    Qnn_BackendHandle_t backend = nullptr;
    Qnn_DeviceHandle_t device = nullptr;
    Qnn_ContextHandle_t context = nullptr;
    Qnn_LogHandle_t logger = nullptr;
    Qnn_ProfileHandle_t profile = nullptr;
    std::vector<void*> libraries;
    struct Tensor {
        std::string name;
        std::vector<uint32_t> dims;
        Qnn_Tensor_t value = QNN_TENSOR_INIT;
    };
    struct Graph {
        Qnn_GraphHandle_t handle = nullptr;
        std::list<Tensor> tensors;
        std::vector<Qnn_Tensor_t> inputs;
        Qnn_Tensor_t output = QNN_TENSOR_INIT;
        std::vector<float> input, noise, result;
        uint32_t inWidth = 0, outWidth = 0;
    } selector, student;

    ~Impl() {
        if (context && api.contextFree) api.contextFree(context,nullptr);
        if (profile && api.profileFree) api.profileFree(profile);
        if (device && api.deviceFree) api.deviceFree(device);
        if (backend && api.backendFree) api.backendFree(backend);
        if (logger && api.logFree) api.logFree(logger);
        for (auto it=libraries.rbegin();it!=libraries.rend();++it) dlclose(*it);
    }
    Qnn_Tensor_t TensorFor(Graph& graph,const std::string& name,std::vector<uint32_t> dims,
                          Qnn_TensorType_t type,const float* data=nullptr) {
        graph.tensors.emplace_back(); auto& tensor=graph.tensors.back();
        tensor.name=name; tensor.dims=std::move(dims);
        auto& v=tensor.value.v1;
        v.name=tensor.name.c_str(); v.type=type; v.dataFormat=QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        v.dataType=QNN_DATATYPE_FLOAT_32; v.rank=uint32_t(tensor.dims.size()); v.dimensions=tensor.dims.data();
        v.memType=QNN_TENSORMEMTYPE_RAW;
        if (data) {
            uint32_t elements=1; for (uint32_t d:tensor.dims) elements*=d;
            v.clientBuf={const_cast<float*>(data),elements*uint32_t(sizeof(float))};
        }
        Check(api.tensorCreateGraphTensor(graph.handle,&tensor.value),"tensor "+name);
        return tensor.value;
    }
    void Node(Graph& graph,const std::string& name,const char* type,std::vector<Qnn_Tensor_t> inputs,
              Qnn_Tensor_t output,std::vector<Qnn_Param_t> params={}) {
        Qnn_OpConfig_t op=QNN_OPCONFIG_INIT;
        op.v1.name=name.c_str(); op.v1.packageName=QNN_OP_PACKAGE_NAME_QTI_AISW; op.v1.typeName=type;
        op.v1.numOfInputs=uint32_t(inputs.size()); op.v1.inputTensors=inputs.data();
        op.v1.numOfOutputs=1; op.v1.outputTensors=&output;
        op.v1.numOfParams=uint32_t(params.size()); op.v1.params=params.data();
        Check(api.graphAddNode(graph.handle,op),"node "+name);
    }
    void Build(Graph& graph,const SceneAsset& scene,bool isSelector) {
        graph.inWidth=isSelector?19:62; graph.outWidth=isSelector?2:14;
        const std::string name=isSelector?"selector_b1024":"student32_b1024";
        QnnHtpGraph_CustomConfig_t precision=QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
        precision.option=QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION; precision.precision=QNN_PRECISION_FLOAT16;
        QnnGraph_Config_t config=QNN_GRAPH_CONFIG_INIT;
        config.option=QNN_GRAPH_CONFIG_OPTION_CUSTOM; config.customConfig=&precision;
        const QnnGraph_Config_t* configs[]={&config,nullptr};
        Check(api.graphCreate(context,name.c_str(),configs,&graph.handle),"graphCreate "+name);
        auto current=TensorFor(graph,"input",{kBatch,graph.inWidth},QNN_TENSOR_TYPE_APP_WRITE);
        graph.inputs.push_back(current);
        const std::string prefix=isSelector?"selector":"student";
        for (uint32_t layer=0;layer<3;++layer) {
            const auto& w=scene.Section(prefix+"_w"+std::to_string(layer)+(!isSelector && layer==0?"_fused":""));
            const auto& b=scene.Section(prefix+"_b"+std::to_string(layer));
            auto weights=TensorFor(graph,"weight"+std::to_string(layer),w.shape,QNN_TENSOR_TYPE_STATIC,
                                    reinterpret_cast<const float*>(w.data));
            auto bias=TensorFor(graph,"bias"+std::to_string(layer),b.shape,QNN_TENSOR_TYPE_STATIC,
                                 reinterpret_cast<const float*>(b.data));
            const bool last=layer==2;
            auto next=TensorFor(graph,"fc"+std::to_string(layer),{kBatch,w.shape[0]},
                                last&&!isSelector?QNN_TENSOR_TYPE_APP_READ:QNN_TENSOR_TYPE_NATIVE);
            Node(graph,"fc"+std::to_string(layer),QNN_OP_FULLY_CONNECTED,{current,weights,bias},next);
            current=next;
            if (!last) {
                next=TensorFor(graph,"relu"+std::to_string(layer),{kBatch,w.shape[0]},QNN_TENSOR_TYPE_NATIVE);
                Node(graph,"relu"+std::to_string(layer),QNN_OP_RELU,{current},next); current=next;
            }
        }
        if (isSelector) {
            auto noise=TensorFor(graph,"fixed_noise",{kBatch,2},QNN_TENSOR_TYPE_APP_WRITE);
            graph.inputs.push_back(noise);
            auto noisy=TensorFor(graph,"noisy_logits",{kBatch,2},QNN_TENSOR_TYPE_NATIVE);
            Node(graph,"add_fixed_noise",QNN_OP_ELEMENT_WISE_ADD,{current,noise},noisy);
            auto probability=TensorFor(graph,"probability",{kBatch,2},QNN_TENSOR_TYPE_APP_READ);
            Qnn_Param_t axis=QNN_PARAM_INIT; axis.name=QNN_OP_SOFTMAX_PARAM_AXIS;
            axis.paramType=QNN_PARAMTYPE_SCALAR; axis.scalarParam.dataType=QNN_DATATYPE_UINT_32;
            axis.scalarParam.uint32Value=1;
            Node(graph,"softmax",QNN_OP_SOFTMAX,{noisy},probability,{axis}); current=probability;
            graph.noise.resize(kBatch*2);
        }
        graph.output=current;
        Check(api.graphFinalize(graph.handle,nullptr,nullptr),"graphFinalize "+name);
        graph.input.resize(kBatch*graph.inWidth); graph.result.resize(kBatch*graph.outWidth);
    }
    void Profile(const char* graphName) {
        const QnnProfile_EventId_t* events=nullptr; uint32_t count=0;
        if (!profile || api.profileGetEvents(profile,&events,&count)!=QNN_SUCCESS) return;
        for (uint32_t i=0;i<count;++i) {
            QnnProfile_EventData_t data=QNN_PROFILE_EVENT_DATA_INIT;
            if (api.profileGetEventData(events[i],&data)==QNN_SUCCESS)
                __android_log_print(ANDROID_LOG_INFO,"MyCartGS-QNN","PROFILE %s type=%u unit=%u value=%llu id=%s",
                  graphName,data.type,data.unit,(unsigned long long)data.value,data.identifier?data.identifier:"");
            const QnnProfile_EventId_t* sub=nullptr; uint32_t subCount=0;
            if (api.profileGetSubEvents(events[i],&sub,&subCount)==QNN_SUCCESS) for(uint32_t j=0;j<subCount;++j) {
                if (api.profileGetEventData(sub[j],&data)==QNN_SUCCESS)
                    __android_log_print(ANDROID_LOG_INFO,"MyCartGS-QNN","PROFILE_DEVICE %s type=%u unit=%u value=%llu id=%s",
                      graphName,data.type,data.unit,(unsigned long long)data.value,data.identifier?data.identifier:"");
            }
        }
    }
    void Run(Graph& graph,uint32_t rows,const float* input,const float* noise,float* output,bool profileOnce) {
        for(uint32_t begin=0;begin<rows;begin+=kBatch) {
            const uint32_t valid=std::min(kBatch,rows-begin);
            std::fill(graph.input.begin(),graph.input.end(),0.f);
            std::copy_n(input+size_t(begin)*graph.inWidth,size_t(valid)*graph.inWidth,graph.input.data());
            graph.inputs[0].v1.clientBuf={graph.input.data(),uint32_t(graph.input.size()*sizeof(float))};
            if(noise) {
                std::fill(graph.noise.begin(),graph.noise.end(),0.f);
                std::copy_n(noise+size_t(begin)*2,size_t(valid)*2,graph.noise.data());
                graph.inputs[1].v1.clientBuf={graph.noise.data(),uint32_t(graph.noise.size()*sizeof(float))};
            }
            graph.output.v1.clientBuf={graph.result.data(),uint32_t(graph.result.size()*sizeof(float))};
            Check(api.graphExecute(graph.handle,graph.inputs.data(),uint32_t(graph.inputs.size()),&graph.output,1,
                                    profileOnce?profile:nullptr,nullptr),"HTP graphExecute");
            if(profileOnce) { Profile(noise?"selector":"student32"); profileOnce=false; }
            for(size_t i=0;i<size_t(valid)*graph.outWidth;++i) {
                if(!std::isfinite(graph.result[i]))throw std::runtime_error("HTP graph returned nonfinite output");
                output[size_t(begin)*graph.outWidth+i]=graph.result[i];
            }
        }
    }
};

QnnBackend::QnnBackend() = default;
QnnBackend::~QnnBackend() = default;
void QnnBackend::Close() { impl_.reset(); status_=QnnStatus{}; }
QnnStatus QnnBackend::Initialize(const std::string& directory, bool signedPd) {
    Close(); impl_=std::make_unique<Impl>();
    try {
        const std::string dsp=directory+";/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp";
        setenv("ADSP_LIBRARY_PATH",dsp.c_str(),1);
        // NativeActivity and the XR loader may use different linker namespaces. Preload the declared public FastRPC dependency.
        void* rpc=dlopen("/vendor/lib64/libcdsprpc.so",RTLD_NOW|RTLD_GLOBAL);
        if(!rpc)throw std::runtime_error(std::string("Public libcdsprpc load: ")+dlerror());
        impl_->libraries.push_back(rpc);
        void* library=dlopen((directory+"/libQnnHtp.so").c_str(),RTLD_NOW|RTLD_GLOBAL);
        if(!library)throw std::runtime_error(std::string("libQnnHtp.so: ")+dlerror());
        impl_->libraries.push_back(library);
        auto providers=reinterpret_cast<decltype(&QnnInterface_getProviders)>(dlsym(library,"QnnInterface_getProviders"));
        if(!providers)throw std::runtime_error("HTP interface provider missing");
        const QnnInterface_t** entries=nullptr; uint32_t count=0;
        Check(providers(&entries,&count),"getProviders");
        bool found=false;
        for(uint32_t i=0;i<count;++i) {
            if(entries[i]->apiVersion.coreApiVersion.major==QNN_API_VERSION_MAJOR &&
               entries[i]->apiVersion.coreApiVersion.minor>=QNN_API_VERSION_MINOR) {
                impl_->api=entries[i]->QNN_INTERFACE_VER_NAME; found=true; break;
            }
        }
        if(!found)throw std::runtime_error("HTP QNN API version mismatch");
        // Quest hides sysfs soc_id from shell and apps. Use the documented SoC override before any backend API.
        QnnGlobalConfig_t global=QNN_GLOBAL_CONFIG_INIT;
        global.option=QNN_GLOBAL_CONFIG_OPTION_SOC_MODEL; global.socModel=QNN_SOC_MODEL_SXR2230P;
        const QnnGlobalConfig_t* globalConfigs[]={&global,nullptr};
        Check(impl_->api.globalConfigSet(globalConfigs),"globalConfigSet SXR2230P");
        Check(impl_->api.logCreate(Log,QNN_LOG_LEVEL_INFO,&impl_->logger),"logCreate");
        Check(impl_->api.backendCreate(impl_->logger,nullptr,&impl_->backend),"backendCreate");
        QnnHtpDevice_CustomConfig_t soc{}; soc.option=QNN_HTP_DEVICE_CONFIG_OPTION_SOC; soc.socModel=53;
        QnnHtpDevice_CustomConfig_t arch{}; arch.option=QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
        arch.arch={0,QNN_HTP_DEVICE_ARCH_V69};
        QnnDevice_Config_t sc=QNN_DEVICE_CONFIG_INIT, ac=QNN_DEVICE_CONFIG_INIT;
        sc.option=QNN_DEVICE_CONFIG_OPTION_CUSTOM; sc.customConfig=&soc;
        ac.option=QNN_DEVICE_CONFIG_OPTION_CUSTOM; ac.customConfig=&arch;
        QnnHtpDevice_CustomConfig_t pd{}; pd.option=QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD;
        pd.useSignedProcessDomain={0,signedPd};
        QnnDevice_Config_t pc=QNN_DEVICE_CONFIG_INIT;
        pc.option=QNN_DEVICE_CONFIG_OPTION_CUSTOM; pc.customConfig=&pd;
        const QnnDevice_Config_t* configs[]={&sc,&ac,&pc,nullptr};
        __android_log_print(ANDROID_LOG_INFO,"MyCartGS-QNN","DEVICE_REQUEST soc=53 arch=69 pd=%s",signedPd?"signed":"unsigned");
        status_.errorCode=impl_->api.deviceCreate(impl_->logger,configs,&impl_->device);
        Check(status_.errorCode,"deviceCreate SoC53/V69");
        const QnnDevice_PlatformInfo_t* platform=nullptr;
        Check(impl_->api.deviceGetPlatformInfo(impl_->logger,&platform),"deviceGetPlatformInfo");
        bool verified=false;
        if(platform && platform->version==QNN_DEVICE_PLATFORM_INFO_VERSION_1) {
            for(uint32_t i=0;i<platform->v1.numHwDevices;++i) {
                auto* extension=static_cast<QnnHtpDevice_DeviceInfoExtension_t*>(platform->v1.hwDevices[i].v1.deviceInfoExtension);
                if(extension && extension->devType==QNN_HTP_DEVICE_TYPE_ON_CHIP) {
                    status_.arch=uint32_t(extension->onChipDevice.arch); status_.soc=extension->onChipDevice.socModel;
                    verified=status_.arch==69 && status_.soc==53;
                }
            }
        }
        if(platform) impl_->api.deviceFreePlatformInfo(impl_->logger,platform);
        if(!verified)throw std::runtime_error("Actual HTP device is not verified SoC53/V69");
        Check(impl_->api.profileCreate(impl_->backend,QNN_PROFILE_LEVEL_BASIC,&impl_->profile),"profileCreate");
        status_.state=QnnState::GraphError;
        status_.message="HTP V69 device verified; load a scene to compile graphs";
    } catch(const std::exception& error) {
        status_.state=QnnState::DeviceError; status_.message=error.what();
        __android_log_print(ANDROID_LOG_ERROR,"MyCartGS-QNN","%s",error.what());
        if(status_.errorCode==QNN_DEVICE_ERROR_INVALID_CONFIG) {
            // A direct-node denial alone is normal on HAL-based devices. Only diagnose after QNN itself failed.
            const auto probe=[](const char* path) {const int fd=open(path,O_RDONLY|O_CLOEXEC);
                if(fd<0)return errno;close(fd);return 0;};
            status_.rpcErrno=probe("/dev/adsprpc-smd");
            status_.secureRpcErrno=probe("/dev/adsprpc-smd-secure");
            const auto denied=[](int code){return code==EACCES||code==EPERM;};
            status_.transportAccessDenied=denied(status_.rpcErrno)&&denied(status_.secureRpcErrno);
            __android_log_print(ANDROID_LOG_ERROR,"MyCartGS-QNN",
                "TRANSPORT_DIAGNOSTIC error=%llu pd=%s direct_errno=%d secure_errno=%d; verify FastRPC HAL and SELinux in logcat",
                (unsigned long long)status_.errorCode,signedPd?"signed":"unsigned",status_.rpcErrno,status_.secureRpcErrno);
            if(status_.transportAccessDenied)status_.message="HTP 14001: FASTRPC / DSP ACCESS DENIED";
        }
    }
    return status_;
}
void QnnBackend::Bind(const SceneAsset& scene) {
    if(!impl_ || !impl_->device)throw std::runtime_error(status_.message);
    try {
        if(impl_->context) { Check(impl_->api.contextFree(impl_->context,nullptr),"contextFree"); impl_->context=nullptr; }
        impl_->selector=Impl::Graph{}; impl_->student=Impl::Graph{};
        Check(impl_->api.contextCreate(impl_->backend,impl_->device,nullptr,&impl_->context),"contextCreate");
        impl_->Build(impl_->selector,scene,true); impl_->Build(impl_->student,scene,false);
        status_.state=QnnState::Ready; status_.selectorCalls=status_.studentCalls=0;
        status_.message="HTP V69 / SoC53 / FP16 kernels / FP32 IO";
    } catch(const std::exception& error) { status_.state=QnnState::GraphError; status_.message=error.what(); throw; }
}
void QnnBackend::RunSelector(uint32_t rows,const float* input,const float* noise,float* output) {
    if(!Ready())throw std::runtime_error("Selector HTP not ready: "+status_.message);
    impl_->Run(impl_->selector,rows,input,noise,output,status_.selectorCalls==0);
    if(rows) { ++status_.selectorCalls; status_.execution="HTP_V69_EXECUTED"; }
}
void QnnBackend::RunStudent(uint32_t rows,const float* input,float* output) {
    if(!Ready())throw std::runtime_error("Student HTP not ready: "+status_.message);
    impl_->Run(impl_->student,rows,input,nullptr,output,status_.studentCalls==0);
    if(rows) { ++status_.studentCalls; status_.execution="HTP_V69_EXECUTED"; }
}
}  // namespace mycartgs
