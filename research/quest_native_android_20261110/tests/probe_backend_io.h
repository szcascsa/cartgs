#pragma once
#include "xnnpack_backend.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

template<class T> void ProbeWrite(const std::string& path,const T* data,size_t size) {
    std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(data),size*sizeof(T));
    if(!out)throw std::runtime_error("Probe output failed: "+path);
}
inline std::vector<float> ProbeRead(const std::string& path,size_t size) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    if(!in || size_t(in.tellg())!=size*sizeof(float))throw std::runtime_error("Reference absent or wrong shape: "+path);
    in.seekg(0);std::vector<float> data(size);in.read(reinterpret_cast<char*>(data.data()),size*sizeof(float));
    if(!in)throw std::runtime_error("Reference read failed");return data;
}
class ProbeBackend final : public mycartgs::InferenceBackend {
public:
    ProbeBackend(std::string prefix,uint32_t threads,std::string reference):prefix_(std::move(prefix)),reference_(std::move(reference)) {
        if(reference_.empty())cpu_.Initialize(threads);
        else{status_.state=mycartgs::InferenceState::Ready;status_.message="PyTorch oracle replay; test only";}
    }
    void Bind(const mycartgs::SceneAsset& scene) override {if(reference_.empty())cpu_.Bind(scene);}
    const mycartgs::InferenceStatus& Status() const override{return reference_.empty()?cpu_.Status():status_;}
    void RunSelector(uint32_t n,const float* x,const float* noise,float* out) override {
        ProbeWrite(prefix_+"_selector_input.bin",x,size_t(n)*19);ProbeWrite(prefix_+"_noise.bin",noise,size_t(n)*2);
        if(reference_.empty())cpu_.RunSelector(n,x,noise,out);
        else{auto v=ProbeRead(reference_+"_selector_reference.bin",size_t(n)*2);std::copy(v.begin(),v.end(),out);}
        ProbeWrite(prefix_+"_selector_output.bin",out,size_t(n)*2);
    }
    void RunStudent(uint32_t n,const float* x,float* out) override {
        ProbeWrite(prefix_+"_student_input.bin",x,size_t(n)*62);
        if(reference_.empty())cpu_.RunStudent(n,x,out);
        else {
            const auto expected=ProbeRead(reference_+"_student_reference_input.bin",size_t(n)*62);
            if(std::memcmp(expected.data(),x,size_t(n)*62*sizeof(float)))
                throw std::runtime_error("REFERENCE_INPUT_CAPTURED: regenerate Student oracle using this probe input");
            auto v=ProbeRead(reference_+"_student_reference.bin",size_t(n)*14);std::copy(v.begin(),v.end(),out);
        }
        ProbeWrite(prefix_+"_student_output.bin",out,size_t(n)*14);
    }
private:
    mycartgs::XnnpackBackend cpu_;
    mycartgs::InferenceStatus status_;
    std::string prefix_,reference_;
};
