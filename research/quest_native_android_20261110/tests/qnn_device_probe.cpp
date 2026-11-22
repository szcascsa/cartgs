#include "qnn_backend.h"
#include "scene_asset.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

template<class T> void Write(const std::string& path,const std::vector<T>& values) {
    std::ofstream out(path,std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(T));
    if(!out)throw std::runtime_error("Probe output write failed");
}
int main(int argc,char** argv) {
    try {
        if(argc!=4 && argc!=5)throw std::runtime_error("usage: qnn_device_probe libdir scene.mcgs output-prefix [signed]");
        mycartgs::SceneDescriptor identity;
        identity.slug="Replica__office2__rgbd"; identity.gaussianCount=43197; identity.anchorCount=16723;
        identity.assetSha256="1d735b0871ff94cc6acbf039ad014e0d1490ca4e209caef92648a0b9e53a6685";
        identity.teacherFingerprint="9e2d413fa7864127465394fa22bb3da4e05d912d4c9e0822a4e928933aa36804";
        identity.studentCheckpointSha256="fd85285c6bd5871f4b41c85690597ccadc965d5a5ec6f474a983e4e09f547a75";
        mycartgs::SceneAsset scene; scene.LoadFile(argv[2],identity);
        mycartgs::QnnBackend backend;
        const bool signedPd=argc==5 && std::string(argv[4])=="signed";
        std::cout << "INITIALIZE pd="<<(signedPd?"signed":"unsigned")<<" "<<backend.Initialize(argv[1],signedPd).message<<std::endl;
        backend.Bind(scene);
        constexpr uint32_t rows=1027;
        std::vector<float> selector(rows*19),noise(rows*2),student(rows*62),score(rows*2),residual(rows*14);
        const float* physical=reinterpret_cast<const float*>(scene.Section("selector_static10").data);
        const float* fixedNoise=reinterpret_cast<const float*>(scene.Section("noise2").data);
        const float* offset=reinterpret_cast<const float*>(scene.Section("offset3").data);
        const float* feature=reinterpret_cast<const float*>(scene.Section("residual48_0").data);
        const int32_t* owner=reinterpret_cast<const int32_t*>(scene.Section("owners").data);
        for(uint32_t i=0;i<rows;++i) {
            std::copy_n(physical+i*10,10,selector.data()+i*19); selector[i*19+10]=std::log(.3f);
            std::copy_n(fixedNoise+i*2,2,noise.data()+i*2);
            std::copy_n(offset+i*3,3,student.data()+i*62);
            std::copy_n(feature+owner[i]*48,48,student.data()+i*62+3);
            student[i*62+51]=.2f; student[i*62+52]=.30901699f; student[i*62+53]=.95105652f;
        }
        backend.RunSelector(rows,selector.data(),noise.data(),score.data());
        backend.RunStudent(rows,student.data(),residual.data());
        const std::string prefix=argv[3];
        Write(prefix+"_selector_in.bin",selector); Write(prefix+"_noise.bin",noise);
        Write(prefix+"_student_in.bin",student); Write(prefix+"_score.bin",score); Write(prefix+"_residual.bin",residual);
        for(float value:residual)if(!std::isfinite(value))throw std::runtime_error("Nonfinite Student residual");
        std::cout << "PROBE_PASS rows="<<rows<<" soc="<<backend.Status().soc<<" arch="<<backend.Status().arch
                  <<" execution="<<backend.Status().execution<<std::endl;
    } catch(const std::exception& error) { std::cerr<<"PROBE_FAIL "<<error.what()<<std::endl; return 1; }
}
