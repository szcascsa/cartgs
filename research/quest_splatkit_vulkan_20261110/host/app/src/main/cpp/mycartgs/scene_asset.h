#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <string>
#include <unordered_map>
#include <vector>

struct AAssetManager;

namespace mycartgs {

struct SceneDescriptor {
    std::string slug, label, assetPath;
    std::string assetSha256, teacherFingerprint, studentCheckpointSha256;
    uint32_t gaussianCount = 0;
    uint32_t anchorCount = 0;
    std::array<float,3> viewPosition{};
    std::array<float,9> viewRotation{{1,0,0,0,1,0,0,0,1}};
};

std::vector<SceneDescriptor> LoadSceneCatalog(AAssetManager* manager);

struct SceneSection {
    size_t offset = 0;
    size_t bytes = 0;
    std::string dtype;
    std::vector<uint32_t> shape;
    std::string sha256;
    const uint8_t* data = nullptr;
};

struct SceneIdentity {
    std::string slug;
    std::string sceneId;
    std::string teacherFingerprint;
    std::string studentCheckpointSha256;
    std::string studentSchema;
    uint32_t gaussianCount = 0;
    uint32_t anchorCount = 0;
    uint32_t studentParameters = 0;
    uint32_t selectorParameters = 0;
    float viewSceneScale = 1;
    std::array<float,6> ratioKnots{};
    std::array<float,2> ratioBounds{};
    uint32_t viewSchema = 0;
    bool softplusScale = true;
};

class SceneAsset {
public:
    void Load(AAssetManager* manager, const SceneDescriptor& expected);
    void LoadFile(const std::string& path, const SceneDescriptor& expected);
    const SceneIdentity& Identity() const { return identity_; }
    const SceneSection& Section(const std::string& name) const;
    const std::vector<uint8_t>& Bytes() const { return bytes_; }

private:
    void Validate(const SceneDescriptor& expected);
    std::vector<uint8_t> bytes_;
    std::unordered_map<std::string, SceneSection> sections_;
    SceneIdentity identity_;
};

}  // namespace mycartgs
