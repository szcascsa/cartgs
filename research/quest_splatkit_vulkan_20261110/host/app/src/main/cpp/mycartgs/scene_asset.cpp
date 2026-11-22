#include "scene_asset.h"

#include "sha256.h"

#include <android/asset_manager.h>
#include <json/json.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>

namespace mycartgs {
namespace {

constexpr size_t kMaximumSceneBytes = 512ull * 1024ull * 1024ull;

uint32_t ReadU32(const uint8_t* input) {
    return uint32_t(input[0]) | (uint32_t(input[1]) << 8) | (uint32_t(input[2]) << 16) | (uint32_t(input[3]) << 24);
}

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void RequireShape(const std::unordered_map<std::string, SceneSection>& sections, const char* name,
                  std::initializer_list<uint32_t> expected) {
    auto item = sections.find(name);
    Require(item != sections.end(), std::string("Missing scene section: ") + name);
    Require(item->second.shape == std::vector<uint32_t>(expected), std::string("Invalid section shape: ") + name);
}

}  // namespace

std::vector<SceneDescriptor> LoadSceneCatalog(AAssetManager* manager) {
    Require(manager != nullptr, "Android AssetManager is null");
    std::unique_ptr<AAsset, decltype(&AAsset_close)> asset(
        AAssetManager_open(manager, "scenes/catalog.json", AASSET_MODE_BUFFER), AAsset_close);
    Require(asset != nullptr, "Packaged scene catalog is missing");
    const auto size = AAsset_getLength64(asset.get());
    Require(size > 0 && size <= 65536, "Scene catalog size exceeds its bound");
    const char* begin = static_cast<const char*>(AAsset_getBuffer(asset.get()));
    Require(begin != nullptr, "Scene catalog cannot be read");
    Json::CharReaderBuilder builder;
    Json::Value root;
    std::string error;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    Require(reader->parse(begin, begin + size, &root, &error), "Scene catalog JSON failed: " + error);
    Require(root["version"].asUInt() == 1 && root["scenes"].isArray() && root["scenes"].size() <= 64,
            "Scene catalog schema/count is invalid");
    const auto hashValid = [](const std::string& hash) {
        return hash.size() == 64 && hash.find_first_not_of("0123456789abcdef") == std::string::npos;
    };
    std::vector<SceneDescriptor> result;
    std::set<std::string> slugs;
    for (const auto& entry : root["scenes"]) {
        SceneDescriptor scene;
        scene.slug=entry["slug"].asString(); scene.label=entry["label"].asString();
        scene.assetPath=entry["asset"].asString(); scene.assetSha256=entry["asset_sha256"].asString();
        scene.teacherFingerprint=entry["teacher_fingerprint"].asString();
        scene.studentCheckpointSha256=entry["student_checkpoint_sha256"].asString();
        scene.gaussianCount=entry["N"].asUInt(); scene.anchorCount=entry["A"].asUInt();
        Require(entry["view_position"].size()==3 && entry["view_rotation"].size()==9,"Scene viewer pose missing");
        for(uint32_t i=0;i<3;++i)scene.viewPosition[i]=entry["view_position"][i].asFloat();
        for(uint32_t i=0;i<9;++i)scene.viewRotation[i]=entry["view_rotation"][i].asFloat();
        Require(!scene.slug.empty() && slugs.insert(scene.slug).second, "Duplicate/empty catalog scene slug");
        Require(!scene.label.empty() && scene.label.size() <= 44, "Scene label length exceeds menu limit");
        for (unsigned char c : scene.label) Require(c >= 32 && c <= 126, "Scene labels must use displayable ASCII");
        Require(scene.assetPath.rfind("scenes/", 0) == 0 && scene.assetPath.find("..") == std::string::npos &&
                    scene.assetPath.find('\\') == std::string::npos,
                "Invalid scene asset path");
        Require(hashValid(scene.assetSha256) && hashValid(scene.teacherFingerprint) &&
                    hashValid(scene.studentCheckpointSha256), "Scene catalog is missing a valid model binding");
        Require(scene.gaussianCount > 0 && scene.anchorCount > 0, "Invalid catalog Gaussian/Anchor count");
        // Only show packaged scene files. External files cannot bypass the signed catalog binding.
        std::unique_ptr<AAsset, decltype(&AAsset_close)> check(
            AAssetManager_open(manager, scene.assetPath.c_str(), AASSET_MODE_STREAMING), AAsset_close);
        Require(check != nullptr, "Catalog asset missing: " + scene.assetPath);
        result.push_back(std::move(scene));
    }
    return result;
}

void SceneAsset::Load(AAssetManager* manager, const SceneDescriptor& expected) {
    Require(manager != nullptr, "Android AssetManager is null");
    std::unique_ptr<AAsset, decltype(&AAsset_close)> asset(
        AAssetManager_open(manager, expected.assetPath.c_str(), AASSET_MODE_STREAMING), AAsset_close);
    Require(asset != nullptr, "Scene asset not found: " + expected.assetPath);
    const off64_t length = AAsset_getLength64(asset.get());
    Require(length >= 16 && uint64_t(length) <= kMaximumSceneBytes, "Scene asset size is invalid or exceeds bound");
    bytes_.assign(size_t(length), 0);
    size_t read = 0;
    while (read < bytes_.size()) {
        const int count = AAsset_read(asset.get(), bytes_.data() + read, bytes_.size() - read);
        Require(count > 0, "Scene asset read ended early");
        read += size_t(count);
    }
    Validate(expected);
}

void SceneAsset::LoadFile(const std::string& path, const SceneDescriptor& expected) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    Require(bool(stream), "Scene file cannot be opened: " + path);
    const auto size = stream.tellg();
    Require(size >= 16 && uint64_t(size) <= kMaximumSceneBytes, "Scene file size is invalid");
    bytes_.resize(size_t(size));
    stream.seekg(0); stream.read(reinterpret_cast<char*>(bytes_.data()), size);
    Require(bool(stream), "Scene file read failed");
    Validate(expected);
}

void SceneAsset::Validate(const SceneDescriptor& expected) {
    Require(Sha256Hex(bytes_.data(), bytes_.size()) == expected.assetSha256,
            "Scene SHA256 does not match the audited checkpoint export");
    const uint8_t magic[8] = {'M', 'C', 'G', 'S', 'Q', '3', 0, 0};
    Require(std::memcmp(bytes_.data(), magic, sizeof(magic)) == 0, "Scene magic mismatch");
    Require(ReadU32(bytes_.data() + 8) == 1, "Unsupported scene format version");
    const uint32_t headerBytes = ReadU32(bytes_.data() + 12);
    Require(headerBytes > 0 && uint64_t(headerBytes) + 16 <= bytes_.size(), "Scene JSON header is out of bounds");

    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;
    Json::Value root;
    std::string errors;
    const char* begin = reinterpret_cast<const char*>(bytes_.data() + 16);
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    Require(reader->parse(begin, begin + headerBytes, &root, &errors), "Scene JSON parse failed: " + errors);
    Require(root["format"].asString() == "mycartgs.quest.scene" && root["version"].asUInt() == 1,
            "Scene schema mismatch");

    identity_.slug = root["slug"].asString();
    identity_.sceneId = root["scene_id"].asString();
    identity_.teacherFingerprint = root["teacher_fingerprint"].asString();
    identity_.studentCheckpointSha256 = root["student_checkpoint_sha256"].asString();
    identity_.studentSchema = root["student_schema"].asString();
    identity_.gaussianCount = root["N"].asUInt();
    identity_.anchorCount = root["A"].asUInt();
    identity_.studentParameters = root["student_parameters"].asUInt();
    identity_.selectorParameters = root["selector_parameters"].asUInt();
    identity_.viewSceneScale=root["selector_view_scene_scale"].asFloat();
    identity_.viewSchema=root["joint_view_schema"].asUInt();
    identity_.softplusScale=root["softplus_scale"].asBool();
    Require(root["compensation_values"].size()==6 && root["ratio_bounds"].size()==2,"Residual48 ratio metadata missing");
    for(uint32_t i=0;i<6;++i)identity_.ratioKnots[i]=root["compensation_values"][i].asFloat();
    for(uint32_t i=0;i<2;++i)identity_.ratioBounds[i]=root["ratio_bounds"][i].asFloat();
    Require(identity_.slug == expected.slug, "Scene slug does not match the selected catalog entry");
    Require(identity_.teacherFingerprint == expected.teacherFingerprint, "Teacher/map fingerprint mismatch");
    Require(identity_.studentCheckpointSha256 == expected.studentCheckpointSha256, "Student checkpoint mismatch");
    Require(identity_.gaussianCount == expected.gaussianCount && identity_.anchorCount == expected.anchorCount,
            "Scene Gaussian/Anchor counts do not match the catalog binding");
    Require(identity_.studentSchema == "residual48_student32_direct_heads_v1", "Student is not the audited 32/32 model");
    Require(identity_.studentParameters == 3534 && identity_.selectorParameters == 1762,
            "Network parameter count mismatch; refusing a possible 64/64 teacher");

    const size_t payloadBase = (16ull + headerBytes + 63ull) & ~63ull;
    const Json::Value& sectionRoot = root["sections"];
    Require(sectionRoot.isObject(), "Scene sections are missing");
    sections_.clear();
    for (const std::string& name : sectionRoot.getMemberNames()) {
        const Json::Value& value = sectionRoot[name];
        SceneSection section;
        const uint64_t relativeOffset = value["offset"].asUInt64();
        section.bytes = size_t(value["bytes"].asUInt64());
        section.offset = payloadBase + size_t(relativeOffset);
        section.dtype = value["dtype"].asString();
        section.sha256 = value["sha256"].asString();
        for (const Json::Value& dimension : value["shape"]) section.shape.push_back(dimension.asUInt());
        Require(section.offset <= bytes_.size() && section.bytes <= bytes_.size() - section.offset,
                "Scene section is out of bounds: " + name);
        section.data = bytes_.data() + section.offset;
        Require(Sha256Hex(section.data, section.bytes) == section.sha256, "Scene section SHA256 mismatch: " + name);
        sections_.emplace(name, std::move(section));
    }

    const uint32_t n = identity_.gaussianCount;
    const uint32_t a = identity_.anchorCount;
    RequireShape(sections_, "uids", {n});
    RequireShape(sections_, "owners", {n});
    RequireShape(sections_, "eligible", {n});
    RequireShape(sections_, "xyz", {n, 3});
    RequireShape(sections_, "selector_static10", {n, 10});
    RequireShape(sections_, "offset3", {n, 3});
    RequireShape(sections_, "anchor_centers", {a, 3});
    RequireShape(sections_, "student_w0_fused", {32, 62});
    RequireShape(sections_, "student_w1", {32, 32});
    RequireShape(sections_, "student_w2", {14, 32});
    RequireShape(sections_, "selector_w0", {32, 19});
    RequireShape(sections_, "selector_w1", {32, 32});
    RequireShape(sections_, "selector_w2", {2, 32});
    for (uint32_t level = 0; level < 6; ++level) RequireShape(sections_, ("residual48_" + std::to_string(level)).c_str(), {a, 48});

    const int32_t* owners = reinterpret_cast<const int32_t*>(Section("owners").data);
    const int64_t* uids = reinterpret_cast<const int64_t*>(Section("uids").data);
    std::set<int64_t> uniqueUids;
    for (uint32_t row = 0; row < n; ++row) {
        Require(owners[row] >= 0 && uint32_t(owners[row]) < a, "Gaussian owner is outside Anchor table");
        Require(uniqueUids.insert(uids[row]).second, "Duplicate Gaussian UID in scene asset");
    }
}

const SceneSection& SceneAsset::Section(const std::string& name) const {
    auto item = sections_.find(name);
    if (item == sections_.end()) throw std::out_of_range("Unknown scene section: " + name);
    return item->second;
}

}  // namespace mycartgs
