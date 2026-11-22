#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace mycartgs {

using UiVector = std::array<float, 3>;
struct UiPose {
    UiVector position{};
    std::array<float, 4> orientation{{0, 0, 0, 1}};  // OpenXR xyzw
};
struct UiController {
    UiPose aim;
    float trigger = 0;
    bool tracked = false;
};
struct MenuVertex { UiVector position; UiVector color; };
constexpr uint32_t kMenuVertexCapacity = 65536;
constexpr uint32_t kScenesPerPage = 4;

enum class MenuPage { Catalog, Loading, Details, Error, PointBenchmark };
enum class MenuCommand { None, Select, Load, Previous, Next, Back, Ratio, FixedK, Preview, RunLod, PointBenchmark };
struct MenuAction { MenuCommand command = MenuCommand::None; uint32_t value = 0; };
struct MenuScene { std::string label; uint32_t gaussians = 0; uint32_t anchors = 0; };
struct MenuContent {
    MenuPage page = MenuPage::Catalog;
    std::vector<MenuScene> scenes;
    uint32_t selected = 0;
    uint32_t listPage = 0;
    std::string loadedLabel;
    std::string message;
    uint32_t loadedGaussians = 0;
    uint32_t loadedAnchors = 0;
    float loadMs = 0;
    bool inferenceReady = false;
    bool fixedBudget = false;
    uint32_t budget = 0;
    uint32_t ratioPercent = 10;
    std::string timing;
    bool benchmarkRunning = false;
    std::string benchmarkProgress,benchmarkExport;
    std::vector<std::string> benchmarkRows;
};
struct MenuButton {
    float x, y, width, height;
    std::string label;
    MenuAction action;
    bool enabled = true;
    bool selected = false;
};

class SceneMenu {
public:
    // Interaction and mesh use one predicted-display-time input snapshot for both eyes.
    MenuAction Update(const UiPose& head, const std::array<UiController, 2>& controllers,
                      const MenuContent& content, bool focused);
    void Build(const MenuContent& content);
    void Recenter() { placed_ = false; }
    const std::vector<MenuVertex>& Vertices() const { return vertices_; }
    const std::vector<MenuButton>& Buttons() const { return buttons_; }
    const UiPose& PanelPose() const { return panel_; }
    uint64_t Revision() const { return revision_; }
    static bool Intersect(const UiPose& panel, const UiPose& aim, float& x, float& y, float& distance);

private:
    void Layout(const MenuContent& content);
    void Rect(float x, float y, float width, float height, float z, const UiVector& color);
    void Text(float x, float y, const std::string& text, float pixel, const UiVector& color);
    void Beam(const UiVector& start, const UiVector& end, const UiVector& color);
    UiPose panel_;
    UiPose head_;
    std::array<UiController, 2> controllers_{};
    std::array<bool, 2> armed_{{false, false}};
    std::array<int, 2> hovered_{{-1, -1}};
    std::array<float, 2> hitX_{}, hitY_{}, hitDistance_{};
    std::array<bool, 2> hits_{{false, false}};
    bool placed_ = false;
    uint64_t revision_ = 0;
    std::vector<MenuButton> buttons_;
    std::vector<MenuVertex> vertices_;
};

}  // namespace mycartgs
