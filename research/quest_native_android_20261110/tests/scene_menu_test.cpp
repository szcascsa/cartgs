#include "scene_menu.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace mycartgs;
namespace {
void Check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
UiController AimAt(const SceneMenu& menu, const MenuButton& button) {
    UiController input;
    input.tracked = true;
    input.aim.position = {menu.PanelPose().position[0]+button.x+button.width*.5f,
                          menu.PanelPose().position[1]+button.y+button.height*.5f,
                          menu.PanelPose().position[2]+.6f};
    return input;
}
MenuButton Find(const SceneMenu& menu, MenuCommand command, uint32_t value=0) {
    for (const auto& button:menu.Buttons())
        if (button.action.command==command && button.action.value==value) return button;
    throw std::runtime_error("Requested button not found");
}
void Dump(const SceneMenu& menu, const std::string& directory, const char* name) {
    if (directory.empty()) return;
    std::ofstream out(directory+"/"+name, std::ios::binary);
    const auto& vertices=menu.Vertices();
    out.write(reinterpret_cast<const char*>(vertices.data()), std::streamsize(vertices.size()*sizeof(MenuVertex)));
    Check(bool(out), "Unable to write menu preview mesh");
}
}

int main(int argc,char** argv) {
    SceneMenu menu;
    MenuContent content;
    content.scenes.push_back({"REPLICA / OFFICE2 / RGB-D",43197,16723});
    UiPose head;
    std::array<UiController,2> controllers{};
    menu.Update(head,controllers,content,true);
    menu.Build(content);
    Check(menu.Vertices().size()>1000 && menu.Vertices().size()<kMenuVertexCapacity,"Menu mesh bounds failed");
    const std::string preview=argc>1?argv[1]:"";
    Dump(menu,preview,"catalog.mesh");
    Check(Find(menu,MenuCommand::PointBenchmark).enabled,"Catalog benchmark button unavailable");
    auto button=Find(menu,MenuCommand::Load);
    controllers[1]=AimAt(menu,button);
    Check(menu.Update(head,controllers,content,true).command==MenuCommand::None,"Hover activated a button");
    controllers[1].trigger=1;
    Check(menu.Update(head,controllers,content,true).command==MenuCommand::Load,"Right Trigger did not load");
    Check(menu.Update(head,controllers,content,true).command==MenuCommand::None,"Held Trigger repeated a click");
    content.page=MenuPage::Loading;
    menu.Update(head,controllers,content,true);
    menu.Build(content);
    Check(menu.Buttons().empty(),"Loading page accepts duplicate scene loads");
    Dump(menu,preview,"loading.mesh");
    content.page=MenuPage::Details;
    content.loadedLabel="REPLICA / OFFICE2 / RGB-D";
    content.loadedGaussians=43197; content.loadedAnchors=16723;
    content.message="Required QAIRT library unavailable: libQnnSystem.so";
    menu.Update(head,controllers,content,true);
    menu.Build(content);
    Dump(menu,preview,"details.mesh");
    Check(!Find(menu,MenuCommand::RunLod).enabled,"Unavailable CPU backend allowed LoD");
    content.inferenceReady=true;menu.Build(content);
    Check(Find(menu,MenuCommand::RunLod).enabled,"Ready XNNPACK backend blocked LoD");
    Check(Find(menu,MenuCommand::PointBenchmark).enabled,"Details benchmark button unavailable");
    button=Find(menu,MenuCommand::Ratio,50);
    controllers[1]=AimAt(menu,button);
    controllers[1].trigger=1;
    Check(menu.Update(head,controllers,content,true).command==MenuCommand::None,"Held Trigger clicked after page change");
    controllers[1].trigger=0; menu.Update(head,controllers,content,true);
    controllers[1].trigger=1;
    const auto ratio=menu.Update(head,controllers,content,true);
    Check(ratio.command==MenuCommand::Ratio && ratio.value==50,"Ratio button failed");

    menu.Update(head,controllers,content,false);
    Check(menu.Update(head,controllers,content,true).command==MenuCommand::None,"Focus regain generated a click");
    controllers[1].tracked=false;
    menu.Update(head,controllers,content,true);
    controllers[1].tracked=true;
    Check(menu.Update(head,controllers,content,true).command==MenuCommand::None,"Tracking regain generated a click");
    controllers[1]={};
    controllers[0]=AimAt(menu,Find(menu,MenuCommand::FixedK,3000));
    menu.Update(head,controllers,content,true);
    controllers[0].trigger=1;
    const auto fixed=menu.Update(head,controllers,content,true);
    Check(fixed.command==MenuCommand::FixedK && fixed.value==3000,"Left Trigger / fixed K failed");

    content.page=MenuPage::PointBenchmark;content.benchmarkRunning=true;
    content.message="ASCENDING THEN DESCENDING K. MASTER SUBSETS, NOT LOD FPS.";
    content.benchmarkProgress="ROUND 1/2 CASE 1/18 K 0 SAMPLE 1/18";
    for(int i=0;i<9;++i)content.benchmarkRows.push_back("43197    9622      30.123    33.345    1.00X");
    content.benchmarkExport="CSV SAVED: pointcount_1234567890123.csv";
    menu.Update(head,{},content,true);menu.Build(content);Dump(menu,preview,"benchmark.mesh");
    Check(!Find(menu,MenuCommand::PointBenchmark).enabled,"Running test allowed duplicate benchmark");
    Check(menu.Vertices().size()<kMenuVertexCapacity,"Benchmark menu overflow");
    controllers={};controllers[0]=AimAt(menu,Find(menu,MenuCommand::Back));menu.Update(head,controllers,content,true);
    controllers[0].trigger=1;Check(menu.Update(head,controllers,content,true).command==MenuCommand::Back,"Benchmark cancel button failed");
    content.benchmarkRunning=false;menu.Build(content);
    Check(Find(menu,MenuCommand::PointBenchmark).enabled,"Completed benchmark cannot rerun");
    controllers[0]=AimAt(menu,Find(menu,MenuCommand::PointBenchmark));controllers[0].trigger=0;
    menu.Update(head,controllers,content,true);controllers[0].trigger=1;
    Check(menu.Update(head,controllers,content,true).command==MenuCommand::PointBenchmark,"Benchmark rerun trigger failed");

    float x=0,y=0,distance=0;
    UiPose panel; panel.position={0,0,-1};
    Check(SceneMenu::Intersect(panel,head,x,y,distance) && std::abs(distance-1)<1e-6f,"Center ray miss");
    UiPose parallel=head; parallel.orientation={0,.70710678f,0,.70710678f};
    Check(!SceneMenu::Intersect(panel,parallel,x,y,distance),"Parallel ray hit");
    UiPose behind=head; behind.position[2]=-2;
    Check(!SceneMenu::Intersect(panel,behind,x,y,distance),"Behind panel ray hit");
    UiPose invalid=head; invalid.position[0]=NAN;
    Check(!SceneMenu::Intersect(panel,invalid,x,y,distance),"NaN ray hit");
    UiPose rotated; rotated.position={-1,0,0}; rotated.orientation={0,.70710678f,0,.70710678f};
    UiPose rotatedAim=head; rotatedAim.orientation=rotated.orientation;
    Check(SceneMenu::Intersect(rotated,rotatedAim,x,y,distance) && std::abs(distance-1)<1e-5f,"Rotated panel hit failed");

    content.page=MenuPage::Catalog;
    content.scenes.clear();
    menu.Update(head,{},content,true); menu.Build(content);
    Check(!Find(menu,MenuCommand::Load).enabled,"Empty catalog allowed loading");
    Check(!Find(menu,MenuCommand::PointBenchmark).enabled,"Empty catalog allowed benchmarking");
    for (int i=0;i<64;++i) content.scenes.push_back({"LONG / SCENE / LABEL / TEST "+std::to_string(i),100,50});
    for (uint32_t page=0;page<16;++page) {
        content.listPage=page;
        menu.Update(head,{},content,true); menu.Build(content);
        Check(menu.Vertices().size()<kMenuVertexCapacity,"64-scene menu exceeded its vertex bound");
        Check(Find(menu,MenuCommand::Next).enabled==(page<15),"Pagination end failed");
        Check(Find(menu,MenuCommand::Previous).enabled==(page>0),"Pagination start failed");
    }
    menu.Recenter();
    head.position={3,2,1};
    menu.Update(head,{},content,true);
    Check(std::abs(menu.PanelPose().position[0]-3)<1e-6f,"Menu recenter failed");
    std::cout << "scene_menu_test: PASS (ray, stereo snapshot, both triggers, debounce, focus, tracking, pages, bounds)\n";
}
