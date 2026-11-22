#include "scene_menu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace mycartgs {
namespace {
const UiVector kText{{.85f, .91f, 1.f}}, kMuted{{.39f, .51f, .66f}}, kAccent{{.12f, .8f, .65f}};
UiVector Add(const UiVector& a, const UiVector& b) { return {a[0]+b[0], a[1]+b[1], a[2]+b[2]}; }
UiVector Sub(const UiVector& a, const UiVector& b) { return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
UiVector Mul(const UiVector& a, float s) { return {a[0]*s, a[1]*s, a[2]*s}; }
float Dot(const UiVector& a, const UiVector& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
UiVector Cross(const UiVector& a, const UiVector& b) {
    return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}
UiVector Rotate(const std::array<float, 4>& q, const UiVector& v) {
    const UiVector axis{{q[0],q[1],q[2]}};
    return Add(v, Mul(Add(Mul(Cross(axis,v),q[3]), Cross(axis,Cross(axis,v))),2.f));
}
UiVector World(const UiPose& pose, const UiVector& v) { return Add(pose.position, Rotate(pose.orientation,v)); }
bool Valid(const UiPose& pose) {
    for (float x:pose.position) if (!std::isfinite(x)) return false;
    float norm=0;
    for (float x:pose.orientation) { if (!std::isfinite(x)) return false; norm+=x*x; }
    return std::abs(norm-1.f)<.01f;
}

// Original 5x7 glyph rows. Rectangular row runs are batched into one Vulkan draw per eye.
std::array<uint8_t, 7> Glyph(char value) {
    if (value>='a' && value<='z') value=char(value-'a'+'A');
    switch(value) {
        case 'A':return {14,17,17,31,17,17,17}; case 'B':return {30,17,17,30,17,17,30};
        case 'C':return {14,17,16,16,16,17,14}; case 'D':return {30,17,17,17,17,17,30};
        case 'E':return {31,16,16,30,16,16,31}; case 'F':return {31,16,16,30,16,16,16};
        case 'G':return {14,17,16,23,17,17,15}; case 'H':return {17,17,17,31,17,17,17};
        case 'I':return {31,4,4,4,4,4,31}; case 'J':return {7,2,2,2,18,18,12};
        case 'K':return {17,18,20,24,20,18,17}; case 'L':return {16,16,16,16,16,16,31};
        case 'M':return {17,27,21,21,17,17,17}; case 'N':return {17,25,25,21,19,19,17};
        case 'O':return {14,17,17,17,17,17,14}; case 'P':return {30,17,17,30,16,16,16};
        case 'Q':return {14,17,17,17,21,18,13}; case 'R':return {30,17,17,30,20,18,17};
        case 'S':return {15,16,16,14,1,1,30}; case 'T':return {31,4,4,4,4,4,4};
        case 'U':return {17,17,17,17,17,17,14}; case 'V':return {17,17,17,17,17,10,4};
        case 'W':return {17,17,17,21,21,21,10}; case 'X':return {17,17,10,4,10,17,17};
        case 'Y':return {17,17,10,4,4,4,4}; case 'Z':return {31,1,2,4,8,16,31};
        case '0':return {14,17,19,21,25,17,14}; case '1':return {4,12,4,4,4,4,14};
        case '2':return {14,17,1,2,4,8,31}; case '3':return {30,1,1,14,1,1,30};
        case '4':return {2,6,10,18,31,2,2}; case '5':return {31,16,16,30,1,1,30};
        case '6':return {14,16,16,30,17,17,14}; case '7':return {31,1,2,4,8,8,8};
        case '8':return {14,17,17,14,17,17,14}; case '9':return {14,17,17,15,1,1,14};
        case ':':return {0,4,4,0,4,4,0}; case '.':return {0,0,0,0,0,6,6};
        case ',':return {0,0,0,0,6,6,4}; case '-':return {0,0,0,31,0,0,0};
        case '/':return {1,1,2,4,8,16,16}; case '_':return {0,0,0,0,0,0,31};
        case '(':return {2,4,8,8,8,4,2}; case ')':return {8,4,2,2,2,4,8};
        case '%':return {25,25,2,4,8,19,19}; case '=':return {0,31,0,31,0,0,0};
        case '+':return {0,4,4,31,4,4,0}; case '<':return {1,2,4,8,4,2,1};
        case '>':return {16,8,4,2,4,8,16}; case '[':return {14,8,8,8,8,8,14};
        case ']':return {14,2,2,2,2,2,14}; case ' ':return {};
        default:return {14,17,1,2,4,0,4};
    }
}
}  // namespace

bool SceneMenu::Intersect(const UiPose& panel, const UiPose& aim, float& x, float& y, float& distance) {
    if (!Valid(panel) || !Valid(aim)) return false;
    const auto& q=panel.orientation;
    const std::array<float,4> inverse{{-q[0],-q[1],-q[2],q[3]}};
    const UiVector origin=Rotate(inverse,Sub(aim.position,panel.position));
    const UiVector direction=Rotate(inverse,Rotate(aim.orientation,{0,0,-1}));
    if (direction[2]>=-1e-5f) return false;  // No hit from the back, or for a parallel ray.
    distance=-origin[2]/direction[2];
    if (distance<0 || distance>5.f) return false;
    x=origin[0]+distance*direction[0]; y=origin[1]+distance*direction[1];
    return std::abs(x)<=.65f && std::abs(y)<=.5f;
}

void SceneMenu::Layout(const MenuContent& content) {
    buttons_.clear();
    if (content.page==MenuPage::Catalog) {
        const uint32_t first=content.listPage*kScenesPerPage;
        for (uint32_t i=first;i<std::min<uint32_t>(uint32_t(content.scenes.size()),first+kScenesPerPage);++i) {
            buttons_.push_back({-.59f,.25f-.091f*float(i-first),1.18f,.077f,content.scenes[i].label,
                                {MenuCommand::Select,i},true,content.selected==i});
        }
        buttons_.push_back({-.59f,-.2f,.18f,.06f,"PREV",{MenuCommand::Previous,0},content.listPage>0,false});
        buttons_.push_back({-.39f,-.2f,.18f,.06f,"NEXT",{MenuCommand::Next,0},
                            first+kScenesPerPage<content.scenes.size(),false});
        buttons_.push_back({.12f,-.2f,.47f,.06f,"LOAD SCENE",{MenuCommand::Load,0},
                            content.selected<content.scenes.size(),false});
        buttons_.push_back({-.59f,-.445f,1.18f,.065f,"VULKAN POINT COUNT TEST",{MenuCommand::PointBenchmark,0},
                            content.selected<content.scenes.size(),false});
    } else if (content.page==MenuPage::Details || content.page==MenuPage::Error) {
        buttons_.push_back({-.59f,-.28f,.34f,.07f,"SCENE LIST",{MenuCommand::Back,0},true,false});
        if (content.page==MenuPage::Details) {
            buttons_.push_back({-.23f,-.28f,.4f,.07f,"FULL PREVIEW",{MenuCommand::Preview,0},true,false});
            buttons_.push_back({.19f,-.28f,.4f,.07f,"RUN CPU LOD",{MenuCommand::RunLod,0},content.inferenceReady,false});
            buttons_.push_back({-.59f,-.455f,1.18f,.055f,"VULKAN POINT COUNT TEST",{MenuCommand::PointBenchmark,0},true,false});
            for (uint32_t i=1;i<=5;++i) {
                buttons_.push_back({-.59f+.24f*float(i-1),-.055f,.22f,.06f,std::to_string(i*10)+"%",
                                    {MenuCommand::Ratio,i*10},true,!content.fixedBudget && content.ratioPercent==i*10});
            }
            const std::array<uint32_t,6> budgets{{1000,2000,3000,5000,10000,15000}};
            for (uint32_t i=0;i<budgets.size();++i) {
                buttons_.push_back({-.59f+.2f*float(i),-.163f,.18f,.06f,
                                    std::to_string(budgets[i]/1000)+"K",{MenuCommand::FixedK,budgets[i]},true,
                                    content.fixedBudget && content.budget==budgets[i]});
            }
        }
    } else if(content.page==MenuPage::PointBenchmark) {
        buttons_.push_back({-.59f,-.445f,.56f,.065f,content.benchmarkRunning?"CANCEL / BACK":"BACK",
                            {MenuCommand::Back,0},true,false});
        buttons_.push_back({.03f,-.445f,.56f,.065f,"RUN AGAIN",{MenuCommand::PointBenchmark,0},!content.benchmarkRunning,false});
    }
}

MenuAction SceneMenu::Update(const UiPose& head,const std::array<UiController,2>& controllers,
                             const MenuContent& content,bool focused) {
    head_=head; controllers_=controllers;
    if (!placed_ && Valid(head)) {
        const UiVector forward=Rotate(head.orientation,{0,0,-1});
        const float yaw=std::atan2(-forward[0],-forward[2]);
        panel_.orientation={0,std::sin(yaw/2),0,std::cos(yaw/2)};
        panel_.position=Add(head.position,Rotate(panel_.orientation,{0,-.04f,-1.45f}));
        placed_=true;
    }
    Layout(content);
    MenuAction action;
    for (uint32_t hand=0;hand<2;++hand) {
        hovered_[hand]=-1;
        auto& controller=controllers_[hand];
        controller.tracked=focused && controller.tracked && Valid(controller.aim) && std::isfinite(controller.trigger);
        hits_[hand]=controller.tracked && Intersect(panel_,controller.aim,hitX_[hand],hitY_[hand],hitDistance_[hand]);
        if (hits_[hand]) {
            for (size_t i=0;i<buttons_.size();++i) {
                const auto& b=buttons_[i];
                if (b.enabled && hitX_[hand]>=b.x && hitX_[hand]<=b.x+b.width &&
                    hitY_[hand]>=b.y && hitY_[hand]<=b.y+b.height) hovered_[hand]=int(i);
            }
        }
        if (!controller.tracked) { armed_[hand]=false; continue; }
        if (controller.trigger<.2f) armed_[hand]=true;
        if (controller.trigger>.8f) {
            if (armed_[hand] && hovered_[hand]>=0) action=buttons_[size_t(hovered_[hand])].action;
            armed_[hand]=false; // A held trigger cannot activate a different button after navigation.
        }
    }
    return action;
}

void SceneMenu::Rect(float x,float y,float width,float height,float z,const UiVector& color) {
    if (vertices_.size()+6>kMenuVertexCapacity) throw std::runtime_error("Scene menu vertex capacity exceeded");
    const std::array<UiVector,4> p{{World(panel_,{x,y,z}),World(panel_,{x+width,y,z}),
                                   World(panel_,{x+width,y+height,z}),World(panel_,{x,y+height,z})}};
    for (uint32_t index:{0u,1u,2u,0u,2u,3u}) vertices_.push_back({p[index],color});
}

void SceneMenu::Text(float x,float y,const std::string& text,float pixel,const UiVector& color) {
    float column=x;
    for (char character:text) {
        if (character=='\n') { y-=9*pixel; column=x; continue; }
        if (column+5*pixel>.6f) { y-=9*pixel; column=x; }
        if (y<-.465f) break;
        const auto rows=Glyph(character);
        for (uint32_t row=0;row<7;++row) {
            uint32_t c=0;
            while (c<5) {
                if (!(rows[row]&(1u<<(4-c)))) { ++c; continue; }
                const uint32_t start=c;
                while (c<5 && (rows[row]&(1u<<(4-c)))) ++c;
                Rect(column+float(start)*pixel,y-float(row+1)*pixel,float(c-start)*pixel,pixel,.006f,color);
            }
        }
        column+=6*pixel;
    }
}

void SceneMenu::Beam(const UiVector& start,const UiVector& end,const UiVector& color) {
    const UiVector direction=Sub(end,start);
    UiVector side=Cross(direction,Sub(head_.position,start));
    float norm=std::sqrt(Dot(side,side));
    side=norm>1e-5f?Mul(side,.0014f/norm):UiVector{{.0014f,0,0}};
    const std::array<UiVector,4> p{{Sub(start,side),Add(start,side),Add(end,side),Sub(end,side)}};
    if (vertices_.size()+6>kMenuVertexCapacity) throw std::runtime_error("Scene menu beam capacity exceeded");
    for (uint32_t i:{0u,1u,2u,0u,2u,3u}) vertices_.push_back({p[i],color});
}

void SceneMenu::Build(const MenuContent& content) {
    Layout(content);
    vertices_.clear();
    vertices_.reserve(kMenuVertexCapacity);
    Rect(-.65f,-.5f,1.3f,1.f,0,{.014f,.025f,.047f});
    Rect(-.65f,.485f,1.3f,.015f,.001f,kAccent);
    Text(-.59f,.44f,"MYCARTGS  /  SCENES",.0045f,kText);
    if (content.page==MenuPage::Catalog) {
        Text(-.59f,.355f,"POINT EITHER CONTROLLER. PRESS INDEX TRIGGER TO SELECT.",.0032f,kMuted);
        if (content.scenes.empty()) Text(-.59f,.24f,"NO PACKAGED SCENES AVAILABLE",.0037f,{1,.45f,.27f});
        Text(-.59f,-.265f,"PACKAGED SCENES: "+std::to_string(content.scenes.size())+"   PAGE "+
             std::to_string(content.listPage+1),.0032f,kMuted);
        Text(-.59f,-.31f,"SELECT A SCENE, THEN PRESS LOAD SCENE.",.0033f,kText);
        if (!content.message.empty()) Text(-.59f,-.352f,content.message,.003f,kMuted);
    } else if (content.page==MenuPage::Loading) {
        Text(-.59f,.3f,"LOADING SCENE...",.0045f,kAccent);
        Text(-.59f,.21f,content.loadedLabel,.0038f,kText);
        Text(-.59f,.11f,"CHECKING MAP, UIDS, SELECTOR AND STUDENT32.",.0036f,kText);
        Text(-.59f,.02f,"THE MENU REMAINS TRACKED WHILE ASSETS LOAD.",.0034f,kMuted);
    } else if (content.page==MenuPage::Details) {
        Text(-.59f,.33f,content.loadedLabel,.004f,kText);
        char counts[100];
        std::snprintf(counts,sizeof(counts),"GAUSSIANS: %u   ANCHORS: %u",content.loadedGaussians,content.loadedAnchors);
        Text(-.59f,.265f,counts,.0034f,kMuted);
        Text(-.59f,.21f,"ASSET / UID / STUDENT32 BINDING: VERIFIED",.0034f,kAccent);
        Text(-.59f,.15f,content.inferenceReady?"XNNPACK ARM CPU READY / STUDENT32":"LOD BLOCKED: CPU MODEL NOT READY",.0034f,{1,.5f,.3f});
        Text(-.59f,.105f,"FULL PREVIEW: REAL GAUSSIANS / NO MLP",.0031f,kMuted);
        Text(-.59f,.035f,"RATIO",.003f,kMuted);
        Text(-.59f,-.084f,"FIXED GAUSSIAN BUDGET",.003f,kMuted);
        const std::string budget=content.fixedBudget?std::to_string(content.budget):std::to_string(content.ratioPercent)+"%";
        Text(-.59f,-.195f,"REQUESTED BUDGET: "+budget,.0029f,kMuted);
        Text(-.59f,-.365f,content.timing,.0026f,kMuted);
        Text(-.59f,-.325f,content.message,.0029f,kMuted);
    } else if(content.page==MenuPage::PointBenchmark) {
        Text(-.59f,.34f,"VULKAN POINT COUNT TEST",.004f,kAccent);
        Text(-.59f,.275f,"FROZEN VIEW / MASTER ATTRIBUTES / NO MLP",.003f,kMuted);
        Text(-.59f,.23f,"GPU: OFFSCREEN RASTER ONLY. 2 ROUNDS / 30 SAMPLES PER K",.0027f,kMuted);
        Text(-.59f,.177f,content.benchmarkProgress,.0028f,kText);
        Text(-.59f,.1f,"K        VISIBLE     GPU MS    P95 MS    X FULL",.003f,kAccent);
        for(uint32_t i=0;i<content.benchmarkRows.size();++i)
            Text(-.59f,.058f-.035f*i,content.benchmarkRows[i],.003f,kText);
        Text(-.59f,-.302f,content.message,.0026f,kMuted);
        Text(-.59f,-.35f,content.benchmarkExport,.0024f,kMuted);
    } else {
        Text(-.59f,.3f,"SCENE LOAD FAILED",.0043f,{1,.45f,.27f});
        Text(-.59f,.19f,content.message,.0033f,kText);
    }
    for (size_t i=0;i<buttons_.size();++i) {
        const MenuButton& b=buttons_[i];
        // Re-evaluate hover because a scene-state transition can replace the button list.
        bool hovered=false;
        for (uint32_t h=0;h<2;++h) hovered|=hits_[h] && hitX_[h]>=b.x && hitX_[h]<=b.x+b.width &&
                                                     hitY_[h]>=b.y && hitY_[h]<=b.y+b.height;
        const UiVector color=!b.enabled?UiVector{{.025f,.04f,.065f}}:hovered?UiVector{{.07f,.28f,.31f}}:
                                 b.selected?UiVector{{.035f,.19f,.18f}}:UiVector{{.04f,.075f,.13f}};
        Rect(b.x,b.y,b.width,b.height,.002f,color);
        if (b.selected) Rect(b.x,b.y,.006f,b.height,.004f,kAccent);
        const float pixel=b.action.command==MenuCommand::Select?.0038f:.0033f;
        Text(b.x+.018f,b.y+b.height*.5f+3.5f*pixel,b.label,pixel,b.enabled?kText:kMuted);
    }
    Text(-.59f,-.463f,"L "+std::string(controllers_[0].tracked?"TRACKED":"---")+"  R "+
         (controllers_[1].tracked?"TRACKED":"---")+" / XNNPACK CPU / VULKAN / 0.4.1",.002f,kMuted);
    for (uint32_t hand=0;hand<2;++hand) {
        if (!controllers_[hand].tracked) continue;
        const UiVector end=hits_[hand]?World(panel_,{hitX_[hand],hitY_[hand],.012f}):
            World(controllers_[hand].aim,{0,0,-2.f});
        Beam(controllers_[hand].aim.position,end,kAccent);
        if (hits_[hand]) Rect(hitX_[hand]-.005f,hitY_[hand]-.005f,.01f,.01f,.013f,{.9f,1.f,.9f});
    }
    ++revision_;
}

}  // namespace mycartgs
