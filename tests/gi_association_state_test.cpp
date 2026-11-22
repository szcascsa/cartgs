#include "ORB-SLAM3/include/Atlas.h"
#include "ORB-SLAM3/include/GIAssociationState.h"
#include <future>
#include <iostream>
#include <stdexcept>

using namespace ORB_SLAM3;
namespace {
void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
class FixtureKF : public KeyFrame {
 public:
  FixtureKF(uint64_t id, Map* map, KeyFrameDatabase* db) {
    mnId = id; mpMap = map; mpKeyFrameDB = db; mpCamera2 = nullptr;
    mvpMapPoints.resize(4, nullptr);
  }
};
std::vector<uint64_t> authoritative(KeyFrame& frame) {
  std::vector<uint64_t> ids;
  for (auto point : frame.GetMapPoints()) if (!point->isBad()) ids.push_back(point->mnId);
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}
void compare(KeyFrame& frame) {
  GIAssociationState::Read transaction;
  auto ids = frame.GIPointIds();
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  check(ids == authoritative(frame), "GI read differs from ORB authoritative points");
}
}
int main() {
  Map map(0);
  ORBVocabulary vocabulary;
  KeyFrameDatabase db(vocabulary);
  FixtureKF initial(0, &map, &db), frame(1, &map, &db);
  map.AddKeyFrame(&initial); map.AddKeyFrame(&frame);
  MapPoint old_point, replacement;
  old_point.mnId = 10; old_point.UpdateMap(&map);
  replacement.mnId = 11; replacement.UpdateMap(&map);
  uint64_t revision = frame.GIAssociationRevision();
  frame.AddMapPoint(&old_point, 0);
  check(frame.GIAssociationRevision() > revision, "Insertion lost invalidation");
  compare(frame);
  frame.ReplaceMapPointMatch(0, &replacement); compare(frame);
  frame.EraseMapPointMatch(0); compare(frame);
  frame.AddMapPoint(&old_point, 0);
  const auto bad_revision = GIAssociationState::badRevision();
  // Missing reverse observation must not hide this bad point from invalidation.
  old_point.Replace(&replacement);
  check(GIAssociationState::badRevision() > bad_revision, "Replace lost bad invalidation");
  compare(frame);
  frame.AddMapPoint(&replacement, 1);
  replacement.SetBadFlag(); compare(frame);
  revision = frame.GIAssociationRevision();
  frame.EraseMapPointMatch(&replacement); compare(frame);
  check(frame.GIAssociationRevision() > revision, "Pointer erase lost invalidation");
  revision = frame.GIAssociationRevision();
  frame.SetPose(Sophus::SE3f());
  check(frame.GIAssociationRevision() == revision, "Pose-only update invalidated associations");
  frame.SetBadFlag();
  { GIAssociationState::Read read; check(frame.GIBad(), "KF culling not published"); }
  const auto generation = map.GIGeneration();
  map.clear();
  check(map.GIGeneration() != generation, "Reset reused old generation");

  std::future<void> blocked;
  std::promise<void> started;
  {
    GIAssociationState::Read read;
    GIAssociationState::Read nested_read;
    blocked = std::async(std::launch::async, [&] {
      started.set_value();
      GIAssociationState::Write write;
      GIAssociationState::Write nested_write;
      GIAssociationState::invalidateBadPoints();
    });
    started.get_future().wait();
    check(blocked.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
          "Snapshot observed a partially committed writer");
  }
  blocked.get();
  Atlas atlas(0);
  { GIAssociationState::Read read; check(atlas.GetCurrentMapForGI(), "Initial map missing"); }
  atlas.clearAtlas();
  { GIAssociationState::Read read; check(!atlas.GetCurrentMapForGI(), "Reset map still visible"); }
  atlas.CreateNewMap();
  Map* first = atlas.GetCurrentMap();
  atlas.CreateNewMap();
  Map* second = atlas.GetCurrentMap();
  check(first->GIGeneration() != second->GIGeneration(), "Maps share cache generation");
  atlas.ChangeMap(first);
  { GIAssociationState::Read read; check(atlas.GetCurrentMapForGI() == first, "Map switch not atomic"); }
  std::cout << "PASS: ORB mutations, bad/replace, culling/reset, nested transaction exclusion\n";
}
