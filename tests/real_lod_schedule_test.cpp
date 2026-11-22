#include "include/real_lod_schedule.h"
#include <array>
#include <iostream>
#include <stdexcept>

int main() {
  auto check=[](bool ok){if(!ok)throw std::runtime_error("RealLoD schedule regression");};
  RealLodSchedule s;s.enabled=true;
  for(int i=0;i<7;++i)s.completeFull();
  s.endOnline();s.endOnline();
  for(int i=0;i<3;++i)s.completeFull();
  s.beginLod();
  std::array<int,4> counts{};
  while(s.optimizingLod()){++counts[s.ratioIndex(4)];s.completeLod();}
  check(s.online_steps==7&&s.tail_steps==3&&s.target_steps==7&&s.lod_steps==7);
  check(counts==std::array<int,4>{2,2,2,1});
  bool rejected=false;try{s.completeFull();}catch(const std::logic_error&){rejected=true;}check(rejected);
  rejected=false;try{s.beginLod();}catch(const std::logic_error&){rejected=true;}check(rejected);
  RealLodSchedule empty;empty.enabled=true;empty.endOnline();empty.beginLod();
  check(empty.phase==RealLodSchedule::Phase::Complete&&!empty.optimizingLod());
  std::cout<<"PASS RealLoD: separate online/tail counts, online-sized LoD, no Full slots, zero/invalid boundaries\n";
}
