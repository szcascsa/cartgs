#include "include/ultrafast_rotation.h"
#include <stdexcept>
#include <iostream>

int main() {
  auto check=[](bool condition){if(!condition)throw std::runtime_error("Post-topology schedule invariant failed");};
  UltrafastRotation r;r.post_topology_full_steps=100;
  r.complete(0,0,true,true,false,false);
  const auto phase=r.phase();r.beginRecovery();
  for(int i=0;i<100;++i) {
    check(r.recovering());r.completeRecovery(0,false);++r.recovery_steps;--r.recovery_remaining;
    check(r.phase()==phase && r.balanced());
  }
  check(!r.recovering() && r.full_render_count==100 && r.lod_render_count==1);
  r.complete(0,0,true,true,false,false);check(r.phase()==2);
  r.beginRecovery();--r.recovery_remaining;r.beginRecovery();check(r.recovery_remaining==100);
  r.endOnline(102);check(r.tailRatioIndex()==2);
  for(int i=0;i<100;++i) {r.completeRecovery(0,false);--r.recovery_remaining;}
  check(!r.recovering() && r.tailRatioIndex()==2 && r.balanced());
  r.completeRotatingTail(0,0,true,true,true,false,false);check(r.tailRatioIndex()==3 && r.balanced());
  check(r.full_render_count+r.lod_render_count==r.step && r.tail_recovery_steps==100);
  UltrafastRotation legacy;legacy.beginRecovery();check(!legacy.recovering());
  UltrafastRotation serial;serial.serial_full_lod=true;
  for(int i=0;i<7;++i) {
    check(serial.full() && serial.phase()==i%4 && serial.ratio()==serial.ratios[i%4]);
    serial.completeSerial(0,0,true,true,true,false);check(serial.balanced());
  }
  serial.endOnline(7);
  for(int i=0;i<9;++i) {
    check(serial.tailRatioIndex()==(7+i)%4);
    serial.completeTail(0,true,true,true,true,false);check(serial.balanced());
  }
  check(serial.full_render_count==16 && serial.lod_render_count==16 && serial.step==16);
  std::cout<<"PASS post-topology recovery: 100 next mappings, restart, online/tail clock and balance, disabled path\n";
}
