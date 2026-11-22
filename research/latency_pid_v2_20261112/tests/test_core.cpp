#define latency_pid legacy_latency_pid
#include "latency_pid_v1_core.h"
#undef latency_pid
#include "latency_pid_core.h"
#include <cassert>
#include <iostream>
#include <limits>

using latency_pid::Config;
using latency_pid::Core;

int main(){
  Config c;c.target_ms=.2;c.ema=1.;c.deadband=.05;
  int count=0;
  {Core p(c);p.applied(3000);p.observe(.3,0.,3000,12000);assert(p.budget()<3000);++count;}
  {Core p(c);p.applied(3000);p.observe(.1,0.,3000,12000);assert(p.budget()>3000&&p.budget()<=3240);++count;}
  {Core p(c);p.applied(3000);p.observe(.2,0.,3000,12000);assert(p.budget()==3000);++count;}
  // 死区后轻微超时：旧位置式PI可反向加点；新控制器应减点。
  {legacy_latency_pid::Config d;d.target_ms=.2;d.ema=1.;
    legacy_latency_pid::Core old(d);old.applied(500);old.observe(.01,0.,500,500);
    old.observe(.2,.01,500,12000);old.observe(.212,.02,500,12000);assert(old.budget()>500);
    Core p(c);p.applied(500);p.observe(.01,0.,500,500);
    p.observe(.2,.01,500,12000);p.observe(.212,.02,500,12000);
    assert(p.budget()<500);std::cout<<"LEGACY_WRONG_DIRECTION "<<old.budget()<<" V2 "<<p.budget()<<'\n';++count;}
  {Core p(c);double previous=3000.;for(int j=0;j<1000;++j){auto k=p.budget();
      p.applied(k);p.observe(.8,j/90.,k,20000);assert(p.budget()>=256&&p.budget()>=std::floor(k*.75));previous=k;}
    assert(p.budget()==256);++count;}
  {Core p(c);for(int j=0;j<1000;++j){auto k=p.budget();p.applied(k);p.observe(.01,j/90.,k,20000);}
    assert(p.budget()==8192);++count;}
  {Core p(c);p.applied(50);for(int j=0;j<1000;++j)p.observe(.01,j/90.,50,50);
    assert(p.budget()==256);p.observe(.01,12.,256,12000);assert(p.budget()<=277);++count;}
  {Core p(c);p.applied(0);p.observe(.1,0.,0,0);assert(p.updates==0);++count;}
  {Core p(c);p.applied(4500);p.reset();assert(p.budget()==4500);p.target(.3);assert(p.budget()==4500);++count;}
  for(double fixed:{.02,.08,.15,.25}){
    Config d=c;d.ema=.25;Core p(d);double ms=0.;
    for(int j=0;j<2000;++j){auto k=p.budget();p.applied(k);ms=fixed+.000025*k;p.observe(ms,j/90.,k,20000);}
    if(fixed<.2)assert(std::abs(ms-.2)<=.0101);else assert(p.budget()==256);++count;
  }
  {Config d=c;d.ema=.25;Core p(d);double ms=0.;
    for(int j=0;j<3000;++j){auto k=p.budget();p.applied(k);ms=(j<1000?.35:.08)+.000025*k;p.observe(ms,j/90.,k,20000);}
    assert(std::abs(ms-.2)<=.0101);++count;}
  {Core p(c);p.applied(3000);p.observe(.2,1.,3000,12000);bool rejected=false;
    try{p.observe(.2,.9,3000,12000);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);++count;}
  {Core p(c);bool rejected=false;try{p.observe(std::numeric_limits<double>::infinity(),0.,3000,12000);}
    catch(const std::invalid_argument&){rejected=true;}assert(rejected);++count;}
  {Core p(c);bool rejected=false;try{p.observe(.2,0.,3000,2000);}catch(const std::invalid_argument&){rejected=true;}
    assert(rejected);++count;}
  {Config d=c;d.target_ms=0.;bool rejected=false;try{Core p(d);}catch(const std::invalid_argument&){rejected=true;}
    assert(rejected);++count;}
  std::cout<<"PID_V2_CORE_PASS "<<count<<" scenarios\n";
}
