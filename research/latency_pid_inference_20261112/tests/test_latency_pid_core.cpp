#include "latency_pid_core.h"
#include <cassert>
#include <iostream>

using latency_pid::Config;
using latency_pid::Core;

int main() {
  Config c;c.ema=1.;c.deadband=.01;
  {Core p(c);p.applied(3000);p.observe(.6,0.,3000,12000);assert(p.budget()<3000);}
  {Core p(c);p.applied(3000);p.observe(.2,0.,3000,12000);assert(p.budget()>3000 && p.budget()<=3240);}
  {Core p(c);p.applied(3000);p.observe(.4,0.,3000,12000);assert(p.budget()==3000 && p.integral==0.);}
  {Core p(c);p.applied(3000);for(int j=0;j<2000;++j){p.observe(.1,j*.01,3000,3000);p.applied(p.budget());}
    assert(std::abs(p.integral)<1e-12);}
  {Core p(c);p.applied(c.min_budget);for(int j=0;j<2000;++j)p.observe(2.,j*.01,c.min_budget,12000);
    assert(p.budget()==c.min_budget && std::abs(p.integral)<1e-12);}
  {Core p(c);p.applied(50);for(int j=0;j<1000;++j)p.observe(.1,j*.01,50,50);
    assert(std::abs(p.integral)<1e-12 && p.budget()==c.min_budget);}
  {Core p(c);p.applied(0);p.observe(.1,0.,0,0);assert(p.updates==0 && p.integral==0.);}
  {Core p(c);p.applied(4500);p.reset();assert(p.budget()==4500 && p.integral==0. && p.derivative==0.);
    p.target(.8);assert(p.budget()==4500 && p.config.target_ms==.8);}
  {Config d=c;d.kd=.01;Core p(d);p.applied(3000);p.observe(.4,0.,3000,12000);
    p.observe(.5,.01,3000,12000);assert(p.derivative>0 && p.budget()<3000);}
  {Core p(c);bool rejected=false;try{p.observe(.4,0.,3000,2000);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);}
  {Core p(c);p.observe(.4,1.,3000,12000);bool rejected=false;try{p.observe(.4,.9,3000,12000);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);}
  {Config d=c;d.target_ms=0;bool rejected=false;try{Core p(d);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);}
  // Closed-loop plant has a fixed cost; convergence must not rely on T proportional to K.
  {Config d=c;d.ema=.25;Core p(d);double ms=0.;
    for(int j=0;j<1000;++j){const auto k=p.budget();p.applied(k);ms=.28+.00004*k;
      p.observe(ms,j/90.,k,20000);}
    assert(std::abs(ms-d.target_ms)<.01);}
  // A new workload, then recovery: integrator must not remain charged at saturation.
  {Core p(c);double ms=0.;
    for(int j=0;j<2000;++j){const auto k=p.budget();p.applied(k);ms=(j<800?.6:.25)+.00004*k;p.observe(ms,j/90.,k,20000);}
    assert(std::abs(ms-c.target_ms)<.01);}
  std::cout<<"PID_CORE_PASS: 14 scenarios\n";
}
