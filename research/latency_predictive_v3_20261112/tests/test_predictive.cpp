#include "latency_predictive_core.h"
#include "latency_view_summary.h"
#include <cassert>
#include <iostream>

using latency_predictive::Core;
using latency_predictive::View;
int main(){
  int count=0;
  {Core c(.2,3000,256,50000);View v{18000,1.};assert(c.budget(v)==3000);++count;}
  {Core c(.2,3000,256,50000);View v{10000,1.};
    for(int j=0;j<150;++j){auto k=c.budget(v);c.applied(k);c.observe(.08+.000008*k,j/90.,k,10000,v);}
    assert(c.budget(v)==10000);++count;}
  {Core c(.2,3000,256,50000);View v{500,1.};c.observe(.1,0.,500,500,v);
    assert(c.budget(v)==500);assert(c.budget(View{18000,1.})>500);++count;}
  for(double slope:{.000004,.00001,.00003}){
    Core c(.2,3000,256,50000);View v{30000,1.};double ms=0.;
    for(int j=0;j<1500;++j){auto k=c.budget(v);c.applied(k);ms=.08+slope*k;c.observe(ms,j/90.,k,v.available,v);}
    assert(std::abs(ms-.2)<.012);++count;
  }
  {Core c(.2,3000,256,50000);View v{30000,.7};double ms=0.;
    for(int j=0;j<2500;++j){auto k=c.budget(v);ms=(j<1000?.3:.08)+.00001*k;c.applied(k);c.observe(ms,j/90.,k,v.available,v);}
    assert(std::abs(ms-.2)<.012);++count;}
  {Core c(.2,3000,256,50000);c.applied(0);c.observe(.1,0.,0,0,{0,1.});assert(c.updates==0);++count;}
  {Core c(.2,3000,256,50000);bool bad=false;try{c.observe(.2,1.,3000,2000,{2000,1.});}
    catch(const std::invalid_argument&){bad=true;}assert(bad);++count;}
  {Core c(.2,3000,256,50000);c.target(.3);assert(c.target_ms==.3);++count;}
  {Core c(.2,3000,256,50000);double error=0.;
    for(int j=0;j<3000;++j){View v{int64_t(23000+6000*std::sin(j*.009)),.45+.2*std::sin(j*.013)};
      auto k=c.budget(v);const double ms=.07+.000006*k+.03*std::sqrt(k/4096.)*v.footprint;
      const double feasible=std::min(.2,.07+.000006*v.available+.03*std::sqrt(v.available/4096.)*v.footprint);
      c.applied(k);c.observe(ms,j/90.,k,v.available,v);if(j>=500)error+=std::abs(ms-feasible);}
    assert(error/2500.<.015);++count;}
  {latency_predictive::ViewSummary s;std::vector<latency_predictive::Anchor> anchors;
    int expected=0;for(int i=0;i<101;++i){float x=(i-50)*.04f;anchors.push_back({x,0.f,1.f,.01f,2.f,.0001f});
      if(std::abs(x)<=1.01f)expected+=2;}
    std::array<float,16> projection{};projection[0]=projection[5]=1.f;projection[10]=1.0001f;projection[11]=1.f;projection[14]=-.010001f;
    s.configure(anchors,projection,1200,680);std::vector<double> pose(16,0.);pose[0]=pose[5]=pose[10]=pose[15]=1.;
    assert(s.evaluate(pose).available==expected);pose[11]=5.;assert(s.evaluate(pose).available==0);++count;}
  std::cout<<"PREDICTIVE_CORE_PASS "<<count<<" scenarios\n";
}
