#include "geometry_ranker.h"
#include <cassert>
#include <random>
using namespace geo_marginal;
Point point(double x,double y,double radius,double w,int64_t id){return {{x,y,0},radius,w,id};}
void parity(const std::vector<Point>&p,double h){auto ref=rank(p,h,p.size(),false,true);auto opt=rank(p,h,p.size());assert(ref.order==opt.order);assert(ref.gains==opt.gains);auto repeat=rank(p,h,p.size());assert(opt.order==repeat.order);for(size_t n:{size_t(1),p.size()/2,p.size()}){auto z=rank(p,h,n);assert(std::equal(z.order.begin(),z.order.end(),ref.order.begin()));}}
int main(){
 std::vector<Point>d={point(0,0,1,10,0),point(0,0,1,9,1),point(0,0,1,8,2)};auto z=rank(d,8,3);assert(z.order[0]==0&&z.gains[0]==27&&z.gains[1]==0);parity(d,8);
 auto independent=std::vector<Point>{point(1,0,1e-5,8,0),point(2,0,1e-5,10,1),point(3,0,1e-5,9,2)};z=rank(independent,8,3);assert((z.order==std::vector<int64_t>{1,2,0}));parity(independent,8);
 parity({point(0,0,.1,4,7)},8);
 std::mt19937 gen(2026);std::uniform_real_distribution<double>u(0,1);std::vector<Point>large;for(int i=0;i<256;++i)large.push_back(point(u(gen),u(gen),.05+u(gen)*.1,u(gen),i));parity(large,8);
 for(int i=0;i<256;++i)large[i].x[0]+=i%9;parity(large,.5);
 large.push_back(point(10,10,.1,2,1000));large.erase(large.begin()+5);std::reverse(large.begin(),large.end());parity(large,.5);auto rebuilt=rank(large,.5,large.size());for(auto row:rebuilt.order)assert(row!=5);
}
