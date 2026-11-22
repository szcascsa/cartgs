#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <vector>

// CPU FP64 exact facility-location ranking. No contribution vector or view API.
namespace geo_marginal {
using Clock=std::chrono::steady_clock;
inline double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct Point {double x[3],radius,weight; int64_t id;};
struct CellStats {size_t n=0,evaluations=0,selected=0,pairs=0,hits=0;double similarity_ms=0,init_ms=0,rescore_ms=0,coverage_ms=0,total_ms=0;};
struct Result {std::vector<int64_t> order;std::vector<double> gains;std::vector<int64_t> cell;std::vector<CellStats> stats;double grid_ms=0,membership_ms=0,merge_ms=0,total_ms=0;size_t pair_bytes=0;};
struct Node {double gain;int64_t id;size_t row,version;};
struct Lower {bool operator()(const Node&a,const Node&b)const{return a.gain!=b.gain?a.gain<b.gain:a.id>b.id;}};
struct GlobalNode {Node node;size_t cell;};
struct GlobalLower {bool operator()(const GlobalNode&a,const GlobalNode&b)const{if(a.node.gain!=b.node.gain)return a.node.gain<b.node.gain;if(a.cell!=b.cell)return a.cell>b.cell;return a.node.id>b.node.id;}};
struct Cell {
 const std::vector<Point>* input=nullptr;
 double at(size_t i,size_t j)const{if(i==j)return 1.;const auto&a=(*input)[rows[i]];const auto&b=(*input)[rows[j]];double d=0;for(int z=0;z<3;++z)d+=(a.x[z]-b.x[z])*(a.x[z]-b.x[z]);double s=.5*(a.radius+b.radius);return std::exp(-d/(2*s*s+1e-12));}
 std::vector<size_t> rows;std::vector<double> k,cover;std::vector<bool> taken;std::priority_queue<Node,std::vector<Node>,Lower> heap;size_t version=0;CellStats stat;
 double gain(size_t i,const std::vector<Point>&p){double sum=0;size_t n=rows.size();for(size_t j=0;j<n;++j)sum+=p[rows[j]].weight*std::max(at(i,j)-cover[j],0.);++stat.evaluations;stat.hits+=0;return sum;}
 void initialize(const std::vector<Point>&p){size_t n=rows.size();stat.n=n;auto t=Clock::now();input=&p;cover.assign(n,0);taken.assign(n,false);stat.pairs=0;stat.similarity_ms=ms(t);t=Clock::now();for(size_t i=0;i<n;++i)heap.push({gain(i,p),p[rows[i]].id,i,0});stat.init_ms=ms(t);}
 Node head(const std::vector<Point>&p,bool lazy){auto t=Clock::now();if(!lazy){while(!heap.empty())heap.pop();for(size_t i=0;i<rows.size();++i)if(!taken[i])heap.push({gain(i,p),p[rows[i]].id,i,version});}
   while(!heap.empty()&&heap.top().version!=version){auto v=heap.top();heap.pop();v.gain=gain(v.row,p);v.version=version;heap.push(v);}stat.rescore_ms+=ms(t);return heap.top();}
 void take(const Node&v){heap.pop();taken[v.row]=true;auto t=Clock::now();for(size_t j=0;j<rows.size();++j)cover[j]=std::max(cover[j],at(v.row,j));++version;++stat.selected;stat.coverage_ms+=ms(t);}
};
inline Result rank(const std::vector<Point>&p,double h,size_t limit,bool lazy_local=true,bool full_reference=false){
 if(!(h>0)||!std::isfinite(h))throw std::invalid_argument("Invalid fixed grid h");
 auto start=Clock::now();Result out;std::map<std::array<int64_t,3>,std::vector<size_t>> groups;
 auto t=Clock::now();for(size_t i=0;i<p.size();++i){if(!(p[i].radius>0)||!(p[i].weight>=0)||!std::isfinite(p[i].radius)||!std::isfinite(p[i].weight))throw std::invalid_argument("Invalid geometry marginal candidate");std::array<int64_t,3> key;for(int z=0;z<3;++z){if(!std::isfinite(p[i].x[z]))throw std::invalid_argument("Nonfinite position");key[z]=int64_t(std::floor(p[i].x[z]/h));}groups[key].push_back(i);}out.grid_ms=ms(t);
 t=Clock::now();std::vector<Cell> cells;out.cell.resize(p.size());for(auto&entry:groups){Cell c;c.rows=std::move(entry.second);for(auto i:c.rows)out.cell[i]=cells.size();cells.push_back(std::move(c));}out.membership_ms=ms(t);
 std::priority_queue<GlobalNode,std::vector<GlobalNode>,GlobalLower> global;
 for(size_t ci=0;ci<cells.size();++ci){auto&c=cells[ci];c.initialize(p);out.pair_bytes+=c.k.size()*sizeof(double);if(!full_reference)global.push({c.head(p,true),ci});}
 if(full_reference){std::vector<GlobalNode> all;for(size_t ci=0;ci<cells.size();++ci){auto&c=cells[ci];while(c.version<c.rows.size()){auto head=c.head(p,false);all.push_back({head,ci});c.take(head);}}t=Clock::now();std::sort(all.begin(),all.end(),[](const GlobalNode&a,const GlobalNode&b){return GlobalLower{}(b,a);});for(auto&v:all){out.order.push_back(p[cells[v.cell].rows[v.node.row]].id);out.gains.push_back(v.node.gain);}out.merge_ms=ms(t);
 }else{limit=std::min(limit,p.size());while(out.order.size()<limit){auto head=global.top();global.pop();auto&c=cells[head.cell];out.order.push_back(p[c.rows[head.node.row]].id);out.gains.push_back(head.node.gain);c.take(head.node);if(c.version<c.rows.size()&&out.order.size()<limit)global.push({c.head(p,lazy_local),head.cell});}}
 for(auto&c:cells){c.stat.total_ms=c.stat.similarity_ms+c.stat.init_ms+c.stat.rescore_ms+c.stat.coverage_ms;out.stats.push_back(c.stat);}out.total_ms=ms(start);return out;
}
}
