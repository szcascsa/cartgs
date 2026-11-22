#include "geometry_ranker.h"
#include <fstream>
#include <iostream>
#include <ctime>
#include <sys/resource.h>
int main(int argc,char**argv){if(argc!=5)return 2;std::ifstream f(argv[1],std::ios::binary);int64_t n;double h;f.read((char*)&n,8);f.read((char*)&h,8);std::vector<geo_marginal::Point>p(n);for(auto&v:p){f.read((char*)v.x,24);f.read((char*)&v.radius,8);f.read((char*)&v.weight,8);f.read((char*)&v.id,8);}if(!f)return 3;
 std::string mode=argv[3];size_t limit=std::stoull(argv[4]);auto cpu=std::clock();auto r=geo_marginal::rank(p,h,limit,mode!="global",mode=="reference");double cpu_ms=1000.*(std::clock()-cpu)/CLOCKS_PER_SEC;
 std::ofstream o(std::string(argv[2])+".bin",std::ios::binary);int64_t size=r.order.size();o.write((char*)&size,8);o.write((char*)r.order.data(),size*8);o.write((char*)r.gains.data(),size*8);
 std::ofstream c(std::string(argv[2])+"_cells.csv");c<<"cell,n,evaluations,selected,cached_pairs,cache_hits,similarity_ms,init_ms,rescore_ms,coverage_ms,total_ms\n";for(size_t i=0;i<r.stats.size();++i){auto&s=r.stats[i];c<<i<<','<<s.n<<','<<s.evaluations<<','<<s.selected<<','<<s.pairs<<','<<s.hits<<','<<s.similarity_ms<<','<<s.init_ms<<','<<s.rescore_ms<<','<<s.coverage_ms<<','<<s.total_ms<<'\n';}
 struct rusage usage;getrusage(RUSAGE_SELF,&usage);std::ofstream j(std::string(argv[2])+".json");j<<"{\"wall_ms\":"<<r.total_ms<<",\"cpu_ms\":"<<cpu_ms<<",\"grid_ms\":"<<r.grid_ms<<",\"membership_ms\":"<<r.membership_ms<<",\"merge_ms\":"<<r.merge_ms<<",\"pair_bytes\":"<<r.pair_bytes<<",\"peak_rss_kb\":"<<usage.ru_maxrss<<",\"gpu_ms\":0,\"selected\":"<<r.order.size()<<",\"cells\":"<<r.stats.size()<<"}\n";
}
