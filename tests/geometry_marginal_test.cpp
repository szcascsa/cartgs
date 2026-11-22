#undef NDEBUG
#include "include/geometry_marginal_importance.h"
#include <cassert>
#include <unistd.h>
int main(){
 torch::manual_seed(2026);auto x=torch::rand({50,3});auto scale=torch::full({50,3},.01);auto raw=torch::rand({50});auto valid=torch::ones({50},torch::kBool);GeometryMarginalImportance g;
 g.reset(0,x.options());g.append(50);g.assertAligned(50);
 auto dir=std::filesystem::temp_directory_path()/("cartgs_geometry_marginal_test_"+std::to_string(getpid()));
 g.refresh(x,scale,raw,valid,1000,5,1,dir,true,true,false);assert(g.ready());auto old=g.score().clone();auto keep=torch::arange(50)%3!=0;
 g.prune(keep);g.assertAligned(keep.sum().item<int64_t>());assert(torch::equal(g.score(),old.index({keep})));
 int64_t n=g.score().numel();g.append(7);g.assertAligned(n+7);assert(!g.valid().slice(0,n).any().item<bool>());assert(g.score().slice(0,n).sum().item<float>()==0);
 // Reorder is explicit reset + current-geometry rebuild, never inferred row IDs.
 auto order=torch::arange(49,-1,-1,torch::kLong);g.reset(50,x.options());assert(!g.ready());g.refresh(x.index_select(0,order),scale,raw.index_select(0,order),valid,2000,6,2,dir,true,true,false);assert(g.ready());g.assertAligned(50);
}
