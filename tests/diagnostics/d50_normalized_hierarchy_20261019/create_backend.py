"""隔离source snapshot只增加固定32*d50 root backend，核心数学逐字保持。"""
from pathlib import Path
import hashlib,json,difflib
ROOT=Path(__file__).resolve().parent
names=['include/sparse_hash_root_grid.h','src/sparse_hash_root_grid.cpp','include/hierarchical_representative_si.h','src/hierarchical_representative_si.cpp']
before={n:(ROOT/'source'/n).read_text()for n in names};after=dict(before)
after[names[0]]=after[names[0]].replace('}  // namespace hierarchical_si','// Fixed scale experiment: no target count, search, Raw SI or quality inputs.\nSparseRootGrid makeD50NormalizedRootGrid(const std::vector<Position>& candidate_xyz,\n    const std::vector<double>& candidate_radius, const std::vector<int64_t>& source_rows);\n}  // namespace hierarchical_si')
function='''
SparseRootGrid makeD50NormalizedRootGrid(const std::vector<Position>& xyz,
    const std::vector<double>& radius, const std::vector<int64_t>& rows) {
  require(xyz.size() == radius.size() && xyz.size() == rows.size(), "D50 root arrays mismatch");
  SparseRootGrid result;
  if (xyz.empty()) return result;
  const auto start = Clock::now();
  auto sorted_radius = radius;
  for (size_t i = 0; i < xyz.size(); ++i) {
    require(radius[i] > 0 && std::isfinite(radius[i]), "D50 radius must be finite and positive");
    for (double value : xyz[i]) require(std::isfinite(value), "D50 xyz must be finite");
  }
  std::sort(sorted_radius.begin(), sorted_radius.end());
  const size_t middle = sorted_radius.size() / 2;
  const double median = sorted_radius.size() % 2 ? sorted_radius[middle] :
      sorted_radius[middle - 1] * .5 + sorted_radius[middle] * .5;
  result.d50 = 2 * median;
  result.h = 32 * result.d50;
  require(result.h > 0 && std::isfinite(result.h), "32*d50 outside finite positive range");
  for (size_t i = 0; i < xyz.size(); ++i)
    result.roots[sparseRootKey(xyz[i], result.h)].push_back(rows[i]);
  for (auto& item : result.roots) std::sort(item.second.begin(), item.second.end());
  result.grouping_ms = milliseconds(start);
  return result;
}
'''
after[names[1]]=after[names[1]].replace('}  // namespace hierarchical_si',function+'}  // namespace hierarchical_si')
after[names[2]]=after[names[2]].replace('LegacyAabb2048, SparseHashTarget2048','LegacyAabb2048, SparseHashTarget2048, SparseHashD50x32')
s=after[names[3]]
s=s.replace('if (backend == RootBackend::SparseHashTarget2048)', 'if (backend != RootBackend::LegacyAabb2048)')
s=s.replace('auto sparse = makeSparseRootGrid(candidate_xyz, candidate_radius, source_rows);','auto sparse = backend == RootBackend::SparseHashD50x32 ?\n        makeD50NormalizedRootGrid(candidate_xyz, candidate_radius, source_rows) :\n        makeSparseRootGrid(candidate_xyz, candidate_radius, source_rows);')
s=s.replace('<< ",\\\"initial_cells\\\":" << kInitialCells << ",\\\"dimensions\\\":";', '<< ",\\\"dimensions\\\":";')
s=s.replace('(result.backend == RootBackend::LegacyAabb2048 ? "legacy_aabb_2048" : "sparse_hash_target_2048")','(result.backend == RootBackend::LegacyAabb2048 ? "legacy_aabb_2048" :\n      result.backend == RootBackend::SparseHashD50x32 ? "sparse_hash_d50x32" : "sparse_hash_target_2048")')
s=s.replace('json << ",\\\"target_occupied_roots\\\":" << kTargetOccupiedRoots << ",\\\"achieved_occupied_roots\\\":" << counts[0];','if (result.backend != RootBackend::SparseHashD50x32)\n    json << ",\\\"target_occupied_roots\\\":" << kTargetOccupiedRoots;\n  else json << ",\\\"root_scale_rule\\\":\\\"32*d50\\\"";\n  json << ",\\\"achieved_occupied_roots\\\":" << counts[0];')
old='''  json << ",\\"h_root\\":" << result.h_root << ",\\"d50\\":" << result.d50 << ",\\"h_seed\\":" << result.h_seed
       << ",\\"bracket_low\\":" << result.bracket_low << ",\\"bracket_high\\":" << result.bracket_high
       << ",\\"crossing_bracketed\\":" << result.crossing_bracketed << ",\\"search_evaluations\\":" << result.search_samples.size();'''
new='''  json << ",\\"h_root\\":" << result.h_root << ",\\"d50\\":" << result.d50;
  if (result.backend == RootBackend::SparseHashD50x32)
    json << ",\\"h_L1\\":" << result.h_root << ",\\"h_L2\\":" << result.h_root/2 << ",\\"h_L3\\":" << result.h_root/4;
  else json << ",\\"h_seed\\":" << result.h_seed << ",\\"bracket_low\\":" << result.bracket_low
       << ",\\"bracket_high\\":" << result.bracket_high << ",\\"crossing_bracketed\\":" << result.crossing_bracketed
       << ",\\"search_evaluations\\":" << result.search_samples.size();'''
assert old in s;s=s.replace(old,new);after[names[3]]=s
def block(text,start,end):return text[text.index(start):text.index(end,text.index(start))]
pieces={'geometry':('torch::Tensor geometry(', '\ndouble quantile('),
        'selectCell':('CellSelection selectCell(', '\nResult build('),
        'recursive_greedy_split_redistribution':('  visit = [&]', '\n  int64_t root_ordinal')}
proof={}
for name,(a,b)in pieces.items():
    old=block(before[names[3]],a,b);new=block(after[names[3]],a,b);assert old==new,name
    proof[name]=hashlib.sha256(old.encode()).hexdigest()
for n,s in after.items():(ROOT/'source'/n).write_text(s)
(ROOT/'mathematical_parity.json').write_text(json.dumps(dict(all_passed=True,unchanged_blocks=proof,production_modified=False),indent=2)+'\n')
(ROOT/'source_modification.diff').write_text(''.join(''.join(difflib.unified_diff(before[n].splitlines(True),after[n].splitlines(True),fromfile='before/'+n,tofile='after/'+n))for n in names))
print('ROOT_BACKEND_ONLY_PATCH_PASS',proof)
