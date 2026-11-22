#include "include/hierarchical_representative_si.h"
#include <torch/cuda.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>

namespace hierarchical_si {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void sync(const torch::Tensor& tensor) {
  if (tensor.is_cuda()) torch::cuda::synchronize(tensor.get_device());
}
torch::Tensor geometry(const torch::Tensor& x, const torch::Tensor& radius,
    const torch::Tensor& all_x, const torch::Tensor& all_radius) {
  auto distance = (x.unsqueeze(1) - all_x.unsqueeze(0)).square().sum(2);
  auto sigma = .5 * (radius.unsqueeze(1) + all_radius.unsqueeze(0));
  return torch::exp(-distance / (2 * sigma.square() + kEpsilon)).clamp(0, 1);
}
double quantile(std::vector<double> values, double p) {
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end());
  double index = p * (values.size() - 1);
  auto lo = static_cast<size_t>(std::floor(index)), hi = static_cast<size_t>(std::ceil(index));
  return values[lo] + (values[hi] - values[lo]) * (index - lo);
}
double mean(const std::vector<double>& values) {
  double sum = 0;
  for (double value : values) sum += value;
  return values.empty() ? 0 : sum / values.size();
}
template <typename T> void array(std::ostream& stream, const T& values) {
  stream << '[';
  bool first = true;
  for (auto value : values) { if (!first) stream << ','; stream << value; first = false; }
  stream << ']';
}
}  // namespace

CellSelection selectCell(const torch::Tensor& xyz, const torch::Tensor& radius,
    const torch::Tensor& raw, const torch::Tensor& source_rows,
    double* similarity_ms, double* selection_ms) {
  torch::NoGradGuard guard;
  const int64_t n = raw.numel();
  TORCH_CHECK(xyz.dim() == 2 && xyz.size(0) == n && xyz.size(1) == 3 &&
      radius.numel() == n && source_rows.numel() == n && xyz.scalar_type() == torch::kFloat64 &&
      radius.scalar_type() == torch::kFloat64 && raw.scalar_type() == torch::kFloat64,
      "Hierarchical SI cell shape/dtype mismatch");
  CellSelection result;
  result.cover = torch::zeros_like(raw);
  result.mass = raw.sum().item<double>();
  auto chosen = torch::zeros({n}, raw.options().dtype(torch::kBool));
  for (int k = 0; k < std::min<int64_t>(kRepresentatives, n); ++k) {
    auto gains = torch::empty_like(raw);
    for (int64_t begin = 0; begin < n; begin += 128) {
      int64_t end = std::min(begin + 128, n);
      sync(raw); auto start = Clock::now();
      auto kernel = geometry(xyz.slice(0, begin, end), radius.slice(0, begin, end), xyz, radius);
      auto diagonal = torch::arange(begin, end, source_rows.options());
      kernel.index_put_({diagonal - begin, diagonal}, 1.0);
      sync(raw); if (similarity_ms) *similarity_ms += elapsed(start);
      start = Clock::now();
      gains.slice(0, begin, end).copy_(((kernel - result.cover.unsqueeze(0)).clamp_min(0) * raw).sum(1));
      sync(raw); if (selection_ms) *selection_ms += elapsed(start);
    }
    auto start = Clock::now();
    gains.masked_fill_(chosen, -std::numeric_limits<double>::infinity());
    auto tied_rows = torch::where(gains == gains.max(), source_rows,
        torch::full_like(source_rows, std::numeric_limits<int64_t>::max()));
    int64_t best = tied_rows.argmin().item<int64_t>();
    result.representatives.push_back(best);
    result.gains.push_back(gains[best].item<double>());
    chosen[best] = true;
    sync(raw); if (selection_ms) *selection_ms += elapsed(start);
    start = Clock::now();
    auto row = geometry(xyz.slice(0, best, best + 1), radius.slice(0, best, best + 1), xyz, radius).reshape({n});
    row[best] = 1.0;
    sync(raw); if (similarity_ms) *similarity_ms += elapsed(start);
    start = Clock::now();
    result.cover = torch::maximum(result.cover, row);
    sync(raw); if (selection_ms) *selection_ms += elapsed(start);
  }
  result.covered_mass = (raw * result.cover).sum().item<double>();
  result.quality = result.covered_mass / (result.mass + kEpsilon);
  return result;
}

Result build(const torch::Tensor& xyz, const torch::Tensor& activated_scale,
    const torch::Tensor& raw, const torch::Tensor& valid, const torch::Tensor& observed, RootBackend backend) {
  torch::NoGradGuard guard;
  TORCH_CHECK(raw.dim() == 1 && raw.is_floating_point() && xyz.dim() == 2 && xyz.size(1) == 3 &&
      xyz.size(0) == raw.numel() && activated_scale.sizes() == xyz.sizes() &&
      valid.sizes() == raw.sizes() && observed.sizes() == raw.sizes() &&
      valid.scalar_type() == torch::kBool && observed.scalar_type() == torch::kBool,
      "Hierarchical SI population shape mismatch");
  for (const auto& tensor : {xyz, activated_scale, valid, observed})
    TORCH_CHECK(tensor.device() == raw.device(), "Hierarchical SI device mismatch");
  sync(raw); auto start = Clock::now();
  Result result; result.backend = backend; result.population = raw.numel();
  if (backend != RootBackend::LegacyAabb2048) result.dimensions = {0,0,0};
  result.importance = torch::zeros_like(raw);
  result.leaf_assignment = torch::full({raw.numel()}, -1, raw.options().dtype(torch::kLong).device(torch::kCPU));
  auto x = xyz.detach().to(torch::kFloat64);
  auto scale = activated_scale.detach().to(torch::kFloat64);
  auto weight = raw.detach().to(torch::kFloat64);
  auto indices = torch::nonzero(valid & observed).reshape({-1});
  result.candidates = indices.numel();
  auto finite = torch::isfinite(x).all(1);
  if (!finite.any().item<bool>()) {
    TORCH_CHECK(!result.candidates, "Observed GI rows have no finite geometry");
    result.build_ms = elapsed(start); return result;
  }
  auto finite_x = backend == RootBackend::LegacyAabb2048 ? x.index({finite}) : x.index_select(0, indices);
  if (!finite_x.size(0)) { result.dimensions = {0,0,0}; result.build_ms = elapsed(start); return result; }
  auto minimum = std::get<0>(finite_x.min(0)).to(torch::kCPU);
  auto maximum = std::get<0>(finite_x.max(0)).to(torch::kCPU);
  double largest_extent = (maximum - minimum).max().item<double>();
  for (int axis = 0; axis < 3; ++axis) {
    double lo = minimum[axis].item<double>(), hi = maximum[axis].item<double>();
    double pad = std::max({1e-9, largest_extent * 1e-9,
        std::max(std::abs(lo), std::abs(hi)) * 32 * std::numeric_limits<double>::epsilon()});
    result.lower[axis] = lo - pad; result.upper[axis] = hi + pad;
  }
  // Diagnostic legacy backend only: preserve the previous AABB factorization exactly.
  // Equal objectives retain lexicographically first (nx,ny,nz).
  double best_shape = std::numeric_limits<double>::infinity();
  if (backend == RootBackend::LegacyAabb2048)
  for (int ex = 0; ex <= kGridBits; ++ex) for (int ey = 0; ey <= kGridBits - ex; ++ey) {
    std::array<int, 3> dims{1 << ex, 1 << ey, 1 << (kGridBits - ex - ey)};
    std::array<double, 3> logs{}; double average = 0;
    for (int a = 0; a < 3; ++a) { logs[a] = std::log((result.upper[a] - result.lower[a]) / dims[a]); average += logs[a] / 3; }
    double objective = 0; for (double value : logs) objective += (value - average) * (value - average);
    if (objective < best_shape - 1e-12) { best_shape = objective; result.dimensions = dims; }
  }
  if (backend == RootBackend::LegacyAabb2048) {
    for (int a = 0; a < 3; ++a) result.cell_size[a] = (result.upper[a] - result.lower[a]) / result.dimensions[a];
    TORCH_CHECK(result.dimensions[0] * result.dimensions[1] * result.dimensions[2] == kInitialCells,
        "Legacy initial cell count must equal 2048");
  }
  if (result.candidates) {
    TORCH_CHECK(finite.index_select(0, indices).all().item<bool>() &&
        torch::isfinite(scale.index_select(0, indices)).all().item<bool>() &&
        (scale.index_select(0, indices) > 0).all().item<bool>() &&
        torch::isfinite(weight.index_select(0, indices)).all().item<bool>() &&
        (weight.index_select(0, indices) >= 0).all().item<bool>(), "Invalid candidate geometry/scale/Raw SI");
  }
  // Log-product avoids intermediate overflow/underflow; input is activated physical scale.
  auto radius = torch::exp(scale.log().mean(1));
  auto cpu_x = x.to(torch::kCPU).contiguous();
  auto cpu_rows = indices.to(torch::kCPU).contiguous();
  auto positions = cpu_x.accessor<double, 2>(); auto rows = cpu_rows.accessor<int64_t, 1>();
  std::map<RootKey, std::vector<int64_t>> roots;
  auto grouping_start = Clock::now();
  if (backend != RootBackend::LegacyAabb2048) {
    std::vector<Position> candidate_xyz; std::vector<double> candidate_radius; std::vector<int64_t> source_rows;
    auto cpu_radius = radius.to(torch::kCPU).contiguous(); auto radii = cpu_radius.accessor<double,1>();
    for (int64_t k = 0; k < result.candidates; ++k) {
      auto row = rows[k]; candidate_xyz.push_back({positions[row][0],positions[row][1],positions[row][2]});
      candidate_radius.push_back(radii[row]); source_rows.push_back(row);
    }
    auto sparse = backend == RootBackend::SparseHashD50x32 ?
        makeD50NormalizedRootGrid(candidate_xyz, candidate_radius, source_rows) :
        makeSparseRootGrid(candidate_xyz, candidate_radius, source_rows);
    result.h_search_ms = sparse.search_ms; result.root_grouping_ms = sparse.grouping_ms;
    result.h_root = sparse.h; result.d50 = sparse.d50; result.h_seed = sparse.h_seed;
    result.bracket_low = sparse.bracket_low; result.bracket_high = sparse.bracket_high;
    result.crossing_bracketed = sparse.crossing_bracketed; result.search_samples = std::move(sparse.samples);
    roots = std::move(sparse.roots); result.cell_size = {result.h_root,result.h_root,result.h_root};
    result.dimensions = {0,0,0};  // Sparse triplets are never interpreted as a dense shape.
  } else for (int64_t k = 0; k < result.candidates; ++k) {
    int coordinate[3];
    for (int a = 0; a < 3; ++a) coordinate[a] = std::clamp(static_cast<int>(std::floor(
        (positions[rows[k]][a] - result.lower[a]) / result.cell_size[a])), 0, result.dimensions[a] - 1);
    roots[{coordinate[0], coordinate[1], coordinate[2]}].push_back(rows[k]);
  }
  if (backend == RootBackend::LegacyAabb2048) result.root_grouping_ms = elapsed(grouping_start);
  auto output = torch::zeros_like(weight);
  auto assignment = result.leaf_assignment.accessor<int64_t, 1>();
  std::function<void(const std::vector<int64_t>&, Node)> visit;
  visit = [&](const std::vector<int64_t>& members, Node node) {
    auto local_rows = torch::tensor(members, indices.options());
    auto local_raw = weight.index_select(0, local_rows);
    auto selection = selectCell(x.index_select(0, local_rows), radius.index_select(0, local_rows),
        local_raw, local_rows, &result.similarity_ms, &result.selection_ms);
    node.occupancy = members.size(); node.mass = selection.mass;
    node.covered_mass = selection.covered_mass; node.quality = selection.quality;
    node.gains = selection.gains;
    for (auto i : selection.representatives) {
      node.representatives.push_back(members[i]); node.representative_raw.push_back(local_raw[i].item<double>());
    }
    node.leaf = node.level == kMaxLevel || node.quality >= kSplitCoverage;
    int64_t node_id = result.nodes.size(); result.nodes.push_back(node);
    if (!node.leaf) {
      auto split_start = Clock::now();
      std::array<std::vector<int64_t>, 8> children;
      std::array<double, 3> midpoint;
      for (int a = 0; a < 3; ++a) midpoint[a] = node.lower[a] + .5 * (node.upper[a] - node.lower[a]);
      for (auto row : members) {
        int child = 0;
        for (int a = 0; a < 3; ++a) child |= int(positions[row][a] >= midpoint[a]) << a;
        children[child].push_back(row);
      }
      int64_t child_count = 0; for (const auto& child : children) child_count += child.size();
      TORCH_CHECK(child_count == node.occupancy, "Hierarchy children must exactly partition parent rows");
      result.recursive_split_ms += elapsed(split_start);
      for (int child = 0; child < 8; ++child) if (!children[child].empty()) {
        Node next; next.level = node.level + 1; next.root = node.root; next.root_key = node.root_key; next.path = node.path + char('0' + child);
        for (int a = 0; a < 3; ++a) {
          next.lower[a] = child & (1 << a) ? midpoint[a] : node.lower[a];
          next.upper[a] = child & (1 << a) ? node.upper[a] : midpoint[a];
        }
        visit(children[child], next);
      }
      return;  // Parent provisional deltas never enter output.
    }
    auto local_output = local_raw * (1 - selection.cover);
    for (size_t k = 0; k < selection.representatives.size(); ++k)
      local_output[selection.representatives[k]] = selection.gains[k];
    double actual_mass = local_output.sum().item<double>();
    auto& leaf = result.nodes[node_id];
    double gain_sum = 0; for (double gain : selection.gains) gain_sum += gain;
    leaf.residual_mass = actual_mass - gain_sum;
    leaf.conservation_absolute = std::abs(actual_mass - leaf.mass);
    leaf.conservation_relative = leaf.conservation_absolute / std::max(leaf.mass, kEpsilon);
    TORCH_CHECK(leaf.conservation_absolute <= 1e-12 + 1e-6 * leaf.mass,
        "Hierarchical SI leaf mass not conserved; no renormalization is permitted");
    output.index_copy_(0, local_rows, local_output);
    for (auto row : members) { TORCH_CHECK(assignment[row] == -1, "Duplicate hierarchy leaf assignment"); assignment[row] = node_id; }
  };
  int64_t root_ordinal = 0;
  for (const auto& item : roots) {
    Node root; root.root_key = item.first;
    root.root = backend == RootBackend::LegacyAabb2048 ?
        (item.first[0] * result.dimensions[1] + item.first[1]) * result.dimensions[2] + item.first[2] : root_ordinal;
    ++root_ordinal;
    for (int a = 0; a < 3; ++a) {
      if (backend == RootBackend::LegacyAabb2048) {
        root.lower[a] = result.lower[a] + item.first[a] * result.cell_size[a];
        root.upper[a] = result.lower[a] + (item.first[a] + 1) * result.cell_size[a];
      } else {
        root.lower[a] = static_cast<double>(item.first[a]) * result.h_root;
        root.upper[a] = root.lower[a] + result.h_root;
        TORCH_CHECK(std::isfinite(root.lower[a]) && std::isfinite(root.upper[a]) && root.upper[a] > root.lower[a],
            "Sparse cube bounds cannot be resolved in FP64 at this coordinate scale");
      }
    }
    visit(item.second, root);
  }
  result.importance = output.to(raw.scalar_type());
  TORCH_CHECK(torch::isfinite(result.importance).all().item<bool>() && (result.importance >= 0).all().item<bool>(),
      "Hierarchical SI output must be finite and nonnegative");
  TORCH_CHECK((result.leaf_assignment >= 0).sum().item<int64_t>() == result.candidates, "Hierarchy population lost rows");
  result.raw_sum = weight.index_select(0, indices).sum().item<double>();
  result.output_sum = result.importance.to(torch::kFloat64).sum().item<double>();
  result.conservation_absolute = std::abs(result.raw_sum - result.output_sum);
  result.conservation_relative = result.conservation_absolute / std::max(result.raw_sum, kEpsilon);
  TORCH_CHECK(result.conservation_absolute <= 1e-12 + 1e-6 * result.raw_sum, "Hierarchy total mass not conserved");
  sync(raw); result.build_ms = elapsed(start);
  return result;
}

torch::Tensor score(bool enabled, const torch::Tensor& xyz, const torch::Tensor& scale,
    const torch::Tensor& raw, const torch::Tensor& valid, const torch::Tensor& observed, Result* result) {
  if (!enabled) return raw;
  auto built = build(xyz, scale, raw, valid, observed);
  auto importance = built.importance;
  if (result) *result = std::move(built);
  return importance;
}

void writeDiagnostics(const Result& result, const std::filesystem::path& directory, const std::string& stem) {
  std::filesystem::create_directories(directory);
  std::array<int64_t, 3> counts{}, splits{}, leaves{}, representatives{};
  std::array<std::vector<double>, 3> occupancy;
  std::vector<double> quality; int64_t unresolved = 0;
  std::ofstream csv(directory / (stem + "_tree.csv"));
  csv << std::setprecision(17) << "node,leaf,level,root,path,key_x,key_y,key_z,xmin,ymin,zmin,xmax,ymax,zmax,occupancy,raw_mass,covered_mass,Q,residual_mass,conservation_absolute,conservation_relative,rep1,raw1,delta1,rep2,raw2,delta2,rep3,raw3,delta3\n";
  for (size_t i = 0; i < result.nodes.size(); ++i) {
    const auto& node = result.nodes[i]; auto level = node.level - 1;
    ++counts[level]; occupancy[level].push_back(node.occupancy);
    if (node.leaf) { ++leaves[level]; representatives[level] += node.representatives.size(); quality.push_back(node.quality);
      if (node.level == 3 && node.quality < kSplitCoverage) ++unresolved;
    } else ++splits[level];
    csv << i << ',' << node.leaf << ',' << node.level << ',' << node.root << ',' << node.path;
    for (auto value : node.root_key) csv << ',' << value;
    for (double value : node.lower) csv << ',' << value;
    for (double value : node.upper) csv << ',' << value;
    csv << ',' << node.occupancy << ',' << node.mass << ',' << node.covered_mass << ',' << node.quality << ',' << node.residual_mass
        << ',' << node.conservation_absolute << ',' << node.conservation_relative;
    for (size_t k = 0; k < 3; ++k) {
      if (k < node.representatives.size()) csv << ',' << node.representatives[k] << ',' << node.representative_raw[k] << ',' << node.gains[k];
      else csv << ",,,";
    }
    csv << '\n';
  }
  TORCH_CHECK(bool(csv), "Hierarchy tree diagnostic write failed");
  std::ofstream json(directory / (stem + "_telemetry.json")); json << std::setprecision(17);
  json << "{\"population\":" << result.population << ",\"candidates\":" << result.candidates << ",\"dimensions\":";
  array(json, result.dimensions); json << ",\"AABB_min\":"; array(json, result.lower); json << ",\"AABB_max\":"; array(json, result.upper);
  json << ",\"root_backend\":\"" << (result.backend == RootBackend::LegacyAabb2048 ? "legacy_aabb_2048" :
      result.backend == RootBackend::SparseHashD50x32 ? "sparse_hash_d50x32" : "sparse_hash_target_2048") << '"';
  json << ",\"origin\":";
  array(json, result.backend == RootBackend::LegacyAabb2048 ? result.lower : std::array<double,3>{0,0,0});
  if (result.backend != RootBackend::SparseHashD50x32)
    json << ",\"target_occupied_roots\":" << kTargetOccupiedRoots;
  else json << ",\"root_scale_rule\":\"32*d50\"";
  json << ",\"achieved_occupied_roots\":" << counts[0];
  json << ",\"h_root\":" << result.h_root << ",\"d50\":" << result.d50;
  if (result.backend == RootBackend::SparseHashD50x32)
    json << ",\"h_L1\":" << result.h_root << ",\"h_L2\":" << result.h_root/2 << ",\"h_L3\":" << result.h_root/4;
  else json << ",\"h_seed\":" << result.h_seed << ",\"bracket_low\":" << result.bracket_low
       << ",\"bracket_high\":" << result.bracket_high << ",\"crossing_bracketed\":" << result.crossing_bracketed
       << ",\"search_evaluations\":" << result.search_samples.size();
  json << ",\"cell_size\":"; array(json, result.cell_size); json << ",\"cells_by_level\":"; array(json, counts);
  json << ",\"splits_by_level\":"; array(json, splits); json << ",\"leaves_by_level\":"; array(json, leaves);
  json << ",\"representatives_by_level\":"; array(json, representatives);
  json << ",\"Q_mean\":" << mean(quality) << ",\"Q_median\":" << quantile(quality, .5)
       << ",\"Q_p10\":" << quantile(quality, .1) << ",\"Q_p90\":" << quantile(quality, .9)
       << ",\"unresolved_L3\":" << unresolved << ",\"occupancy_by_level\":[";
  for (int l = 0; l < 3; ++l) { if (l) json << ',';
    json << "{\"mean\":" << mean(occupancy[l]) << ",\"median\":" << quantile(occupancy[l], .5)
         << ",\"max\":" << quantile(occupancy[l], 1) << '}'; }
  json << "],\"raw_sum\":" << result.raw_sum << ",\"hierarchical_sum\":" << result.output_sum
       << ",\"conservation_absolute\":" << result.conservation_absolute << ",\"conservation_relative\":" << result.conservation_relative
       << ",\"build_ms\":" << result.build_ms << ",\"similarity_ms\":" << result.similarity_ms
       << ",\"selection_ms\":" << result.selection_ms << ",\"h_search_ms\":" << result.h_search_ms
       << ",\"root_grouping_ms\":" << result.root_grouping_ms << ",\"recursive_split_ms\":" << result.recursive_split_ms << "}\n";
  TORCH_CHECK(bool(json), "Hierarchy telemetry diagnostic write failed");
  if (!result.search_samples.empty()) {
    std::ofstream search(directory / (stem + "_h_search.csv")); search << std::setprecision(17) << "evaluation,h,occupied,absolute_error\n";
    for (size_t i = 0; i < result.search_samples.size(); ++i) {
      const auto& sample = result.search_samples[i];
      search << i << ',' << sample.h << ',' << sample.occupied << ',' << std::abs(sample.occupied - kTargetOccupiedRoots) << '\n';
    }
    TORCH_CHECK(bool(search), "Sparse root search diagnostic write failed");
  }
}
}  // namespace hierarchical_si
