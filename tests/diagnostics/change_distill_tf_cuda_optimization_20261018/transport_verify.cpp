// Included after the benchmark, so this uses its real frozen Snapshot loader.
Json::Value transportParity(Snapshot& snapshot) {
  torch::NoGradGuard no_grad;
  Json::Value result;
  auto compare = [&](const std::string& label, const ChangeEvent& event,
                     const GaussianAttrState& input, const torch::Tensor& indices) {
    const auto expected = referenceTransport(input, indices, event);
    const TargetTransport transport(event);
    const auto actual = transportTarget(input, indices, transport);
    const std::array<torch::Tensor, 3> a{expected.xyz, expected.log_scaling, expected.rotation};
    const std::array<torch::Tensor, 3> b{actual.xyz, actual.log_scaling, actual.rotation};
    for (int j = 0; j < 3; ++j) {
      auto delta = (a[j] - b[j]).abs();
      Json::Value row;
      row["max_abs"] = delta.numel() ? delta.max().item<double>() : 0.0;
      row["mean_abs"] = delta.numel() ? delta.mean().item<double>() : 0.0;
      row["allclose"] = torch::allclose(a[j], b[j], 2e-4, 2e-5);
      row["bitwise_equal"] = torch::equal(a[j], b[j]);
      result[label][j] = row;
      TORCH_CHECK(row["allclose"].asBool(), label, " attribute ", j, " transport mismatch");
    }
  };
  ChangeEvent event;
  event.old_aabb = snapshot.old_aabb;
  event.correction_groups = snapshot.groups;
  event.correction_group_id = torch::full({snapshot.before.xyz.size(0)}, -1,
      snapshot.mature.options().dtype(torch::kLong));
  for (size_t g = 0; g < snapshot.groups.size(); ++g)
    event.correction_group_id.index_fill_(0, snapshot.groups[g].indices, g);
  auto indices = torch::arange(snapshot.before.xyz.size(0), event.correction_group_id.options());
  compare("all_points_all_groups", event, snapshot.before, indices);
  for (size_t r = 0; r < snapshot.ratios.size(); ++r) {
    auto target = snapshot.tf->applyLogResidual(snapshot.before, snapshot.ratios[r]);
    compare("teacher_ratio_" + std::to_string(r), event, target, indices);
  }
  // Mixed groups, every flag combination, an empty group and identity IDs.
  ChangeEvent fixture;
  fixture.old_aabb = snapshot.old_aabb;
  fixture.correction_group_id = torch::tensor({-1, 0, 1, 2, 3, 4, 5, 6}, indices.options());
  for (int g = 0; g < 8; ++g) {
    auto group = snapshot.groups.empty() ? CorrectionGroup{} : snapshot.groups.at(g % snapshot.groups.size());
    if (snapshot.groups.empty()) {
      group.transform_4x4 = torch::eye(4, snapshot.before.xyz.options());
      group.rotation_wxyz = torch::tensor({1.0f, 0.0f, 0.0f, 0.0f}, snapshot.before.xyz.options());
    }
    group.indices = g == 7 ? indices.slice(0, 0, 0) : indices.slice(0, g + 1, g + 2);
    group.scale = 0.8f + g * 0.1f;
    group.changed_attributes = g + 1;
    fixture.correction_groups.push_back(group);
  }
  auto subset = indices.slice(0, 0, 8);
  auto input = snapshot.before.select(subset);
  input.rotation = input.rotation * 2.0; // Identity must not normalize this.
  compare("mixed_flags_empty_group_identity", fixture, input, subset);
  for (auto& group : fixture.correction_groups)
    group.rotation_wxyz = group.rotation_wxyz * 2.0;
  compare("non_unit_correction", fixture, input, subset);
  auto repeated = torch::tensor({7, 1, 1, 0, 3, 7, 2}, indices.options());
  compare("repeated_ids", fixture, input.select(repeated), repeated);
  compare("single_point_group", fixture, input.select(subset.slice(0,1,2)), subset.slice(0,1,2));
  for (auto& group : fixture.correction_groups) {
    group.transform_4x4 = torch::eye(4, input.xyz.options());
    group.scale = 1.0f;
    group.rotation_wxyz = torch::tensor({1.0f,0.0f,0.0f,0.0f}, input.xyz.options());
  }
  compare("identity_transform", fixture, input, subset);
  fixture.correction_group_id.fill_(-1);
  compare("all_minus_one_with_groups", fixture, input, subset);
  fixture.correction_groups.clear();
  fixture.correction_group_id.fill_(-1);
  compare("aabb_only_no_groups", fixture, input, subset);
  compare("empty_batch", fixture, input.select(subset.slice(0, 0, 0)), subset.slice(0, 0, 0));
  return result;
}
