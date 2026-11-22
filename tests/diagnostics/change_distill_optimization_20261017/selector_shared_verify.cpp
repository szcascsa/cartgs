Json::Value sharedParity(Snapshot& snapshot) {
  Json::Value result;
  auto record = [&](const std::string& name, const torch::Tensor& a, const torch::Tensor& b) {
    const auto delta = (a - b).abs();
    Json::Value row;
    row["max_abs"] = a.numel() ? delta.max().item<double>() : 0.0;
    row["mean_abs"] = a.numel() ? delta.mean().item<double>() : 0.0;
    row["allclose"] = torch::allclose(a, b, 2e-4, 2e-5);
    row["bitwise_equal"] = torch::equal(a, b);
    result[name] = row;
    TORCH_CHECK(row["allclose"].asBool(), "Shared Selector mismatch: ", name);
  };
  for (int n : {1, 16, 17, 3800, 7430, 8192}) {
    for (int ratio_count : {1, 2, 4}) {
      auto original = student(snapshot.selector), shared = student(snapshot.selector);
      auto x = snapshot.after.xyz.slice(0, 0, n).clone().set_requires_grad(true);
      auto q = snapshot.after.rotation.slice(0, 0, n).clone().set_requires_grad(true);
      auto s = snapshot.after.log_scaling.slice(0, 0, n).exp().detach().set_requires_grad(true);
      auto xx = x.detach().clone().set_requires_grad(true);
      auto qq = q.detach().clone().set_requires_grad(true);
      auto ss = s.detach().clone().set_requires_grad(true);
      std::vector<float> ratios(snapshot.ratios.begin(), snapshot.ratios.begin() + ratio_count);
      auto outputs = shared->forwardProbabilities(xx, qq, ss, ratios);
      torch::Tensor old_loss, new_loss;
      const auto prefix = "n" + std::to_string(n) + "_r" + std::to_string(ratio_count);
      for (int r = 0; r < ratio_count; ++r) {
        auto output = original->forwardProbability(x, q, s,
            torch::full({n, 1}, ratios[r], x.options()));
        record(prefix + "_output_" + std::to_string(r), output, outputs[r]);
        auto a = (output - 0.3).square().mean() * (r + 1);
        auto b = (outputs[r] - 0.3).square().mean() * (r + 1);
        old_loss = old_loss.defined() ? old_loss + a : a;
        new_loss = new_loss.defined() ? new_loss + b : b;
      }
      record(prefix + "_loss", old_loss, new_loss);
      old_loss.backward(); new_loss.backward();
      for (size_t i = 0; i < original->parameters().size(); ++i)
        record(prefix + "_gradient_" + std::to_string(i),
            original->parameters()[i].grad(), shared->parameters()[i].grad());
      record(prefix + "_dx", x.grad(), xx.grad());
      record(prefix + "_dq", q.grad(), qq.grad());
      record(prefix + "_ds", s.grad(), ss.grad());
      auto a = torch::optim::Adam(original->parameters(), adamOptions(snapshot.meta["selector_adam"][0]));
      auto b = torch::optim::Adam(shared->parameters(), adamOptions(snapshot.meta["selector_adam"][0]));
      tf_cuda::tfCudaAdamDirect(a, {0}); tf_cuda::tfCudaAdamDirect(b, {0});
      for (size_t i = 0; i < original->parameters().size(); ++i)
        record(prefix + "_updated_" + std::to_string(i), original->parameters()[i], shared->parameters()[i]);
    }
  }
  return result;
}
