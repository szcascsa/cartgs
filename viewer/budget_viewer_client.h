#pragma once

#include <json/json.h>
#include <opencv2/core.hpp>

// 固定预算模式专用的本机推理客户端；不参与训练或原Full Viewer路径。
class BudgetViewerClient {
 public:
  struct Frame {
    cv::Mat image, reference;  // FP32 RGB, HWC
    Json::Value metadata;
  };

  explicit BudgetViewerClient(int port);
  const Json::Value& metadata() const { return metadata_; }
  Frame render(const Json::Value& request) const;

 private:
  std::string request(const std::string& path, const std::string& body = "") const;
  int port_;
  Json::Value metadata_;
};
