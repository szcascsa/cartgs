#include "viewer/budget_viewer_client.h"

#include <cstdint>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  try {
    BudgetViewerClient client(std::stoi(argv[1]));
    Json::Value request;
    request["frame"] = std::stoi(argv[2]);
    request["budget"] = std::stoi(argv[3]);
    const auto frame = client.render(request);
    std::uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < frame.image.total() * frame.image.elemSize(); ++i) {
      hash ^= frame.image.data[i];
      hash *= 1099511628211ULL;
    }
    Json::Value result = frame.metadata;
    result["image_fnv64"] = std::to_string(hash);
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    std::cout << Json::writeString(writer, result) << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
