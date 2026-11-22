#include "viewer/budget_viewer_client.h"
#include "viewer/imgui_viewer.h"

#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " local_backend_port\n";
    return 1;
  }
  try {
    size_t parsed = 0;
    const std::string port_text(argv[1]);
    const int port = std::stoi(port_text, &parsed);
    if (parsed != port_text.size()) throw std::runtime_error("Invalid backend port");
    ImGuiViewer(std::make_shared<BudgetViewerClient>(port)).run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[view_budget] " << error.what() << std::endl;
    return 1;
  }
}
