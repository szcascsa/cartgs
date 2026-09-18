#pragma once

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string_view>

namespace improvements::selector {

inline std::optional<bool> extractSelectorEnabledOverride(int& argc,
                                                           char** argv) {
  constexpr std::string_view kOption = "--selector-enabled";
  std::optional<bool> enabled;
  int write_index = 1;

  for (int read_index = 1; read_index < argc; ++read_index) {
    const std::string_view argument(argv[read_index]);
    std::string_view value;
    bool is_selector_option = false;

    if (argument == kOption) {
      if (++read_index == argc) {
        std::cerr << kOption << " requires a value of 0 or 1." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      value = argv[read_index];
      is_selector_option = true;
    } else if (argument.size() > kOption.size() &&
               argument.substr(0, kOption.size()) == kOption &&
               argument[kOption.size()] == '=') {
      value = argument.substr(kOption.size() + 1);
      is_selector_option = true;
    }

    if (!is_selector_option) {
      argv[write_index++] = argv[read_index];
      continue;
    }

    if (enabled.has_value() || (value != "0" && value != "1")) {
      std::cerr << kOption << " accepts one value: 0 or 1." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    enabled = value == "1";
  }

  argc = write_index;
  argv[argc] = nullptr;
  return enabled;
}

}  // namespace improvements::selector
