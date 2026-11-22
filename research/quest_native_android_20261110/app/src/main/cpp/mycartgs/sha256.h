#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace mycartgs {

std::array<uint8_t, 32> Sha256(const uint8_t* data, size_t size);
std::string Sha256Hex(const uint8_t* data, size_t size);

}  // namespace mycartgs
