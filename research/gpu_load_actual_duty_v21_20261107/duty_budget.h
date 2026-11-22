#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

constexpr double maximumDuty = 0.8;

inline void validateDuty(double duty) {
    if (!std::isfinite(duty) || duty < 0. || duty > maximumDuty) {
        throw std::invalid_argument("Duty must be finite and between 0 and 0.8");
    }
}

inline int64_t requiredWindowNanoseconds(int64_t chargedNanoseconds, double duty) {
    validateDuty(duty);
    if (chargedNanoseconds < 0 || (chargedNanoseconds > 0 && duty == 0.)) {
        throw std::invalid_argument("Invalid charged load duration");
    }
    if (chargedNanoseconds == 0) return 0;
    return static_cast<int64_t>(std::ceil(static_cast<long double>(chargedNanoseconds) / duty));
}

inline int64_t remainingOffNanoseconds(int64_t chargedNanoseconds, int64_t elapsedNanoseconds, double duty) {
    return std::max(int64_t(0), requiredWindowNanoseconds(chargedNanoseconds, duty) - elapsedNanoseconds);
}
