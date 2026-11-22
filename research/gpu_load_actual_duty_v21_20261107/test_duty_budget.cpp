#include "duty_budget.h"

#include <cassert>
#include <iostream>
#include <limits>

int main() {
    assert(requiredWindowNanoseconds(100000, .8) == 125000);
    assert(requiredWindowNanoseconds(100000, .2) == 500000);
    assert(remainingOffNanoseconds(100000, 110000, .8) == 15000);
    assert(remainingOffNanoseconds(100000, 200000, .8) == 0);
    assert(requiredWindowNanoseconds(0, 0.) == 0);
    for (double duty : {-1., .80001, 1., std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
        bool rejected = false;
        try { validateDuty(duty); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    for (double duty : {.2, .4, .6, .8}) {
        int64_t elapsed = 0, charged = 0;
        for (int pulse = 0; pulse < 10000; ++pulse) {
            const int64_t measured = 80000 + (pulse * 701) % 500000;
            elapsed += measured + (pulse % 7) * 1000;
            charged += measured;
            elapsed += remainingOffNanoseconds(charged, elapsed, duty);
            assert(static_cast<long double>(charged) / elapsed <= duty + 1e-12);
        }
    }
    std::cout << "DUTY_BUDGET_CAP_AND_VARIABLE_DURATION_PASS\n";
}
