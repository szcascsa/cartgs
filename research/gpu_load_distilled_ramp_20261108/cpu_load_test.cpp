#include "sm_area.h"
#include "duty_budget.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

int main() {
    SmIntervals empty;
    assert(equivalentSmNanoseconds(empty,82)==0);
    SmIntervals overlap{{0,{{50,150},{0,100},{200,250}}},{1,{{25,75}}}};
    assert(equivalentSmNanoseconds(overlap,2)==125);
    SmIntervals nested{{0,{{5,90},{20,30},{0,1000}}}};
    assert(equivalentSmNanoseconds(nested,1)==1000);
    SmIntervals touching{{0,{{2,4},{1,2},{4,4}}}};
    assert(equivalentSmNanoseconds(touching,1)==3);
    SmIntervals full{{0,{{0,100},{0,100},{0,100}}},{1,{{0,100},{0,100},{0,100}}}};
    assert(equivalentSmNanoseconds(full,2)==100);
    SmIntervals partial{{0,{{0,100},{0,100},{0,100}}}};
    assert(equivalentSmNanoseconds(partial,2)==50);
    const unsigned long long origin=23000000000000000ull;
    SmIntervals absolute{{0,{{origin,origin+1},{origin+1,origin+3}}}};
    assert(equivalentSmNanoseconds(absolute,2)==1.5L);
    assert(std::ceil(equivalentSmNanoseconds(absolute,2))==2);
    for (double duty:{0.,.2,.4,.6,.8}) validateDuty(duty);
    for (double duty:{-.01,.80001,std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::quiet_NaN()}) {
        bool rejected=false;
        try {validateDuty(duty);} catch(const std::invalid_argument&) {rejected=true;}
        assert(rejected);
    }
    // GPU等待和SM驻留面积不同：250ns包络中仅125ns等效整卡驻留。
    // 若按包络计费会少施加一半实际负载，测试确保使用SM面积计费。
    const auto charged=std::ceil(equivalentSmNanoseconds(overlap,2));
    assert(charged==125 && charged!=250);
    std::cout<<"CPU_SM_AREA_AND_DUTY_TESTS_PASS\n";
}
