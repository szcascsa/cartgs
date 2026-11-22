#pragma once

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

using SmIntervals = std::map<unsigned int,
    std::vector<std::pair<unsigned long long,unsigned long long>>>;

inline long double equivalentSmNanoseconds(SmIntervals& intervals,int smCount) {
    long double area=0;
    for (auto& item:intervals) {
        auto& spans=item.second;
        if (spans.empty()) continue;
        std::sort(spans.begin(),spans.end());
        auto begin=spans.front().first,end=spans.front().second;
        for (size_t index=1;index<spans.size();++index) {
            if (spans[index].first>end) {
                area+=end-begin;
                begin=spans[index].first;
            }
            end=std::max(end,spans[index].second);
        }
        area+=end-begin;
    }
    return area/smCount;
}
