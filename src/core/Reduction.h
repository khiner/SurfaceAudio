#pragma once

#include <metal_stdlib>

// Every lane participates. Only lane zero receives the sum, in ascending SIMD-group order.
template<typename T> inline T SumThreadgroup(T value, threadgroup T *sums, uint groups, uint lane, uint simd_lane, uint simd_id) {
    const T sum = metal::simd_sum(value);
    if (!simd_lane) sums[simd_id] = sum;
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    T total = 0;
    if (!lane)
        for (uint index = 0; index < groups; ++index) total += sums[index];
    return total;
}
