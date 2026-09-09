#include "core/Reduction.h"
using namespace metal;

struct EndpointMixBlock {
    uint Modes, Frames, Groups;
};

kernel void EndpointMixSynthesize(constant EndpointMixBlock &p [[buffer(0)]], device const float *parameters [[buffer(1)]], device const float *basis [[buffer(2)]], device const float *location [[buffer(3)]], device float *components [[buffer(4)]], uint2 position [[thread_position_in_grid]]) {
    if (position.x >= p.Frames || position.y >= p.Modes) return;
    const uint index = position.y * p.Frames + position.x;
    const float x = location[position.x];
    components[index] = basis[index] * exp((1 - x) * parameters[position.y] + x * parameters[p.Modes + position.y]);
}

kernel void EndpointMixDifferentiate(constant EndpointMixBlock &p [[buffer(0)]], device const float *components [[buffer(1)]], device const float *location [[buffer(2)]], device const float *adjoint [[buffer(3)]], device float2 *partial [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]]) {
    const uint frame = group.x * 256 + lane;
    const float x = frame < p.Frames ? location[frame] : 0;
    const float derivative = frame < p.Frames ? adjoint[frame] * components[group.y * p.Frames + frame] : 0;
    const float2 value = derivative * float2(1 - x, x);
    threadgroup float2 sums[8];
    const float2 total = SumThreadgroup(value, sums, 8, lane, simd_lane, simd_id);
    if (!lane) partial[group.y * p.Groups + group.x] = total;
}

kernel void EndpointMixReduce(constant EndpointMixBlock &p [[buffer(0)]], device const float2 *partial [[buffer(1)]], device float *gradient [[buffer(2)]], uint mode [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]]) {
    float2 value = 0;
    for (uint group = lane; group < p.Groups; group += 256) value += partial[mode * p.Groups + group];
    threadgroup float2 sums[8];
    const float2 total = SumThreadgroup(value, sums, 8, lane, simd_lane, simd_id);
    if (!lane) {
        gradient[mode] = total.x;
        gradient[p.Modes + mode] = total.y;
    }
}
