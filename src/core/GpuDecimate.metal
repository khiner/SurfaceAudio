#include <metal_stdlib>
using namespace metal;

struct DecimateBlock {
    uint InputFrames, OutputFrames, Channels, Factor, Radius;
};
kernel void DecimateCopy(constant DecimateBlock &p [[buffer(0)]], device const float *input [[buffer(2)]], device float *output [[buffer(3)]], uint2 index [[thread_position_in_grid]]) {
    if (index.x < p.OutputFrames && index.y < p.Channels) output[index.y * p.OutputFrames + index.x] = input[index.y * p.InputFrames + index.x];
}
kernel void DecimateKaiserSinc(constant DecimateBlock &p [[buffer(0)]], device const float *coefficients [[buffer(1)]], device const float *input [[buffer(2)]], device float *output [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]]) {
    const uint center = group.x * p.Factor, base = group.y * p.InputFrames;
    float sum = 0;
    for (uint tap = lane; tap <= 2 * p.Radius; tap += 128) {
        const int offset = int(tap) - int(p.Radius);
        const uint sample = offset < 0 ? center - min(center, uint(-offset)) : center + min(p.InputFrames - 1 - center, uint(offset));
        sum += coefficients[tap] * input[base + sample];
    }
    const float subtotal = simd_sum(sum);
    threadgroup float partial[4];
    if (!simd_lane) partial[simd_id] = subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!lane) output[group.y * p.OutputFrames + group.x] = (partial[0] + partial[1]) + (partial[2] + partial[3]);
}
