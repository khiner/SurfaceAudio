#include "core/Reduction.h"
using namespace metal;
struct FullSpectrumBlock {
    uint Samples, Size, Scale;
    float FloorSquared, Delta;
};
kernel void FullSpectrumPack(constant FullSpectrumBlock &p [[buffer(0)]], device const float *input [[buffer(1)]], device float2 *output [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.Size) output[i] = float2(i < p.Samples ? input[i] : 0, 0);
}
kernel void FullSpectrumCompare(constant FullSpectrumBlock &p [[buffer(0)]], device const float2 *spectrum [[buffer(1)]], device const float2 *target [[buffer(2)]], device float2 *gradient [[buffer(3)]], device float *loss [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.Size) return;
    float2 derivative = 0;
    if (i <= p.Size / 2) {
        const float2 x = spectrum[i], y = target[i];
        const float a = dot(x, x) + p.FloorSquared, b = dot(y, y) + p.FloorSquared;
        const float difference = p.Scale == 0 ? (10 / log(10.f)) * (log(a) - log(b)) : p.Scale == 1 ? .5f * (log(a) - log(b)) :
                                                                                                      sqrt(a) - sqrt(b);
        const float weight = 1.f / (p.Size / 2 + 1), absolute = abs(difference);
        loss[i] = weight * (absolute <= p.Delta ? .5f * difference * difference : p.Delta * (absolute - .5f * p.Delta));
        const float factor = p.Scale == 0 ? 20 / log(10.f) / a : p.Scale == 1 ? 1 / a :
                                                                                rsqrt(a);
        derivative = clamp(difference, -p.Delta, p.Delta) * weight * factor * x;
    }
    gradient[i] = derivative;
}
kernel void FullSpectrumAccumulate(constant FullSpectrumBlock &p [[buffer(0)]], device const float2 *inverse [[buffer(1)]], device float *gradient [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.Samples) gradient[i] += inverse[i].x * p.Size;
}
kernel void FullSpectrumReduce(constant FullSpectrumBlock &p [[buffer(0)]], device const float *bins [[buffer(1)]], device float *loss [[buffer(2)]], device float *total [[buffer(3)]], uint lane [[thread_index_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]]) {
    float4 sums = 0;
    for (uint i = lane; i <= p.Size / 2; i += 1024)
        for (uint j = 0; j < 4; ++j)
            if (i + j * 256 <= p.Size / 2) sums[j] += bins[i + j * 256];
    threadgroup float partial[8];
    const float value = SumThreadgroup((sums.x + sums.y) + (sums.z + sums.w), partial, 8, lane, simd_lane, simd_id);
    if (!lane) {
        loss[0] = value;
        total[0] += value;
    }
}
