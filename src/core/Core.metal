#include "Random.h"
#include "Reduction.h"
using namespace metal;

struct FirBlock {
    uint TapCount, FrameCount;
};
struct FixedFirBlock {
    uint Inputs, Taps, Offset, Frames;
};
kernel void FixedFirConvolve(constant FixedFirBlock &p [[buffer(0)]], device const float *response [[buffer(1)]], device const float *excitation [[buffer(2)]], device float *output [[buffer(3)]], uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]], uint lanes [[threads_per_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]]) {
    if (group >= p.Frames) return;
    const uint frame = p.Offset + group;
    const uint begin = frame >= p.Inputs ? frame - p.Inputs + 1 : 0, end = min(p.Taps, frame + 1);
    float sum = 0;
    for (uint lag = begin + lane; lag < end; lag += lanes) sum = fma(response[lag], excitation[frame - lag], sum);
    threadgroup float partials[32];
    const float total = SumThreadgroup(sum, partials, (lanes + 31) / 32, lane, simd_lane, simd_id);
    if (!lane) output[frame] = total;
}
kernel void FirConvolve(constant FirBlock &p [[buffer(0)]], device const float *coefficients [[buffer(1)]], device const float *excitation [[buffer(2)]], device float *output [[buffer(3)]], uint frame [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]], uint lanes [[threads_per_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]]) {
    if (frame >= p.FrameCount) return;
    float sum = 0;
    for (uint lag = lane; lag < p.TapCount; lag += lanes) sum = fma(coefficients[frame * p.TapCount + lag], excitation[p.TapCount - 1 + frame - lag], sum);
    threadgroup float partials[32];
    const float total = SumThreadgroup(sum, partials, (lanes + 31) / 32, lane, simd_lane, simd_id);
    if (!lane) output[frame] = total;
}

kernel void RandomSequence(device uint *output [[buffer(0)]], constant uint &count [[buffer(1)]], uint lane [[thread_position_in_grid]]) {
    auto state = surface_audio::MakeRandom(42, lane);
    for (uint i = 0; i < count; ++i) output[lane * count + i] = surface_audio::NextRandom(state);
}

struct MixBlock {
    uint Voices, Frames, Stride, Field;
    float Gain;
};
kernel void MixVoices(constant MixBlock &p [[buffer(0)]], device const float *input [[buffer(1)]], device float *output [[buffer(2)]], uint frame [[thread_position_in_grid]]) {
    if (frame >= p.Frames) return;
    float sum = 0;
    for (uint voice = 0; voice < p.Voices; ++voice) sum += input[(voice * p.Frames + frame) * p.Stride + p.Field];
    output[frame] = sum * p.Gain;
}
