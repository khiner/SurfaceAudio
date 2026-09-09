#include "core/Reduction.h"
using namespace metal;

struct ContactFitBlock {
    uint Modes, ForceFrames, Frames, Taps, SampleRate, Groups;
};
inline float2 ContactMultiply(float2 a, float2 b) { return float2(fma(a.x, b.x, -a.y * b.y), fma(a.x, b.y, a.y * b.x)); }
inline float2 ContactPower(float2 pole, uint power) {
    float2 result = float2(1, 0);
    while (power) {
        if (power & 1) result = ContactMultiply(result, pole);
        pole = ContactMultiply(pole, pole);
        power >>= 1;
    }
    return result;
}

kernel void ContactFitSynthesize(constant ContactFitBlock &p [[buffer(0)]], device const float *parameters [[buffer(1)]], device const float2 *frequencies [[buffer(2)]], device const float *force [[buffer(3)]], device float *output [[buffer(4)]], device float *decay_derivative [[buffer(5)]], uint mode [[thread_position_in_grid]]) {
    if (mode >= p.Modes) return;
    const float amplitude = exp(parameters[mode]), inverse_decay = exp(-parameters[p.Modes + mode]) / float(p.SampleRate);
    const float2 pole = exp(-inverse_decay) * frequencies[mode], cutoff = ContactPower(pole, p.Taps);
    float2 state = 0, derivative = 0;
    for (uint frame = 0; frame < p.Frames; ++frame) {
        const float current = frame < p.ForceFrames ? force[frame] : 0;
        const float delayed = frame >= p.Taps && frame - p.Taps < p.ForceFrames ? force[frame - p.Taps] : 0;
        const float2 advanced = ContactMultiply(pole, state);
        // d pole/d log(tau) = pole/(rate*tau), including the finite-tail subtraction.
        derivative = ContactMultiply(pole, derivative) + inverse_decay * advanced - (float(p.Taps) * inverse_decay * delayed) * cutoff;
        state = advanced + float2(current, 0) - delayed * cutoff;
        output[mode * p.Frames + frame] = p.Taps == 1 ? 0 : amplitude * state.y;
        decay_derivative[mode * p.Frames + frame] = p.Taps == 1 ? 0 : amplitude * derivative.y;
    }
}

kernel void ContactFitDifferentiate(constant ContactFitBlock &p [[buffer(0)]], device const float *output [[buffer(1)]], device const float *decay_derivative [[buffer(2)]], device const float *adjoint [[buffer(3)]], device float2 *partial [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]]) {
    const uint frame = group.x * 256 + lane;
    const float2 value = frame < p.Frames ? adjoint[frame] * float2(output[group.y * p.Frames + frame], decay_derivative[group.y * p.Frames + frame]) : float2(0);
    threadgroup float2 sums[8];
    const float2 total = SumThreadgroup(value, sums, 8, lane, simd_lane, simd_id);
    if (!lane) partial[group.y * p.Groups + group.x] = total;
}

kernel void ContactFitReduce(constant ContactFitBlock &p [[buffer(0)]], device const float2 *partial [[buffer(1)]], device float *gradient [[buffer(2)]], uint mode [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]]) {
    float2 value = 0;
    for (uint group = lane; group < p.Groups; group += 256) value += partial[mode * p.Groups + group];
    threadgroup float2 sums[8];
    const float2 total = SumThreadgroup(value, sums, 8, lane, simd_lane, simd_id);
    if (!lane) {
        gradient[mode] = total.x;
        gradient[p.Modes + mode] = total.y;
    }
}
