#include "core/Reduction.h"
using namespace metal;

struct ResponseBlock {
    uint Frames, Groups, NoiseBands;
    float SampleRate;
};

// Compensated cycle reduction limits phase error in long impulse-response tails.
inline float ResponsePhase(float frequency, uint frame, float sample_rate) {
    const float sample = float(frame), product = frequency * sample;
    const float product_error = fma(frequency, sample, -product) + frequency * float(int(frame) - int(sample));
    const float cycles = product / sample_rate;
    const float remainder = (fma(-cycles, sample_rate, product) + product_error) / sample_rate;
    return 2 * M_PI_F * ((cycles - floor(cycles)) + remainder);
}

kernel void ResponseSynthesize(constant ResponseBlock &p [[buffer(0)]], device const float *parameters [[buffer(1)]], device const float *noise [[buffer(2)]], device float *output [[buffer(3)]], uint frame [[thread_position_in_grid]]) {
    if (frame >= p.Frames) return;
    const float t = float(frame) / p.SampleRate;
    float sum = 0;
    for (uint mode = 0; mode < 10; ++mode) {
        sum += pow(10.f, parameters[10 + mode] / 20 - 3 * t / parameters[20 + mode]) * sin(ResponsePhase(parameters[mode], frame, p.SampleRate));
        sum += pow(10.f, parameters[30 + mode] / 20 - 3 * t / parameters[30 + p.NoiseBands + mode]) * noise[mode * p.Frames + frame];
    }
    for (uint band = 10; band < p.NoiseBands; ++band)
        sum += pow(10.f, parameters[30 + band] / 20 - 3 * t / parameters[30 + p.NoiseBands + band]) * noise[band * p.Frames + frame];
    output[frame] = sum;
}

// Dispatch one group per parameter and 256-frame block, then reduce the partial gradients.
kernel void ResponseDifferentiate(constant ResponseBlock &p [[buffer(0)]], device const float *parameters [[buffer(1)]], device const float *noise [[buffer(2)]], device const float *adjoint [[buffer(3)]], device float *partial [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]]) {
    const uint frame = group.x * 256 + lane, parameter = group.y, mode = parameter < 30 ? parameter % 10 : (parameter - 30) % p.NoiseBands;
    float value = 0;
    if (frame < p.Frames) {
        const float t = float(frame) / p.SampleRate;
        const bool is_noise = parameter >= 30;
        const float rt = parameters[(is_noise ? 30 + p.NoiseBands : 20) + mode];
        const float envelope = pow(10.f, parameters[(is_noise ? 30 : 10) + mode] / 20 - 3 * t / rt);
        if (parameter < 10) value = envelope * 2 * M_PI_F * t * cos(ResponsePhase(parameters[mode], frame, p.SampleRate));
        else {
            value = envelope * (is_noise ? noise[mode * p.Frames + frame] : sin(ResponsePhase(parameters[mode], frame, p.SampleRate)));
            value *= (parameter < 20 || (parameter >= 30 && parameter < 30 + p.NoiseBands)) ? log(10.f) / 20 : log(10.f) * 3 * t / (rt * rt);
        }
        value *= adjoint[frame];
    }
    threadgroup float sums[8];
    const float total = SumThreadgroup(value, sums, 8, lane, simd_lane, simd_id);
    if (!lane) partial[parameter * p.Groups + group.x] = total;
}

kernel void ResponseReduce(constant ResponseBlock &p [[buffer(0)]], device const float *partial [[buffer(1)]], device float *gradient [[buffer(2)]], uint parameter [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]]) {
    float value = 0;
    for (uint index = lane; index < p.Groups; index += 256) value += partial[parameter * p.Groups + index];
    threadgroup float sums[8];
    const float total = SumThreadgroup(value, sums, 8, lane, simd_lane, simd_id);
    if (!lane) gradient[parameter] = total;
}
