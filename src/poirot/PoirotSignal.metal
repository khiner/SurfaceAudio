#include "core/Reduction.h"
#include <metal_stdlib>
using namespace metal;
#include "SignalGpuTypes.h"
using namespace surface_audio::poirot;
kernel void PoirotSignalSynthesize(device const SignalGpuControls *parameters [[buffer(0)]], device const SignalGpuMode *modes [[buffer(1)]], device float *state [[buffer(2)]], device float *output [[buffer(3)]], device const SignalGpuParameters &h [[buffer(4)]], uint voice [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]], uint width [[threads_per_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]]) {
    const SignalGpuControls p = parameters[voice];
    const uint index = voice * h.Modes + lane;
    const bool valid = lane < h.Modes;
    const SignalGpuMode m = valid ? modes[index] : SignalGpuMode{0, 0, INFINITY, 0, 0};
    float power = valid ? state[3 * index] : 0, upper = valid ? state[3 * index + 1] : 0, lower = valid ? state[3 * index + 2] : 0;
    const float step = 2 * M_PI_F / p.SampleRate, split_hz = modes[voice * h.Modes].Frequency / 3;
    threadgroup float sums[32], power_total;
    for (uint frame = 0; frame < h.Frames; ++frame) {
        const bool active = h.Frame + frame >= p.Activation;
        const float donated = active ? p.Lambda * max(power - m.Threshold, 0.f) : 0;
        const float donated_total = SumThreadgroup(donated, sums, width / 32, lane, simd_lane, simd_id);
        if (!lane) power_total = donated_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float total = power_total;
        const float split = active ? 1 - exp(-p.SplitSlope * max(total * p.PowerScale - p.SplitThreshold, 0.f)) : 0;
        const float c = split * (p.ShapeWeightedSplit > 0 ? m.Shape : m.Weight), b = sqrt(2 * power / (1 + c * c));
        if (p.ResetPhaseAtActivation > 0 && h.Frame + frame == p.Activation) {
            upper = step * (m.Frequency + c * split_hz);
            lower = step * (m.Frequency - split_hz);
        }
        const float sample = b * (sin(upper) + c * sin(lower));
        const float output_total = SumThreadgroup(sample, sums, width / 32, lane, simd_lane, simd_id);
        if (!lane) output[voice * h.Frames + frame] = output_total;
        upper += step * (m.Frequency + c * split_hz);
        upper -= floor((upper + M_PI_F) / (2 * M_PI_F)) * (2 * M_PI_F);
        lower += step * (m.Frequency - split_hz);
        lower -= floor((lower + M_PI_F) / (2 * M_PI_F)) * (2 * M_PI_F);
        power = max(0.f, power - donated + p.ReturnGain * m.Weight * total) * m.Loss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (valid) {
        state[3 * index] = power;
        state[3 * index + 1] = upper;
        state[3 * index + 2] = lower;
    }
}
