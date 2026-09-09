#include <metal_stdlib>
using namespace metal;

struct FiniteModalBlock {
    uint Frames, ForceFrames, Modes, Taps;
};

// Subtracting p^Taps times the delayed input truncates each exponential-sine response exactly.
kernel void FiniteModalConvolve(constant FiniteModalBlock &p [[buffer(0)]], device const float *coefficients [[buffer(1)]], device const float *force [[buffer(2)]], device const float *morph [[buffer(3)]], device float *output [[buffer(4)]], uint mode [[thread_position_in_grid]]) {
    if (mode >= p.Modes) return;
    const float pr = coefficients[mode], pi = coefficients[p.Modes + mode], kr = coefficients[2 * p.Modes + mode], ki = coefficients[3 * p.Modes + mode];
    const float a0 = coefficients[4 * p.Modes + mode], a1 = coefficients[5 * p.Modes + mode];
    float real = 0, imaginary = 0;
    for (uint frame = 0; frame < p.Frames; ++frame) {
        const float delayed = frame >= p.Taps ? force[frame - p.Taps] : 0;
        const float current = frame < p.ForceFrames ? force[frame] : 0;
        const float next_real = fma(pr, real, -pi * imaginary) + current - kr * delayed;
        imaginary = fma(pi, real, pr * imaginary) - ki * delayed;
        real = next_real;
        output[mode * p.Frames + frame] = imaginary * exp(mix(log(a0), log(a1), morph[min(frame, p.ForceFrames - 1)]));
    }
}
