#include <metal_stdlib>
using namespace metal;
struct ImpactParameters {
    float Amplitude, Omega, Limit, Duration;
};
struct ImpactBlock {
    uint Frames, Voices, Sampling;
    float SampleRate;
};

kernel void ImpactForces(constant ImpactBlock &b [[buffer(0)]], device const ImpactParameters *settings [[buffer(1)]], device float *output [[buffer(2)]], uint2 id [[thread_position_in_grid]]) {
    if (id.x >= b.Frames || id.y >= b.Voices) return;
    const auto p = settings[id.y];
    const float begin = float(id.x) / b.SampleRate, end = min(float(id.x + 1) / b.SampleRate, p.Duration);
    float value = 0;
    if (begin < end) {
        if (b.Sampling == 0) {
            const float force = p.Amplitude * sin(p.Omega * begin);
            value = isinf(p.Limit) ? force : p.Limit * tanh(force / p.Limit);
        } else {
            const float half_width = (end - begin) * .5f, mid = (begin + end) * .5f;
            if (isinf(p.Limit)) value = 2 * p.Amplitude * sin(p.Omega * mid) * sin(p.Omega * half_width) / p.Omega * b.SampleRate;
            else {
                // Eight-point Gauss-Legendre integration over each clipped sample interval.
                constexpr float nodes[] = {.1834346425f, .5255324099f, .7966664774f, .9602898565f};
                constexpr float weights[] = {.3626837834f, .3137066459f, .2223810345f, .1012285363f};
                for (uint i = 0; i < 4; ++i) {
                    const float left = p.Amplitude * sin(p.Omega * (mid - half_width * nodes[i]));
                    const float right = p.Amplitude * sin(p.Omega * (mid + half_width * nodes[i]));
                    value += weights[i] * p.Limit * (tanh(left / p.Limit) + tanh(right / p.Limit));
                }
                value *= half_width * b.SampleRate;
            }
        }
    }
    output[id.y * b.Frames + id.x] = value;
}

struct ContactForceBlock {
    uint Frames;
    float Stiffness, Dissipation;
};
kernel void ContactForceMix(constant ContactForceBlock &p [[buffer(0)]], device const float *scraping [[buffer(1)]], device const float *elastic [[buffer(2)]], device const float *dissipative [[buffer(3)]], device float *output [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i < p.Frames) output[i] = scraping[i] + p.Stiffness * elastic[i] + p.Dissipation * dissipative[i];
}
