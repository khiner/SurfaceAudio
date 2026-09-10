#include <metal_stdlib>
using namespace metal;
struct TraerBlock {
    uint Frames, Voices, Modes, Bands;
    float SampleRate;
};
struct TraerMode {
    float Frequency, OnsetDb, DecayDbPerSecond;
    uint EndFrame;
};
struct TraerBand {
    float OnsetDb, DecayDbPerSecond;
};
kernel void TraerResponse(constant TraerBlock &b [[buffer(0)]], device const TraerMode *modes [[buffer(1)]], device const TraerBand *bands [[buffer(2)]], device const float *noise [[buffer(3)]], device float *output [[buffer(4)]], uint2 index [[thread_position_in_grid]]) {
    if (index.x >= b.Frames || index.y >= b.Voices) return;
    const float time = float(index.x) / b.SampleRate;
    float sum = 0;
    for (uint mode = 0; mode < b.Modes; ++mode) {
        const auto m = modes[index.y * b.Modes + mode];
        if (index.x < m.EndFrame) {
            const float increment = m.Frequency / b.SampleRate;
            const float increment_error = fma(-increment, b.SampleRate, m.Frequency) / b.SampleRate;
            const float high = float(index.x) * increment;
            const float cycles = (high - floor(high)) + fma(float(index.x), increment, -high) + float(index.x) * increment_error;
            sum += pow(10.f, (m.OnsetDb - m.DecayDbPerSecond * time) / 20.f) * cos(6.283185307179586f * cycles);
        }
    }
    for (uint band = 0; band < b.Bands; ++band) {
        const auto n = bands[index.y * b.Bands + band];
        sum += pow(10.f, (n.OnsetDb - n.DecayDbPerSecond * time) / 20.f) * noise[(index.y * b.Bands + band) * b.Frames + index.x];
    }
    output[index.y * b.Frames + index.x] = sum;
}
