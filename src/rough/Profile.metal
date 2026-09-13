#include "core/Random.h"
using namespace metal;
using namespace surface_audio;
struct ProfileConstants {
    ulong Seed;
    uint Nodes, Size;
    float Spacing, Correlation, Rms;
};
kernel void RoughGaussianInput(constant ProfileConstants &c [[buffer(0)]], device float2 *values [[buffer(1)]], uint n [[thread_position_in_grid]]) {
    if (n >= c.Size) return;
    if (n >= c.Nodes) {
        values[n] = values[c.Size + n] = float2(0);
        return;
    }
    auto random = MakeIndexedRandom(c.Seed, n);
    const float x = (float(n) - .5f * float(c.Nodes - 1)) * c.Spacing / c.Correlation;
    values[n] = {c.Rms * Normal(random), 0};
    // Bergström rsgeng1D uses this centered filter with circular convolution.
    values[c.Size + n] = {exp(-2 * x * x), 0};
}
struct WhiteConstants {
    ulong Seed;
    uint Nodes;
    float Rms;
};
struct FilterConstants {
    uint Nodes, Period, Stride, Start, Count;
};
kernel void RoughWhiteNoise(constant WhiteConstants &c [[buffer(0)]], device float *values [[buffer(1)]], uint n [[thread_position_in_grid]]) {
    if (n >= c.Nodes) return;
    auto random = MakeIndexedRandom(c.Seed, n);
    values[n] = c.Rms * Normal(random);
}
kernel void RoughGaussianAxis(constant FilterConstants &c [[buffer(0)]], device const float *input [[buffer(1)]], device const float *taps [[buffer(2)]], device float *output [[buffer(3)]], uint n [[thread_position_in_grid]]) {
    if (n >= c.Nodes) return;
    const uint coordinate = (n / c.Stride) % c.Period, base = n - coordinate * c.Stride;
    float sum{};
    for (uint k = 0; k < c.Count; ++k) {
        const int position = int(coordinate) - int(c.Start + k);
        const uint sample = uint(position < 0 ? position + int(c.Period) : position);
        sum = fma(taps[k], input[base + sample * c.Stride], sum);
    }
    output[n] = sum;
}
