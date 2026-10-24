#include <metal_stdlib>
#include "core/Random.h"
#include "rough/ContactGpuTypes.h"
using namespace metal;
using namespace surface_audio;
using namespace surface_audio::rough;
kernel void SelfAffineSpectrum(constant SurfaceSpectrumConstants &c [[buffer(0)]], device float2 *output [[buffer(1)]], uint2 i [[thread_position_in_grid]]) {
    if (i.x >= c.X || i.y >= c.Y) return;
    const uint index = i.y * c.X + i.x, opposite = ((c.Y - i.y) % c.Y) * c.X + (c.X - i.x) % c.X;
    const float x = float(i.x <= c.X / 2 ? int(i.x) : int(i.x) - int(c.X)) / (c.X * c.Spacing);
    const float y = float(i.y <= c.Y / 2 ? int(i.y) : int(i.y) - int(c.Y)) / (c.Y * c.Spacing);
    const float frequency = sqrt(x * x + y * y);
    if (!index || frequency < c.Minimum || frequency >= c.High) { output[index] = {}; return; }
    auto random = MakeIndexedRandom(c.Seed, min(index, opposite));
    const float phase = 6.283185307179586f * Uniform(random);
    const float amplitude = c.Scale * (frequency < c.Low ? 1.f : pow(frequency / c.Low, -.5f * c.Exponent));
    output[index] = amplitude * float2(cos(phase), index < opposite ? sin(phase) : -sin(phase));
}
kernel void SurfaceTranspose(constant SurfaceSpectrumConstants &c [[buffer(0)]], device const float2 *input [[buffer(1)]],
                             device float2 *output [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
    if (i.x >= c.X || i.y >= c.Y) return;
    float2 value{};
    if (c.X & (c.X - 1)) {
        const uint size = c.X / 5;
        for (uint branch = 0; branch < 5; ++branch) {
            const float phase = 6.283185307179586f * branch * i.x / c.X;
            const float2 z = input[(i.y * 5 + branch) * size + i.x % size], w{cos(phase), sin(phase)};
            value += float2(z.x * w.x - z.y * w.y, z.x * w.y + z.y * w.x) / 5;
        }
    } else value = input[i.y * c.X + i.x];
    output[i.x * c.Y + i.y] = value;
}
kernel void SurfaceRadixFiveInput(constant SurfaceSpectrumConstants &c [[buffer(0)]], device const float2 *input [[buffer(1)]],
                                  device float2 *output [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
    if (i.x < c.X && i.y < c.Y) output[(i.y * 5 + i.x % 5) * (c.X / 5) + i.x / 5] = input[i.y * c.X + i.x];
}
kernel void GeometricSeparation(constant SeparationConstants &c [[buffer(0)]], device const float *track [[buffer(1)]],
                                device const float *slider [[buffer(2)]], device float *output [[buffer(3)]],
                                uint offset [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float maxima[8];
    float value = -INFINITY;
    for (uint n = lane; n < c.SliderX * c.Y; n += 256)
        value = max(value, track[(n / c.SliderX) * c.TrackX + n % c.SliderX + c.Start + offset] - slider[n]);
    value = simd_max(value);
    if (!(lane % 32)) maxima[lane / 32] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 32) {
        value = simd_max(lane < 8 ? maxima[lane] : -INFINITY);
        if (!lane) output[c.Start + offset] = value;
    }
}
