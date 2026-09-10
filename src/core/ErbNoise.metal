#include <metal_stdlib>
using namespace metal;
struct NoiseBlock {
    uint Frames, Taps, Bands;
};

kernel void ErbNoise(constant NoiseBlock &p [[buffer(0)]], device const float *white [[buffer(1)]], device const float *filters [[buffer(2)]], device float *output [[buffer(3)]], uint2 index [[thread_position_in_grid]]) {
    if (index.x >= p.Frames || index.y >= p.Bands) return;
    float sum = 0;
    for (uint tap = 0; tap < p.Taps; ++tap) sum += filters[index.y * p.Taps + tap] * white[index.x + tap];
    output[index.y * p.Frames + index.x] = sum;
}
