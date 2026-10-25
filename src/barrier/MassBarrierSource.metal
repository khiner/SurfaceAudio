#include "Barrier.h"
using namespace metal;
using namespace surface_audio::barrier;

kernel void MassBarrierSource(const device Parameters<float> *parameters [[buffer(0)]], device State<float> *states [[buffer(1)]], const device Force<float> *input [[buffer(2)]], device Sample<float> *output [[buffer(3)]], constant uint2 &block [[buffer(4)]], uint voice [[thread_position_in_grid]]) {
    if (voice >= block.x) return;
    const auto p = parameters[voice];
    auto s = states[voice];
    for (uint n = 0; n < block.y; ++n) {
        const uint index = voice * block.y + n;
        const auto force = input[index];
        output[index] = Step(p, s, force);
    }
    states[voice] = s;
}
