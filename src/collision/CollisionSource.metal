#include "Collision.h"
using namespace metal;
using namespace surface_audio::collision;
kernel void CollisionSource(const device float *data [[buffer(0)]], const device uint *indices [[buffer(1)]], device float *state [[buffer(2)]], device uint *active [[buffer(3)]], device Sample<float> *output [[buffer(4)]], constant Parameters<float> &config [[buffer(5)]], constant uint2 &block [[buffer(6)]], uint voice [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]], uint width [[threads_per_threadgroup]]) {
    if (voice >= block.x) return;
    threadgroup float shared[128];
    const auto p = config;
    for (uint frame = 0; frame < block.y; ++frame) {
        const auto sample = Step(p, data, indices, state + voice * StateSize(p), active + voice * p.Contacts, lane, width, shared);
        if (lane == 0) output[voice * block.y + frame] = sample;
    }
}
