#include "Ren.h"
using namespace metal;
using namespace surface_audio::ren;

kernel void RenSource(constant Parameters &config [[buffer(0)]], device State *states [[buffer(1)]], const device Normal *normals [[buffer(2)]], const device float *roughness [[buffer(3)]], const device Contact *contacts [[buffer(4)]], device Impulses *output [[buffer(5)]], constant uint2 &block [[buffer(6)]], uint voice [[thread_position_in_grid]]) {
    if (voice >= block.x) return;
    const Parameters p = config;
    State s = states[voice];
    for (uint frame = 0; frame < block.y; ++frame) {
        const uint index = voice * block.y + frame;
        const Contact c = contacts[index];
        output[index] = Step(p, s, c, normals, roughness);
    }
    states[voice] = s;
}
