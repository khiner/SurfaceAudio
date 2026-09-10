#include <metal_stdlib>
using namespace metal;
#include "Continuous.h"
using namespace surface_audio::continuous;
kernel void ContinuousSynthesize(device const Parameters *parameters [[buffer(0)]], device State *states [[buffer(1)]], device float *output [[buffer(2)]], constant uint &frames [[buffer(3)]], uint voice [[thread_position_in_grid]]) {
    const Parameters p = parameters[voice];
    State s = states[voice];
    for (uint frame = 0; frame < frames; ++frame) output[voice * frames + frame] = Step(p, s);
    states[voice] = s;
}
