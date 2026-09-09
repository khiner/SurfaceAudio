#include <metal_stdlib>
using namespace metal;
#include "Conan.h"

kernel void ConanSynthesize(device const surface_audio::conan::Parameters *parameters [[buffer(0)]], device surface_audio::conan::State *states [[buffer(1)]], device float *output [[buffer(2)]], constant uint &frame_count [[buffer(3)]], uint voice [[thread_position_in_grid]]) {
    auto parameters_local = parameters[voice];
    auto state = states[voice];
    for (uint frame = 0; frame < frame_count; ++frame) output[voice * frame_count + frame] = surface_audio::conan::Step(parameters_local, state);
    states[voice] = state;
}
