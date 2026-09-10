// SPDX-License-Identifier: GPL-3.0-only
#include "matusiak/Lumped.h"
using namespace surface_audio::matusiak;
kernel void MatusiakLumped(device const LumpedParameters<float> *parameters [[buffer(0)]], device float *output [[buffer(1)]], device float *energy [[buffer(2)]], device float *residual [[buffer(3)]], device uint *failures [[buffer(4)]], constant uint2 &shape [[buffer(5)]], uint voice [[thread_position_in_grid]]) {
    if (voice >= shape.x) return;
    const auto p = parameters[voice];
    LumpedState<float> state;
    float maximum_energy{}, maximum_residual{};
    uint failed{};
    for (uint i = 0; i < shape.y; ++i) {
        const float vb = min(p.BowVelocity, p.Acceleration * float(i) / p.SampleRate);
        const auto sample = StepLumped(p, state, vb, 1e-6f);
        output[voice * shape.y + i] = sample.Displacement;
        maximum_energy = max(maximum_energy, abs(sample.EnergyError));
        maximum_residual = max(maximum_residual, sample.Residual);
        failed += sample.Iterations == 100 || !isfinite(sample.Displacement);
    }
    energy[voice] = maximum_energy;
    residual[voice] = maximum_residual;
    failures[voice] = failed;
}
