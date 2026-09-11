#include "core/Random.h"
#include "willemsen/Interaction.h"
using namespace surface_audio;
using namespace surface_audio::willemsen;
struct WillemsenDrive {
    float Velocity, NormalForce;
    uint Seed;
};
kernel void WillemsenRender(constant Model<float> &input [[buffer(0)]], device const WillemsenDrive *drives [[buffer(1)]], constant uint *shape [[buffer(2)]], device float *wave [[buffer(3)]], device float *final_state [[buffer(4)]], device float *residual [[buffer(5)]], device uint *failed [[buffer(6)]], uint voice [[thread_position_in_grid]]) {
    if (voice >= shape[0]) return;
    const Model<float> m = input;
    auto state = MakeState(m, drives[voice].Velocity);
    auto random = MakeRandom(drives[voice].Seed);
    float error{};
    uint failures{};
    for (uint i = 0; i < shape[1]; ++i) {
        const auto s = Step(m, state, drives[voice].Velocity, drives[voice].NormalForce, 2 * Uniform(random) - 1, 2e-6f);
        wave[voice * shape[1] + i] = s.Displacement;
        error = max(error, s.Residual);
        failures += s.Failed;
    }
    uint index = voice * (2 * (m.Intervals + 1) + 3);
    for (uint j = 0; j <= m.Intervals; ++j) final_state[index++] = state.U[j];
    for (uint j = 0; j <= m.Intervals; ++j) final_state[index++] = state.Previous[j];
    final_state[index++] = state.Z;
    final_state[index++] = state.Rate;
    final_state[index] = state.Velocity;
    residual[voice] = error;
    failed[voice] = failures;
}
