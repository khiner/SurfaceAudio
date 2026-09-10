#include "falaize/Interaction.h"
using namespace surface_audio::falaize;
kernel void FalaizeRender(constant Model<float> &input [[buffer(0)]], device const float *drives [[buffer(1)]], constant uint *shape [[buffer(2)]], device float *wave [[buffer(3)]], device float *final_state [[buffer(4)]], device float *diagnostics [[buffer(5)]], device uint *failed [[buffer(6)]], uint voice [[thread_position_in_grid]]) {
    if (voice >= shape[0]) return;
    Model<float> model = input;
    State<float> state;
    const bool hammer = shape[2] != 0;
    if (hammer) state.HammerVelocity = drives[voice];
    float error{}, residual{};
    uint failures{};
    for (uint i = 0; i < shape[1]; ++i) {
        const auto sample = Step(model, state, drives[voice], hammer, 2e-6f);
        wave[voice * shape[1] + i] = sample.Velocity;
        error = max(error, abs(sample.BalanceError));
        residual = max(residual, sample.Residual);
        failures += sample.Failed;
    }
    uint offset = voice * (2 * model.Config.Modes + 4);
    for (uint i = 0; i < model.Config.Modes; ++i) final_state[offset++] = state.Displacement[i];
    for (uint i = 0; i < model.Config.Modes; ++i) final_state[offset++] = state.Velocity[i];
    final_state[offset++] = state.Elastic;
    final_state[offset++] = state.HammerVelocity;
    final_state[offset++] = state.ContactVelocity;
    final_state[offset] = state.Force;
    diagnostics[2 * voice] = error;
    diagnostics[2 * voice + 1] = residual;
    failed[voice] = failures;
}
