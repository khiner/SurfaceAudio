// SPDX-License-Identifier: GPL-3.0-or-later
#include "SdtGpu.h"
using namespace metal;
using namespace surface_audio::sdt;

float2 AddCompensated(float value, float error, float increment) {
    const float adjusted = increment - error;
    const float sum = value + adjusted;
    return {sum, (sum - value) - adjusted};
}

float4 NextState(GpuMode mode, GpuModeState state, float force) {
    const float2 position = AddCompensated(state.Position, state.PositionError, mode.PDelta * state.Position + mode.PFromV * state.Velocity + mode.PFromForce * force);
    const float2 velocity = AddCompensated(state.Velocity, state.VelocityError, mode.VFromP * state.Position + mode.VDelta * state.Velocity + mode.VFromForce * force);
    return {position.x, velocity.x, position.y, velocity.y};
}

bool ValidPrediction(device const GpuMode *modes, device const GpuModeState *states, uint offset, uint count, float force) {
    if (!isfinite(force)) return false;
    for (uint index = offset; index < offset + count; ++index) {
        const float4 predicted = NextState(modes[index], states[index], states[index].Force + force * modes[index].ForceGain);
        if (!all(isfinite(predicted)) || abs(predicted.x) >= 10000) return false;
    }
    return true;
}

ContactEnergyT<float> ContactEnergy(device const GpuMode *modes, device const GpuModeState *states, uint offset, uint count) {
    ContactEnergyT<float> energy;
    for (uint index = offset; index < offset + count; ++index) {
        const auto mode = modes[index];
        const auto state = states[index];
        const float4 next = NextState(mode, state, state.Force);
        const float dp = mode.PFromForce * mode.ForceGain, dv = mode.VFromForce * mode.ForceGain;
        energy.Free += 0.5f * (mode.Stiffness * next.x * next.x + mode.Mass * next.y * next.y) * mode.ContactGain;
        energy.Linear += (mode.Stiffness * next.x * dp + mode.Mass * next.y * dv) * mode.ContactGain;
        energy.Quadratic += 0.5f * (mode.Stiffness * dp * dp + mode.Mass * dv * dv) * mode.ContactGain;
    }
    return energy;
}

float3 Pickup(device const GpuMode *modes, device const GpuModeState *states, uint offset, uint count) {
    float3 value = 0;
    for (uint index = offset; index < offset + count; ++index) {
        const auto mode = modes[index];
        const auto state = states[index];
        value += float3(state.Position * mode.ContactGain, state.Velocity * mode.ContactGain, state.Position * mode.OutputGain);
    }
    return value;
}

void ApplyExternal(device const GpuMode *modes, device GpuModeState *states, uint offset, uint count, float force) {
    for (uint index = offset; index < offset + count; ++index) states[index].Force = force * modes[index].ForceGain;
}

void Advance(device const GpuMode *modes, device GpuModeState *states, uint offset, uint count, float force) {
    for (uint index = offset; index < offset + count; ++index) {
        const auto mode = modes[index];
        auto state = states[index];
        const float4 next = NextState(mode, state, state.Force + force * mode.ForceGain);
        state.Velocity = next.y;
        state.PositionError = next.z;
        state.VelocityError = next.w;
        state.PreviousPosition = state.Position;
        state.Position = next.x;
        state.Force = 0;
        states[index] = state;
    }
}

kernel void SdtContacts(constant GpuDispatch &dispatch [[buffer(0)]], device const GpuContact *contacts [[buffer(1)]], device GpuContactState *contact_states [[buffer(2)]], device const GpuMode *modes [[buffer(3)]], device GpuModeState *mode_states [[buffer(4)]], device const GpuInput *inputs [[buffer(5)]], device GpuOutput *outputs [[buffer(6)]], uint index [[thread_position_in_grid]]) {
    if (index >= dispatch.Contacts) return;
    const auto contact = contacts[index];
    auto state = contact_states[index];
    for (uint frame = 0; frame < dispatch.Frames; ++frame) {
        const uint sample = index * dispatch.Frames + frame;
        if (state.Error != GpuContactError::None) {
            outputs[sample] = {};
            continue;
        }
        const auto previous_state = state;
        const auto input = inputs[sample];
        ApplyExternal(modes, mode_states, contact.Offset0, contact.Count0, input.External0);
        ApplyExternal(modes, mode_states, contact.Offset1, contact.Count1, input.External1);
        const float3 first = Pickup(modes, mode_states, contact.Offset0, contact.Count0);
        const float3 second = Pickup(modes, mode_states, contact.Offset1, contact.Count1);
        float force = 0;
        if (contact.Kind == GpuContactKind::Impact) {
            const float compression = second.x - first.x;
            if (compression <= 0) state.Energy = 0;
            force = ImpactForce(contact.Impact, compression, second.y - first.y);
        } else if (contact.Kind == GpuContactKind::Friction) {
            state.Energy = 0;
            force = FrictionForce(contact.Friction, state.Friction, second.y - first.y, input.Noise, dispatch.TimeStep);
        }
        if (!isfinite(state.Energy) || !ValidPrediction(modes, mode_states, contact.Offset0, contact.Count0, 0) || !ValidPrediction(modes, mode_states, contact.Offset1, contact.Count1, 0) || !ValidPrediction(modes, mode_states, contact.Offset0, contact.Count0, force) || !ValidPrediction(modes, mode_states, contact.Offset1, contact.Count1, -force)) {
            state = previous_state;
            state.Error = GpuContactError::PredictionDomain;
            ApplyExternal(modes, mode_states, contact.Offset0, contact.Count0, 0);
            ApplyExternal(modes, mode_states, contact.Offset1, contact.Count1, 0);
            outputs[sample] = {};
            continue;
        }
        const auto energy0 = ContactEnergy(modes, mode_states, contact.Offset0, contact.Count0);
        const auto energy1 = ContactEnergy(modes, mode_states, contact.Offset1, contact.Count1);
        if (!all(isfinite(float3(energy0.Free + energy1.Free, energy0.Linear - energy1.Linear, energy0.Quadratic + energy1.Quadratic)))) {
            state = previous_state;
            state.Error = GpuContactError::PredictionDomain;
            ApplyExternal(modes, mode_states, contact.Offset0, contact.Count0, 0);
            ApplyExternal(modes, mode_states, contact.Offset1, contact.Count1, 0);
            outputs[sample] = {};
            continue;
        }
        force = LimitContactForce(ContactEnergyT<float>{energy0.Free + energy1.Free, energy0.Linear - energy1.Linear, energy0.Quadratic + energy1.Quadratic}, state.Energy, force);
        Advance(modes, mode_states, contact.Offset0, contact.Count0, force);
        Advance(modes, mode_states, contact.Offset1, contact.Count1, -force);
        const float3 output0 = Pickup(modes, mode_states, contact.Offset0, contact.Count0);
        const float3 output1 = Pickup(modes, mode_states, contact.Offset1, contact.Count1);
        outputs[sample] = {force, output0.x, output1.x, output0.y, output1.y, output0.z, output1.z};
    }
    contact_states[index] = state;
}

kernel void SdtSurfaceForces(constant GpuDispatch &dispatch [[buffer(0)]], device const GpuSurfaceParameters *parameters [[buffer(1)]], device GpuSurfaceState *states [[buffer(2)]], device const float *heights [[buffer(3)]], device GpuInput *outputs [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    if (index >= dispatch.Contacts) return;
    const auto parameters0 = parameters[index];
    auto state = states[index];
    for (uint frame = 0; frame < dispatch.Frames; ++frame) {
        const uint sample = index * dispatch.Frames + frame;
        outputs[sample].External0 = parameters0.Kind == GpuSurfaceKind::Rolling ? StepRolling(parameters0.Rolling, state.Rolling, heights[sample]) : StepScraping(parameters0.Scraping, state.Scraping, heights[sample]);
    }
    states[index] = state;
}
