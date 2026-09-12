#include "core/ExtendedFloat.h"
using namespace metal;
using namespace surface_audio;
struct NakatsukaScene {
    uint Columns, Rows, Iterations, ShiftedAdhesion;
    float SampleRate, Spacing, InverseMass, Speed, Gravity, Damping, Stretch, Adhesion, AdhesionDistance, SphereRadius, InitialHeight, ListenerHeight, MotionStart, MotionDuration, SettlingTime, OriginY, MotionRise, MotionFall;
};
float NakatsukaTravel(float time, float ramp) {
    if (time <= 0) return 0;
    if (time >= ramp) return time - .5f * ramp;
    const float u = time / ramp;
    return ramp * u * u * u * (1 - .5f * u);
}
kernel void NakatsukaSimulate(constant NakatsukaScene &s [[buffer(0)]], constant uint &frames [[buffer(1)]], device float4 *output [[buffer(2)]], device float4 *final [[buffer(3)]], uint index [[thread_index_in_threadgroup]]) {
    threadgroup float4 positions[1024];
    const uint count = s.Columns * s.Rows;
    const bool active = index < count;
    const uint col = index % s.Columns, row = index / s.Columns;
    const float3 initial = float3((float(col) - float(s.Columns - 1) * .5f) * s.Spacing, (float(row) - float(s.Rows - 1) * .5f) * s.Spacing + s.OriginY, s.InitialHeight);
    const float dt = 1 / s.SampleRate, weight = row == 0 ? 0 : s.InverseMass;
    float3 position = initial, velocity = 0, anchor = 0;
    bool attached = false;
    const uint settling = uint(s.SettlingTime * s.SampleRate);
    for (uint step = 0; step < settling + frames; ++step) {
        const float elapsed = max((float(step) - settling + 1) * dt - s.MotionStart, 0.f);
        const float travel = s.Speed * (NakatsukaTravel(elapsed, s.MotionRise) - (s.MotionDuration > 0 ? NakatsukaTravel(elapsed - s.MotionDuration + s.MotionFall, s.MotionFall) : 0));
        float3 next = weight == 0 ? initial - float3(0, travel, 0) : position + dt * (velocity + float3(0, 0, -s.Gravity * dt));
        if (active && weight > 0) {
            const float distance = s.SphereRadius > 0 ? length(next) - s.SphereRadius : next.z;
            const float3 normal = s.SphereRadius > 0 ? normalize(next) : float3(0, 0, 1);
            if (attached && (length(next - anchor) > 4 * s.AdhesionDistance || distance > 4 * s.AdhesionDistance)) attached = false;
            if (!attached && distance <= s.AdhesionDistance) {
                anchor = next - distance * normal;
                attached = true;
            }
        }
        if (active) positions[index] = float4(next, weight);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint iteration = 0; iteration < s.Iterations; ++iteration) {
            if (active) {
                const int neighbors[4] = {col > 0 ? int(index) - 1 : -1, col + 1 < s.Columns ? int(index) + 1 : -1, row > 0 ? int(index) - int(s.Columns) : -1, row + 1 < s.Rows ? int(index) + int(s.Columns) : -1};
                float3 correction = 0;
                for (uint i = 0; i < 4; ++i)
                    if (neighbors[i] >= 0) {
                        const float4 neighbor = positions[neighbors[i]];
                        const float3 delta = next - neighbor.xyz;
                        const float distance = length(delta), total = weight + neighbor.w;
                        if (distance > 0 && total > 0) correction -= weight / total * (distance - s.Spacing) * delta / distance;
                    }
                next += .25f * s.Stretch * correction;
                if (weight > 0) {
                    if (attached) {
                        const float3 delta = next - anchor;
                        const float distance = length(delta);
                        if (distance > s.AdhesionDistance && distance < 4 * s.AdhesionDistance) {
                            const float change = s.ShiftedAdhesion ? distance * (1 - distance / s.AdhesionDistance) : distance;
                            next += s.Adhesion * max(change, -distance) * delta / distance;
                        }
                    }
                    const float distance = s.SphereRadius > 0 ? length(next) - s.SphereRadius : next.z;
                    const float3 normal = s.SphereRadius > 0 ? normalize(next) : float3(0, 0, 1);
                    if (distance < 0) next -= distance * normal;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active) positions[index] = float4(next, weight);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (active) {
            const float distance = s.SphereRadius > 0 ? length(next) - s.SphereRadius : next.z;
            const float3 delta = next - position;
            velocity = delta / dt * exp(-s.Damping * dt);
            const float contact = distance <= 4 * s.AdhesionDistance ? 1.f : 0.f;
            const float3 normal = s.SphereRadius > 0 ? normalize(next) : float3(0, 0, 1);
            if (step >= settling) output[index * frames + step - settling] = float4(length(delta - dot(delta, normal) * normal) / dt, length(delta), length(next - float3(0, 0, s.ListenerHeight)), contact);
            position = next;
        }
    }
    if (active) final[index] = float4(position, weight);
}
struct NakatsukaBlock {
    uint Frames, Patches, Modes;
    float SampleRate, AirDensity, SoundSpeed, Attenuation;
};
kernel void NakatsukaRender(constant NakatsukaBlock &p [[buffer(0)]], device const float2 *frequency [[buffer(1)]], device const float4 *trajectory [[buffer(2)]], device float *output [[buffer(3)]], uint patch [[thread_position_in_grid]]) {
    if (patch >= p.Patches) return;
    float2 phase[64]{};
    const float2 timestep = frequency[p.Patches * p.Modes], inverse_speed = frequency[p.Patches * p.Modes + 1], turn = frequency[p.Patches * p.Modes + 2], nyquist = frequency[p.Patches * p.Modes + 3];
    for (uint frame = 0; frame < p.Frames; ++frame) {
        const float4 sample = trajectory[patch * p.Frames + frame];
        const float2 scale = MultiplyExtended(MultiplyExtended(float2(sample.x, 0), float2(sample.x, 0)), inverse_speed);
        float pressure = 0;
        for (uint mode = 0; mode < p.Modes; ++mode) {
            const float2 angular = MultiplyExtended(frequency[patch * p.Modes + mode], scale);
            const float omega = angular.x + angular.y;
            const float2 next = AddExtended(phase[mode], MultiplyExtended(angular, timestep));
            phase[mode] = AddExtended(next, -MultiplyExtended(turn, float2(floor(next.x / (2 * M_PI_F)), 0)));
            if (AddExtended(angular, -nyquist).x >= 0) continue;
            pressure -= p.AirDensity * (sample.y * sample.w / p.Modes) * omega * omega * exp(-p.Attenuation) / (4 * M_PI_F * sample.z) * cos((phase[mode].x - omega * sample.z / p.SoundSpeed) + phase[mode].y);
        }
        output[patch * p.Frames + frame] = pressure;
    }
}
