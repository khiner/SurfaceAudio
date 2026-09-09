#include <metal_stdlib>
using namespace metal;
struct AgarwalAtlasBlock {
    uint Count, Levels;
    float HalfWidth, AlphaScale;
    uint RelativeCorrection;
};
kernel void AgarwalAtlasCurvature(constant AgarwalAtlasBlock &p [[buffer(0)]], device const float *raw [[buffer(1)]], device float *output [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index >= p.Count * p.Levels) return;
    const uint level = index / p.Count, x = index % p.Count;
    const float alpha = mix(.01f, .05f, float(level) / (p.Levels - 1));
    if (p.RelativeCorrection) {
        const float u = alpha * p.AlphaScale * raw[x], q = u * u;
        // tanh(u)/u - 1 for |u| <= .1, without subtracting nearly equal floats.
        output[index] = q * (-1.f / 3 + q * (2.f / 15 + q * (-17.f / 315 + q * (62.f / 2835 - q * 1382.f / 155925))));
        return;
    }
    const float sigma = max(p.HalfWidth * alpha / .03f * .4f, 1e-12f);
    const int radius = int(ceil(p.HalfWidth * alpha / .03f));
    float sum = 0, normalization = 0;
    for (int j = max(0, int(x) - radius); j <= min(int(p.Count) - 1, int(x) + radius); ++j) {
        const float d = float(j) - x, weight = exp(-.5f * d * d / (sigma * sigma));
        const float physical_alpha = alpha * p.AlphaScale;
        sum += weight * tanh(physical_alpha * raw[j]) / physical_alpha;
        normalization += weight;
    }
    output[index] = sum / normalization;
}
struct AgarwalSampleBlock {
    uint Count, Levels, Frames;
    float Spacing, Mass, Beta1, Beta2, Radius, Eccentricity, AlphaScale;
    uint VerticalMode;
};
struct AgarwalLocation {
    uint Cell;
    float Fraction;
};
inline float3 AgarwalSampleAtlas(constant AgarwalSampleBlock &p, device const float *atlas, device const float *motion, device const AgarwalLocation *coordinates, uint frame, bool polynomial) {
    const float alpha = mix(.05f, .01f, pow(clamp((motion[4 * frame + 2] - 1) / 5, 0.f, 1.f), .95f));
    const float gl = clamp((alpha - .01f) / .04f * (p.Levels - 1), 0.f, float(p.Levels - 1)), fraction = coordinates[frame].Fraction;
    const uint l0 = min(uint(gl), p.Levels - 2), x0 = coordinates[frame].Cell;
    if (polynomial) {
        const float distance = fraction * p.Spacing;
        float3 endpoints[2];
        for (uint side = 0; side < 2; ++side) {
            const uint base = (l0 + side) * 3 * p.Count + x0;
            const float h0 = atlas[base], s0 = atlas[base + p.Count], c0 = atlas[base + 2 * p.Count], delta = atlas[base + 2 * p.Count + 1] - c0;
            endpoints[side] = float3(h0 + s0 * distance + distance * distance * (.5f * c0 + delta * fraction / 6), s0 + distance * (c0 + .5f * delta * fraction), c0 + delta * fraction);
        }
        return mix(endpoints[0], endpoints[1], gl - l0);
    }
    float3 result;
    for (uint field = 0; field < 3; ++field) {
        const uint a = l0 * 3 * p.Count + field * p.Count + x0, b = a + 3 * p.Count;
        result[field] = mix(mix(atlas[a], atlas[a + 1], fraction), mix(atlas[b], atlas[b + 1], fraction), gl - l0);
    }
    return result;
}
inline void AgarwalWriteForce(uint frames, uint frame, float3 trajectory, float x, float velocity, float morph, float mass, float beta1, float beta2, float radius, float eccentricity, bool vertical_only, device float *output, device uint *status) {
    if (vertical_only) {
        output[frame] = mass * trajectory.z * velocity * velocity;
        output[frames + frame] = 0;
        output[2 * frames + frame] = 0;
        output[3 * frames + frame] = morph;
        status[frame] = 0;
        return;
    }
    const float rho = radius - eccentricity * cos(x / radius) + trajectory.x;
    if (rho < 0) {
        status[frame] = 2;
        return;
    }
    const float rho_dot = eccentricity / radius * velocity * sin(x / radius) + velocity * trajectory.y;
    output[frame] = beta1 * pow(abs(velocity * trajectory.y), beta2) + mass * trajectory.z * velocity * velocity;
    output[frames + frame] = rho * sqrt(rho);
    output[2 * frames + frame] = rho * sqrt(rho) * rho_dot;
    output[3 * frames + frame] = morph;
    status[frame] = 0;
}
kernel void AgarwalAtlasForcesV2(constant AgarwalSampleBlock &p [[buffer(0)]], device const float *atlas [[buffer(1)]], device const float *motion [[buffer(2)]], device float *output [[buffer(3)]], device uint *status [[buffer(4)]], device const AgarwalLocation *coordinates [[buffer(5)]], uint frame [[thread_position_in_grid]]) {
    if (frame >= p.Frames) return;
    if (coordinates[frame].Cell > p.Count - 2 || coordinates[frame].Fraction < 0 || coordinates[frame].Fraction > 1) {
        status[frame] = 1;
        return;
    }
    AgarwalWriteForce(p.Frames, frame, AgarwalSampleAtlas(p, atlas, motion, coordinates, frame, false), motion[4 * frame], motion[4 * frame + 1], motion[4 * frame + 3], p.Mass, p.Beta1, p.Beta2, p.Radius, p.Eccentricity, p.VerticalMode != 0, output, status);
}
kernel void AgarwalVerticalTrajectoryV2(constant AgarwalSampleBlock &p [[buffer(0)]], device const float *raw [[buffer(1)]], device const float *motion [[buffer(2)]], device float *output [[buffer(3)]], device uint *status [[buffer(4)]], device const AgarwalLocation *coordinates [[buffer(5)]], uint frame [[thread_position_in_grid]]) {
    if (frame >= p.Frames) return;
    const uint cell = coordinates[frame].Cell;
    const float fraction = coordinates[frame].Fraction;
    if (cell > p.Count - 2 || fraction < 0 || fraction > 1) {
        status[frame] = 1;
        return;
    }
    const float alpha = p.AlphaScale * mix(.05f, .01f, pow(clamp((motion[4 * frame + 2] - 1) / 5, 0.f, 1.f), .95f));
    output[frame] = 0;
    output[p.Frames + frame] = 0;
    output[2 * p.Frames + frame] = p.VerticalMode == 2 ? tanh(alpha * mix(raw[cell], raw[cell + 1], fraction)) / alpha : mix(tanh(alpha * raw[cell]) / alpha, tanh(alpha * raw[cell + 1]) / alpha, fraction);
    status[frame] = 0;
}
kernel void AgarwalAtlasTrajectoryV2(constant AgarwalSampleBlock &p [[buffer(0)]], device const float *atlas [[buffer(1)]], device const float *motion [[buffer(2)]], device float *output [[buffer(3)]], device uint *status [[buffer(4)]], device const AgarwalLocation *coordinates [[buffer(5)]], uint frame [[thread_position_in_grid]]) {
    if (frame >= p.Frames) return;
    if (coordinates[frame].Cell > p.Count - 2 || coordinates[frame].Fraction < 0 || coordinates[frame].Fraction > 1) {
        status[frame] = 1;
        return;
    }
    const float3 value = AgarwalSampleAtlas(p, atlas, motion, coordinates, frame, true);
    for (uint field = 0; field < 3; ++field) output[field * p.Frames + frame] = value[field];
    status[frame] = 0;
}
struct AgarwalTemporalBlock {
    uint Frames, SampleRate;
    float HalfWidth, Mass, Beta1, Beta2, Radius, Eccentricity;
    uint VerticalMode;
};
kernel void AgarwalTemporalForcesV2(constant AgarwalTemporalBlock &p [[buffer(0)]], device const float *trajectory [[buffer(1)]], device const float *motion [[buffer(2)]], device float *output [[buffer(3)]], device uint *status [[buffer(4)]], device float *filtered [[buffer(5)]], uint frame [[thread_position_in_grid]]) {
    if (frame >= p.Frames) return;
    const float alpha = mix(.05f, .01f, pow(clamp((motion[4 * frame + 2] - 1) / 5, 0.f, 1.f), .95f));
    const float half_width = p.HalfWidth * (float(p.SampleRate) / 44100) * alpha / .03f, sigma = max(.4f * half_width, 1e-12f);
    const int radius = int(ceil(half_width));
    float3 value = 0;
    float normalization = 0;
    for (int m = max(0, int(frame) - radius); m <= min(int(p.Frames) - 1, int(frame) + radius); ++m) {
        const float d = float(m) - frame, weight = exp(-.5f * d * d / (sigma * sigma));
        value += weight * float3(trajectory[m], trajectory[p.Frames + m], trajectory[2 * p.Frames + m]);
        normalization += weight;
    }
    value /= normalization;
    for (uint field = 0; field < 3; ++field) filtered[field * p.Frames + frame] = value[field];
    AgarwalWriteForce(p.Frames, frame, value, motion[4 * frame], motion[4 * frame + 1], motion[4 * frame + 3], p.Mass, p.Beta1, p.Beta2, p.Radius, p.Eccentricity, p.VerticalMode != 0, output, status);
}

struct AgarwalDenseVerticalBlock {
    uint Frames, VerticalMode;
    float AlphaScale;
};
kernel void AgarwalDenseVerticalCurvature(constant AgarwalDenseVerticalBlock &p [[buffer(0)]], device const float *raw [[buffer(1)]], device const AgarwalLocation *coordinates [[buffer(2)]], device const float *normal [[buffer(3)]], device float *curvature [[buffer(4)]], uint frame [[thread_position_in_grid]]) {
    if (frame >= p.Frames) return;
    const uint cell = coordinates[frame].Cell;
    const float fraction = coordinates[frame].Fraction;
    const float alpha = p.AlphaScale * mix(.05f, .01f, pow(clamp((normal[frame] - 1) / 5, 0.f, 1.f), .95f));
    curvature[frame] = p.VerticalMode == 2 ? tanh(alpha * mix(raw[cell], raw[cell + 1], fraction)) / alpha : mix(tanh(alpha * raw[cell]) / alpha, tanh(alpha * raw[cell + 1]) / alpha, fraction);
}
struct AgarwalDownsampleVerticalBlock {
    uint Frames, DenseFrames, Factor, SampleRate;
    float HalfWidth, Mass, SigmaRatio;
};
kernel void AgarwalDownsampleVerticalForceV2(constant AgarwalDownsampleVerticalBlock &p [[buffer(0)]], device const float *curvature [[buffer(1)]], device const float *motion [[buffer(2)]], device float *force [[buffer(3)]], device float *filtered [[buffer(4)]], uint frame [[thread_position_in_grid]]) {
    if (frame >= p.Frames) return;
    const float alpha = mix(.05f, .01f, pow(clamp((motion[4 * frame + 2] - 1) / 5, 0.f, 1.f), .95f));
    const float width = p.HalfWidth * (float(p.SampleRate) / 44100) * alpha / .03f * p.Factor, sigma = max(p.SigmaRatio * width, 1e-12f);
    const int center = int(frame * p.Factor), radius = int(ceil(width));
    float sum = 0, normalization = 0;
    // Relative integer offsets preserve the Gaussian at large dense-frame indices.
    for (int d = -min(center, radius); d <= min(int(p.DenseFrames) - 1 - center, radius); ++d) {
        const float distance = float(d), weight = exp(-.5f * distance * distance / (sigma * sigma));
        sum += weight * curvature[center + d];
        normalization += weight;
    }
    const float value = sum / normalization, velocity = motion[4 * frame + 1];
    filtered[frame] = value;
    force[frame] = p.Mass * value * velocity * velocity;
}

struct AgarwalDownsampleFullBlock {
    uint Frames, DenseFrames, Factor, SampleRate;
    float HalfWidth, Mass, Beta1, Beta2, Radius, Eccentricity, SigmaRatio;
};
kernel void AgarwalDownsampleFullForce(constant AgarwalDownsampleFullBlock &p [[buffer(0)]], device const float *trajectory [[buffer(1)]], device const float *motion [[buffer(2)]], device float *output [[buffer(3)]], device uint *status [[buffer(4)]], device float *filtered [[buffer(5)]], uint frame [[thread_position_in_grid]]) {
    if (frame >= p.Frames) return;
    const float alpha = mix(.05f, .01f, pow(clamp((motion[4 * frame + 2] - 1) / 5, 0.f, 1.f), .95f));
    const float width = p.HalfWidth * (float(p.SampleRate) / 44100) * alpha / .03f * p.Factor, sigma = max(p.SigmaRatio * width, 1e-12f);
    const int center = int(frame * p.Factor), radius = int(ceil(width));
    float3 value = 0;
    float normalization = 0;
    for (int d = -min(center, radius); d <= min(int(p.DenseFrames) - 1 - center, radius); ++d) {
        const int m = center + d;
        const float distance = float(d), weight = exp(-.5f * distance * distance / (sigma * sigma));
        value += weight * float3(trajectory[m], trajectory[p.DenseFrames + m], trajectory[2 * p.DenseFrames + m]);
        normalization += weight;
    }
    value /= normalization;
    for (uint field = 0; field < 3; ++field) filtered[field * p.Frames + frame] = value[field];
    AgarwalWriteForce(p.Frames, frame, value, motion[4 * frame], motion[4 * frame + 1], motion[4 * frame + 3], p.Mass, p.Beta1, p.Beta2, p.Radius, p.Eccentricity, false, output, status);
}

struct AgarwalRetainBlock {
    uint Frames, DenseFrames, Factor, Fields;
};
kernel void AgarwalRetainFields(constant AgarwalRetainBlock &p [[buffer(0)]], device const float *input [[buffer(1)]], device float *output [[buffer(2)]], uint2 index [[thread_position_in_grid]]) {
    if (index.x >= p.Frames || index.y >= p.Fields) return;
    output[index.y * p.Frames + index.x] = input[index.y * p.DenseFrames + index.x * p.Factor];
}
