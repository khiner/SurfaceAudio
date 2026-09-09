#include <metal_stdlib>
using namespace metal;

struct AgarwalImpulseParameters {
    uint TapCount, FrameCount, SurfaceModeCount, ObjectModeCount;
    float SampleRate, ObjectGain;
};

float AgarwalLogMix(float a, float b, float morph) {
    if (morph == 0) return a;
    if (morph == 1) return b;
    return exp(mix(log(a), log(b), morph));
}

// Dispatch FrameCount * TapCount threads. Each output IR belongs to the current location.
kernel void AgarwalBuildImpulseResponses(
    constant AgarwalImpulseParameters &params [[buffer(0)]],
    device const float *frequencies [[buffer(1)]],
    device const float *amplitudes [[buffer(2)]],
    device const float *morphs [[buffer(3)]],
    device float *coefficients [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= params.FrameCount * params.TapCount) return;
    const uint frame = index / params.TapCount, lag = index % params.TapCount;
    const float morph = morphs[frame], time = float(lag) / params.SampleRate;
    float value = 0;
    for (uint mode = 0; mode < params.SurfaceModeCount; ++mode) {
        const float frequency = AgarwalLogMix(frequencies[mode], frequencies[params.SurfaceModeCount + mode], morph);
        const uint a0 = mode * params.TapCount + lag, a1 = (params.SurfaceModeCount + mode) * params.TapCount + lag;
        value += AgarwalLogMix(amplitudes[a0], amplitudes[a1], morph) * sin(2 * M_PI_F * frequency * time);
    }
    for (uint mode = 0; mode < params.ObjectModeCount; ++mode) {
        const uint offset = 2 * params.SurfaceModeCount + mode;
        value += params.ObjectGain * amplitudes[offset * params.TapCount + lag] * sin(2 * M_PI_F * frequencies[offset] * time);
    }
    coefficients[index] = value;
}

struct AgarwalTrajectoryParameters {
    uint Width, Height, FrameCount, Rolling;
    float SpacingX, SpacingY;
    float NormalMin, NormalMax, AlphaMin, AlphaMax, Exponent, ConstantAlpha;
    float ReferenceAlpha, GaussianHalfWidth, GaussianSigmaRatio;
    float Mass, Beta1, Beta2;
    float Radius, Eccentricity, Stiffness, Dissipation;
};

float AgarwalHeight(device const float *heights, uint width, uint fixed, uint index, bool along_x) {
    return heights[along_x ? fixed * width + index : index * width + fixed];
}

float AgarwalCurvature(device const float *heights, uint width, uint count, uint fixed, uint index, bool along_x, float spacing, float alpha) {
    const uint center = clamp(index, 1u, count - 2);
    const float curvature = (AgarwalHeight(heights, width, fixed, center + 1, along_x) - 2 * AgarwalHeight(heights, width, fixed, center, along_x) + AgarwalHeight(heights, width, fixed, center - 1, along_x)) / (spacing * spacing);
    return alpha == 0 ? curvature : tanh(alpha * curvature) / alpha;
}

float AgarwalSmoothedCurvature(device const float *heights, uint width, uint count, uint fixed, uint index, bool along_x, float spacing, float alpha, float half_width, float sigma) {
    if (half_width <= 0) return AgarwalCurvature(heights, width, count, fixed, index, along_x, spacing, alpha);
    const uint radius = uint(ceil(half_width)), begin = index > radius ? index - radius : 0, end = index + min(radius, count - 1 - index);
    float sum = 0, weight_sum = 0;
    for (uint j = begin; j <= end; ++j) {
        const float distance = (float(j) - float(index)) / sigma, weight = exp(-0.5f * distance * distance);
        sum += weight * AgarwalCurvature(heights, width, count, fixed, j, along_x, spacing, alpha);
        weight_sum += weight;
    }
    return sum / weight_sum;
}

float3 AgarwalSampleAxis(device const float *heights, device const float *boundary_slopes, constant AgarwalTrajectoryParameters &params, uint fixed, bool along_x, float position, float alpha) {
    const uint count = along_x ? params.Width : params.Height;
    const float spacing = along_x ? params.SpacingX : params.SpacingY;
    const float half_width = params.GaussianHalfWidth * alpha / params.ReferenceAlpha, sigma = max(half_width * params.GaussianSigmaRatio, 1e-12f);
    float z = AgarwalHeight(heights, params.Width, fixed, 0, along_x);
    float slope = boundary_slopes[along_x ? fixed : params.Height + fixed];
    float c0 = AgarwalSmoothedCurvature(heights, params.Width, count, fixed, 0, along_x, spacing, alpha, half_width, sigma);
    const uint end = min(uint(position / spacing), count - 2);
    for (uint i = 0; i <= end; ++i) {
        const float c1 = AgarwalSmoothedCurvature(heights, params.Width, count, fixed, i + 1, along_x, spacing, alpha, half_width, sigma);
        const float distance = i == end ? position - float(i) * spacing : spacing, derivative = (c1 - c0) / spacing;
        z += slope * distance + 0.5f * c0 * distance * distance + derivative * distance * distance * distance / 6;
        slope += c0 * distance + 0.5f * derivative * distance * distance;
        if (i == end) return float3(z, slope, c0 + derivative * distance);
        c0 = c1;
    }
    return float3(0);
}

// One independent trajectory and force per thread. Trajectory output is five SoA planes.
kernel void AgarwalPrepareForces(
    constant AgarwalTrajectoryParameters &params [[buffer(0)]],
    device const float *heights [[buffer(1)]],
    device const float *motion [[buffer(2)]],
    device float *trajectory [[buffer(3)]],
    device float *force [[buffer(4)]],
    device uint *status [[buffer(5)]],
    device const float *boundary_slopes [[buffer(6)]],
    uint frame [[thread_position_in_grid]]
) {
    if (frame >= params.FrameCount) return;
    const uint count = params.FrameCount;
    const float x = motion[frame], y = motion[count + frame], vx = motion[2 * count + frame], vy = motion[3 * count + frame], normal = motion[4 * count + frame];
    const float weight = params.NormalMin == params.NormalMax ? 0 : pow(clamp((normal - params.NormalMin) / (params.NormalMax - params.NormalMin), 0.f, 1.f), params.Exponent);
    const float alpha = params.NormalMin == params.NormalMax ? params.ConstantAlpha : mix(params.AlphaMax, params.AlphaMin, weight);
    const float gx = x / params.SpacingX, gy = y / params.SpacingY;
    const uint x0 = min(uint(gx), params.Width - 2), y0 = min(uint(gy), params.Height - 2);
    const float3 sx0 = AgarwalSampleAxis(heights, boundary_slopes, params, y0, true, x, alpha), sx1 = AgarwalSampleAxis(heights, boundary_slopes, params, y0 + 1, true, x, alpha);
    const float3 sy0 = AgarwalSampleAxis(heights, boundary_slopes, params, x0, false, y, alpha), sy1 = AgarwalSampleAxis(heights, boundary_slopes, params, x0 + 1, false, y, alpha);
    const float3 sx = mix(sx0, sx1, gy - float(y0)), sy = mix(sy0, sy1, gx - float(x0));
    trajectory[frame] = sx.x;
    trajectory[count + frame] = sx.y;
    trajectory[2 * count + frame] = sy.y;
    trajectory[3 * count + frame] = sx.z;
    trajectory[4 * count + frame] = sy.z;
    status[frame] = all(isfinite(sx)) && all(isfinite(sy)) ? 0 : 2;
    float result = 0;
    if (normal != 0) {
        result = params.Beta1 * pow(abs(vx * sx.y + vy * sy.y), params.Beta2) + params.Mass * (sx.z * vx * vx + sy.z * vy * vy);
        if (params.Rolling) {
            const float phase = x / params.Radius, rho = params.Radius - params.Eccentricity * cos(phase) + sx.x;
            if (rho < 0) status[frame] = 1;
            const float rho_dot = params.Eccentricity / params.Radius * vx * sin(phase) + vx * sx.y;
            result += rho * sqrt(rho) * (params.Stiffness + params.Dissipation * rho_dot);
        }
    }
    if (!isfinite(result) && status[frame] == 0) status[frame] = 2;
    force[frame] = result;
}
