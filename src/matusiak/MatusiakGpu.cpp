// SPDX-License-Identifier: GPL-3.0-only
#include "matusiak/MatusiakGpu.h"
#include <array>
#include <cmath>
#include <stdexcept>
namespace surface_audio::matusiak {
namespace {
float CheckedFloat(double value) {
    const float result = static_cast<float>(value);
    if (!std::isfinite(result) || (value != 0 && result == 0)) throw std::invalid_argument("String coefficient is outside the GPU float range");
    return result;
}
} // namespace
LumpedBatch RenderLumpedGpu(Gpu &gpu, std::span<const LumpedParameters<float>> parameters, uint32_t frames) {
    if (parameters.empty() || !frames || parameters.size() > UINT32_MAX || parameters.size() * frames > UINT32_MAX) throw std::invalid_argument("Invalid bowed mass batch shape");
    for (const auto &p : parameters) {
        const float values[]{p.SampleRate, p.Mass, p.Stiffness, p.Damping, p.HairMass, p.HairStiffness, p.HairDamping, p.NormalForce, p.BowVelocity, p.Acceleration, p.Friction.Stiffness, p.Friction.Damping, p.Friction.StribeckVelocity, p.Friction.Dynamic, p.Friction.Static, p.Friction.BreakawayRatio};
        for (float v : values)
            if (!std::isfinite(v) || v < 0) throw std::invalid_argument("Invalid bowed mass parameter");
        if (p.SampleRate <= 0 || p.Mass <= 0 || p.HairMass <= 0 || p.NormalForce <= 0 || p.Friction.Stiffness <= 0 || p.Friction.StribeckVelocity <= 0 || p.Friction.Dynamic <= 0 || p.Friction.Static < p.Friction.Dynamic || p.Friction.BreakawayRatio >= 1 || 1 / p.SampleRate >= 2 * std::sqrt(p.Mass / p.Stiffness)) throw std::invalid_argument("Invalid or unstable bowed mass parameters");
    }
    const uint32_t voices = static_cast<uint32_t>(parameters.size());
    const auto input = Upload(gpu, parameters), output = CreateBuffer(gpu, size_t(voices) * frames * sizeof(float));
    const auto energy = CreateBuffer(gpu, voices * sizeof(float)), residual = CreateBuffer(gpu, voices * sizeof(float)), failures = CreateBuffer(gpu, voices * sizeof(uint32_t));
    const auto shape = Upload(gpu, std::array{voices, frames});
    const auto kernel = CreateKernel(gpu, "MatusiakLumped");
    const std::array bindings{GpuBinding{input, 0}, GpuBinding{output, 1}, GpuBinding{energy, 2}, GpuBinding{residual, 3}, GpuBinding{failures, 4}, GpuBinding{shape, 5}};
    BeginGpu(gpu);
    DispatchGpu(gpu, kernel, bindings, {voices, 1, 1}, {32, 1, 1});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto samples = BufferSpan<float>(output), errors = BufferSpan<float>(energy), residuals = BufferSpan<float>(residual);
    const auto failed = BufferSpan<uint32_t>(failures);
    return {{samples.begin(), samples.end()}, {errors.begin(), errors.end()}, {residuals.begin(), residuals.end()}, {failed.begin(), failed.end()}};
}
StringBatch RenderStringsGpu(Gpu &gpu, Parameters p, std::span<const BowDrive> drives, uint32_t frames) {
    if (drives.empty() || !frames || drives.size() > UINT32_MAX || drives.size() * frames > UINT32_MAX || p.BowPoints > 8) throw std::invalid_argument("GPU string batch requires 1-8 contacts and valid dimensions");
    for (const auto &d : drives)
        if (!std::isfinite(d.NormalForce) || !std::isfinite(d.Velocity) || !std::isfinite(d.Acceleration) || d.NormalForce <= 0 || d.Velocity < 0 || d.Acceleration < 0) throw std::invalid_argument("Invalid bow drive");
    const auto s = MakeString(p);
    const uint32_t n = static_cast<uint32_t>(s.U.size()), nt = static_cast<uint32_t>(s.W.size()), m = p.BowPoints, voices = static_cast<uint32_t>(drives.size()), stride = 3 * (n + nt) + 5 * m;
    if (uint64_t(voices) * stride > UINT32_MAX) throw std::invalid_argument("GPU string state indexing exceeds uint32 range");
    const auto fp = p.Friction;
    const DistributedConstants c{
        {CheckedFloat(fp.Stiffness), CheckedFloat(fp.Damping), CheckedFloat(fp.StribeckVelocity), CheckedFloat(fp.Dynamic), CheckedFloat(fp.Static), CheckedFloat(fp.BreakawayRatio)},
        CheckedFloat(s.Step),
        CheckedFloat(s.Spacing),
        CheckedFloat(s.TorsionSpacing),
        CheckedFloat(s.Density),
        CheckedFloat(s.Bending),
        CheckedFloat(s.WaveSpeed),
        CheckedFloat(s.TorsionSpeed),
        CheckedFloat(p.Radius),
        CheckedFloat(p.PolarInertia),
        CheckedFloat(p.Tension),
        CheckedFloat(s.HairMass),
        CheckedFloat(s.HairStiffness),
        CheckedFloat(s.HairDamping),
        CheckedFloat(p.Damping0),
        CheckedFloat(p.Damping1),
        CheckedFloat(p.TorsionDamping),
        CheckedFloat(1 / (2 / s.Step + 2 * p.Damping0)),
        CheckedFloat(1 / (2 / s.Step + 2 * p.TorsionDamping)),
        CheckedFloat(1 / (2 * s.HairMass / s.Step + s.Step * s.HairStiffness / 2 + s.HairDamping)),
        n,
        nt,
        m,
        frames,
        voices,
        stride
    };
    std::vector<float> operators;
    operators.reserve(s.Interpolation.size() + s.TorsionInterpolation.size() + s.Coupling.size());
    for (const auto *v : {&s.Interpolation, &s.TorsionInterpolation, &s.Coupling})
        for (double x : *v) operators.push_back(CheckedFloat(x));
    const auto config = Upload(gpu, c), drive = Upload(gpu, drives), matrices = Upload<float>(gpu, operators), states = CreateBuffer(gpu, size_t(voices) * stride * sizeof(float));
    std::memset(states.Data, 0, states.Size);
    const auto output = CreateBuffer(gpu, size_t(voices) * frames * sizeof(float)), residual = CreateBuffer(gpu, voices * sizeof(float)), failures = CreateBuffer(gpu, voices * sizeof(uint32_t));
    const std::array bindings{GpuBinding{config, 0}, GpuBinding{drive, 1}, GpuBinding{matrices, 2}, GpuBinding{states, 3}, GpuBinding{output, 4}, GpuBinding{residual, 5}, GpuBinding{failures, 6}};
    const auto kernel = CreateKernel(gpu, "MatusiakDistributed");
    BeginGpu(gpu);
    DispatchGpu(gpu, kernel, bindings, {voices, 1, 1}, {32, 1, 1});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto samples = BufferSpan<float>(output), errors = BufferSpan<float>(residual);
    const auto failed = BufferSpan<uint32_t>(failures);
    std::vector<float> final_state;
    final_state.reserve(size_t(voices) * (2 * (n + nt) + 5 * m));
    const auto raw = BufferSpan<float>(states);
    for (uint32_t voice = 0; voice < voices; ++voice) {
        const uint32_t base = voice * stride, parity = (frames - 1) % 2;
        const uint32_t offsets[]{parity * n, (1 - parity) * n, 3 * n + parity * nt, 3 * n + (1 - parity) * nt, 3 * (n + nt), 3 * (n + nt) + m, 3 * (n + nt) + 2 * m, 3 * (n + nt) + 3 * m, 3 * (n + nt) + 4 * m};
        const uint32_t sizes[]{n, n, nt, nt, m, m, m, m, m};
        for (size_t j = 0; j < 9; ++j) final_state.insert(final_state.end(), raw.begin() + base + offsets[j], raw.begin() + base + offsets[j] + sizes[j]);
    }
    return {{samples.begin(), samples.end()}, {errors.begin(), errors.end()}, std::move(final_state), {failed.begin(), failed.end()}};
}
} // namespace surface_audio::matusiak
