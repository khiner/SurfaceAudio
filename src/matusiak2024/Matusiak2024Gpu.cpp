// SPDX-License-Identifier: GPL-3.0-only
#include "matusiak2024/Matusiak2024Gpu.h"
#include <array>
#include <cmath>
#include <stdexcept>
namespace surface_audio::matusiak2024 {
namespace {
float CheckedFloat(double value) {
    const float result = static_cast<float>(value);
    if (!std::isfinite(result) || (value != 0 && result == 0)) throw std::invalid_argument("String coefficient is outside the GPU float range");
    return result;
}
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
        {CheckedFloat(fp.Stiffness), CheckedFloat(fp.Damping), CheckedFloat(fp.StribeckVelocity), CheckedFloat(fp.Dynamic), CheckedFloat(fp.Static), CheckedFloat(fp.BreakawayRatio), fp.Exponential},
        CheckedFloat(s.Step),
        CheckedFloat(s.Spacing),
        CheckedFloat(s.TorsionSpacing),
        CheckedFloat(s.Density),
        CheckedFloat(s.Bending),
        CheckedFloat(s.WaveSpeed),
        CheckedFloat(s.TorsionSpeed),
        CheckedFloat(s.TorsionFeedback),
        CheckedFloat(p.Radius),
        CheckedFloat(p.PolarInertia),
        CheckedFloat(p.Tension),
        CheckedFloat(s.HairStiffness),
        CheckedFloat(s.HairDamping),
        CheckedFloat(p.Damping0),
        CheckedFloat(p.Damping1),
        CheckedFloat(p.TorsionDamping),
        CheckedFloat(1 / (2 / s.Step + 2 * p.Damping0)),
        CheckedFloat(1 / (2 / s.Step + 2 * p.TorsionDamping)),
        CheckedFloat(1 / (s.Step * s.HairStiffness + s.HairDamping)),
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
    const auto energy = CreateBuffer(gpu, size_t(voices) * frames * sizeof(float)), balance = CreateBuffer(gpu, size_t(voices) * frames * sizeof(float));
    const std::array bindings{GpuBinding{config, 0}, GpuBinding{drive, 1}, GpuBinding{matrices, 2}, GpuBinding{states, 3}, GpuBinding{output, 4}, GpuBinding{residual, 5}, GpuBinding{failures, 6}, GpuBinding{energy, 7}, GpuBinding{balance, 8}};
    const auto kernel = CreateKernel(gpu, "Matusiak2024Distributed");
    BeginGpu(gpu);
    DispatchGpu(gpu, kernel, bindings, {voices, 1, 1}, {32, 1, 1});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto samples = BufferSpan<float>(output), errors = BufferSpan<float>(residual);
    const auto failed = BufferSpan<uint32_t>(failures);
    std::vector<float> final_state;
    final_state.reserve(size_t(voices) * (2 * (n + nt) + 5 * m));
    const auto raw = BufferSpan<float>(states);
    const uint32_t parity = (frames - 1) % 2;
    const std::array ranges{std::array{parity * n, n}, std::array{(1 - parity) * n, n}, std::array{3 * n + parity * nt, nt}, std::array{3 * n + (1 - parity) * nt, nt}, std::array{3 * (n + nt), 5 * m}};
    for (uint32_t voice = 0; voice < voices; ++voice) {
        for (const auto [offset, count] : ranges) {
            const auto values = raw.subspan(voice * stride + offset, count);
            final_state.insert(final_state.end(), values.begin(), values.end());
        }
    }
    const auto energies = BufferSpan<float>(energy), balances = BufferSpan<float>(balance);
    return {{samples.begin(), samples.end()}, {errors.begin(), errors.end()}, {energies.begin(), energies.end()}, {balances.begin(), balances.end()}, std::move(final_state), {failed.begin(), failed.end()}};
}
}
