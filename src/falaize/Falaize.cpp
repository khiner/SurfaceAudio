#include "falaize/Falaize.h"
#include <array>
#include <numbers>
#include <stdexcept>
namespace surface_audio::falaize {
Model<double> MakeModel(Parameters<double> p) {
    const double values[]{p.SampleRate, p.Length, p.Density, p.Tension, p.StringDamping, p.Position, p.NormalForce, p.Dynamic, p.Static, p.StribeckVelocity, p.Stiffness, p.ComplianceDamping, p.FluidDamping, p.BreakawayRatio, p.Theta, p.HammerStiffness, p.HammerDamping, p.HammerExponent, p.HammerThickness, p.HammerMass};
    for (const double value : values)
        if (!std::isfinite(value) || value < 0) throw std::invalid_argument("Invalid Falaize parameter");
    if (p.Elements) p.Modes = p.Elements - 1;
    if (!p.Modes || p.Modes > MaximumModes || p.SampleRate == 0 || p.Length == 0 || p.Density == 0 || p.NormalForce == 0 || p.Dynamic == 0 || p.Static < p.Dynamic || p.StribeckVelocity == 0 || p.Stiffness == 0 || p.BreakawayRatio >= 1 || p.Theta > 1 || p.Position > 1 || p.HammerThickness == 0 || p.HammerMass == 0 || p.HammerExponent < 1) throw std::invalid_argument("Invalid Falaize model dimensions or coefficients");
    Model<double> m{p};
    const double dt = 1 / p.SampleRate;
    for (unsigned i = 0; i < p.Modes; ++i) {
        const double mode = i + 1, angle = mode * std::numbers::pi;
        if (!p.Elements) {
            m.OmegaSquared[i] = p.Tension / p.Density * std::pow(angle / p.Length, 2);
            m.Shape[i] = std::sqrt(2 / p.Length) * std::sin(angle * p.Position);
        } else {
            const double theta = angle / p.Elements, h = p.Length / p.Elements, mass = h / 3 * (2 + std::cos(theta));
            m.OmegaSquared[i] = p.Tension / p.Density * (2 - 2 * std::cos(theta)) / (h * mass);
            const double x = p.Position * p.Elements;
            const unsigned left = static_cast<unsigned>(x);
            const double fraction = x - left;
            m.Shape[i] = std::sqrt(2 / (p.Elements * mass)) * ((1 - fraction) * std::sin(theta * left) + fraction * std::sin(theta * (left + 1)));
        }
        m.Response[i] = dt * m.Shape[i] / (2 * p.Density) / (1 + dt * p.StringDamping / (2 * p.Density) + dt * dt / 4 * m.OmegaSquared[i]);
        m.Coupling += m.Shape[i] * m.Response[i];
    }
    return m;
}
Model<float> MakeFloatModel(Parameters<double> p) {
    const auto d = MakeModel(p);
    const auto cast = [](double value) {const float result=static_cast<float>(value);if(!std::isfinite(result)||(value!=0&&result==0))throw std::invalid_argument("Falaize coefficient is outside FP32 range");return result; };
    const auto &q = d.Config;
    Model<float> f{.Config = {cast(q.SampleRate), cast(q.Length), cast(q.Density), cast(q.Tension), cast(q.StringDamping), cast(q.Position), cast(q.NormalForce), cast(q.Dynamic), cast(q.Static), cast(q.StribeckVelocity), cast(q.Stiffness), cast(q.ComplianceDamping), cast(q.FluidDamping), cast(q.BreakawayRatio), cast(q.Theta), cast(q.HammerStiffness), cast(q.HammerDamping), cast(q.HammerExponent), cast(q.HammerThickness), cast(q.HammerMass), q.Modes, q.Elements}};
    f.Coupling = cast(d.Coupling);
    for (unsigned i = 0; i < d.Config.Modes; ++i) {
        f.OmegaSquared[i] = cast(d.OmegaSquared[i]);
        f.Shape[i] = cast(d.Shape[i]);
        f.Response[i] = cast(d.Response[i]);
    }
    return f;
}
Batch RenderGpu(Gpu &gpu, Parameters<double> p, std::span<const float> drives, unsigned frames, bool hammer) {
    if (drives.empty() || !frames || drives.size() * uint64_t(frames) > UINT32_MAX) throw std::invalid_argument("Invalid Falaize batch");
    for (const float v : drives)
        if (!std::isfinite(v)) throw std::invalid_argument("Invalid Falaize drive");
    const unsigned voices = static_cast<unsigned>(drives.size());
    const auto model = MakeFloatModel(p);
    const unsigned stride = 2 * model.Config.Modes + 4;
    if (uint64_t(voices) * stride > UINT32_MAX) throw std::invalid_argument("Falaize state indexing exceeds uint32 range");
    const auto input = Upload(gpu, model), controls = Upload(gpu, drives), shape = Upload(gpu, std::array{voices, frames, unsigned(hammer)});
    const auto wave = CreateBuffer(gpu, size_t(voices) * frames * sizeof(float)), state = CreateBuffer(gpu, size_t(voices) * stride * sizeof(float));
    const auto diagnostics = CreateBuffer(gpu, size_t(voices) * 2 * sizeof(float)), failed = CreateBuffer(gpu, size_t(voices) * sizeof(unsigned));
    const auto kernel = CreateKernel(gpu, "FalaizeRender");
    const std::array bindings{GpuBinding{input, 0}, GpuBinding{controls, 1}, GpuBinding{shape, 2}, GpuBinding{wave, 3}, GpuBinding{state, 4}, GpuBinding{diagnostics, 5}, GpuBinding{failed, 6}};
    BeginGpu(gpu);
    DispatchGpu(gpu, kernel, bindings, {voices, 1, 1}, {32, 1, 1});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto w = BufferSpan<float>(wave), s = BufferSpan<float>(state), d = BufferSpan<float>(diagnostics);
    const auto f = BufferSpan<unsigned>(failed);
    Batch out{{w.begin(), w.end()}, {s.begin(), s.end()}, std::vector<float>(voices), std::vector<float>(voices), {f.begin(), f.end()}};
    for (unsigned i = 0; i < voices; ++i) {
        out.EnergyError[i] = d[2 * i];
        out.Residual[i] = d[2 * i + 1];
    }
    return out;
}
}
