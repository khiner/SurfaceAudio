#include "Hatt.h"
#include "core/ExtendedFloat.h"
#include "core/LineSpectrum.h"
#include <algorithm>
#include <cmath>

namespace surface_audio::hatt {
namespace {
double Bound(double value, double maximum, bool upstream) {
    if (!upstream) return std::clamp(value, 0., maximum);
    if (value > maximum) return maximum - .01;
    return value < .05 ? .0001 : value;
}
}

void Validate(const Texture &t) {
    if (!(t.SampleRate > 0 && std::isfinite(t.SampleRate) && t.Friction >= 0 && std::isfinite(t.Friction) && t.MaximumSpeed > 0 && std::isfinite(t.MaximumSpeed) && t.MaximumForce > 0 && std::isfinite(t.MaximumForce)) || t.Models.empty() || t.Triangles.empty()) throw std::invalid_argument("Invalid HaTT texture");
    const auto ar = t.Models.front().ArLsf.size(), ma = t.Models.front().MaLsf.size();
    if (!ar || ar > MaximumOrder || ma > MaximumOrder) throw std::invalid_argument("Unsupported HaTT order");
    for (const auto &m : t.Models) {
        if (m.ArLsf.size() != ar || m.MaLsf.size() != ma || !(m.Speed >= 0 && m.Speed <= t.MaximumSpeed && m.Force >= 0 && m.Force <= t.MaximumForce && m.Variance >= 0 && std::isfinite(m.Variance) && m.Gain > 0 && std::isfinite(m.Gain))) throw std::invalid_argument("Invalid HaTT model");
        LineSpectrumPolynomial(m.ArLsf);
        LineSpectrumPolynomial(m.MaLsf);
    }
    for (const auto &triangle : t.Triangles) {
        for (auto i : triangle)
            if (i >= t.Models.size()) throw std::invalid_argument("Invalid HaTT triangle index");
        const auto &a = t.Models[triangle[0]], &b = t.Models[triangle[1]], &c = t.Models[triangle[2]];
        if ((b.Speed - a.Speed) * (c.Force - a.Force) == (c.Speed - a.Speed) * (b.Force - a.Force)) throw std::invalid_argument("Degenerate HaTT triangle");
    }
}
Filter Interpolate(const Texture &t, Control control, bool upstream_bounds) {
    if (!std::isfinite(control.Speed) || !std::isfinite(control.Force) || t.Models.empty()) throw std::invalid_argument("Invalid HaTT control");
    const double speed = Bound(control.Speed, t.MaximumSpeed, upstream_bounds), force = Bound(control.Force, t.MaximumForce, upstream_bounds);
    for (const auto &triangle : t.Triangles) {
        const auto &a = t.Models[triangle[0]], &b = t.Models[triangle[1]], &c = t.Models[triangle[2]];
        const double determinant = (b.Speed - a.Speed) * (c.Force - a.Force) - (c.Speed - a.Speed) * (b.Force - a.Force);
        const double v = ((speed - a.Speed) * (c.Force - a.Force) - (c.Speed - a.Speed) * (force - a.Force)) / determinant;
        const double w = ((b.Speed - a.Speed) * (force - a.Force) - (speed - a.Speed) * (b.Force - a.Force)) / determinant;
        if (std::min({1 - v - w, v, w}) < -1e-8) continue;
        const std::array weight{std::max(0., 1 - v - w), std::max(0., v), std::max(0., w)};
        const double scale = 1 / (weight[0] + weight[1] + weight[2]);
        std::vector<double> ar(a.ArLsf.size()), ma(a.MaLsf.size());
        double variance = 0, gain = 0;
        for (size_t j = 0; j < 3; ++j) {
            const auto &m = t.Models[triangle[j]];
            const double k = weight[j] * scale;
            variance += k * m.Variance;
            gain += k * m.Gain;
            for (size_t i = 0; i < ar.size(); ++i) ar[i] += k * m.ArLsf[i];
            for (size_t i = 0; i < ma.size(); ++i) ma[i] += k * m.MaLsf[i];
        }
        const auto denominator = LineSpectrumPolynomial(ar), numerator = LineSpectrumPolynomial(ma);
        Filter result{.Variance = variance, .ArOrder = uint32_t(ar.size()), .MaOrder = uint32_t(ma.size())};
        std::copy(denominator.begin(), denominator.end(), result.Ar.begin());
        std::transform(numerator.begin(), numerator.end(), result.Ma.begin(), [gain](double x) { return x * gain; });
        return result;
    }
    throw std::invalid_argument("HaTT control outside published triangulation");
}
double Tick(const Filter &f, State &s, double standard_normal) {
    const double excitation = standard_normal * std::sqrt(f.Variance);
    double output = f.Ma[0] * excitation;
    for (uint32_t i = 0; i < f.ArOrder; ++i) output -= f.Ar[i + 1] * s.Output[i];
    for (uint32_t i = 0; i < f.MaOrder; ++i) output += f.Ma[i + 1] * s.Excitation[i];
    std::move_backward(s.Output.begin(), s.Output.end() - 1, s.Output.end());
    std::move_backward(s.Excitation.begin(), s.Excitation.end() - 1, s.Excitation.end());
    s.Output[0] = output;
    s.Excitation[0] = excitation;
    return output;
}
double FrictionForce(double velocity, double force, double coefficient, bool upstream) {
    if (!std::isfinite(velocity) || !(force >= 0 && std::isfinite(force) && coefficient >= 0 && std::isfinite(coefficient))) throw std::invalid_argument("Invalid HaTT friction input");
    if (upstream && std::abs(velocity) < coefficient / .004) return -coefficient * force * .004 * velocity;
    return -force * std::clamp(.004 * velocity, -coefficient, coefficient);
}
std::vector<float> RenderGpu(Gpu &gpu, std::span<const Texture> textures, std::span<const Control> controls, std::span<const float> noise, bool upstream_bounds) {
    if (textures.empty() || noise.empty() || noise.size() != controls.size() || noise.size() % textures.size() || noise.size() > UINT32_MAX) throw std::invalid_argument("Invalid HaTT batch dimensions");
    std::vector<std::array<uint32_t, 4>> metadata, triangles;
    std::vector<ExtendedFloat> models, coordinates, weights;
    for (const auto &t : textures) {
        Validate(t);
        const auto offset = uint32_t(models.size() / ModelStride);
        metadata.push_back({uint32_t(triangles.size()), uint32_t(t.Triangles.size()), uint32_t(t.Models.front().ArLsf.size()), uint32_t(t.Models.front().MaLsf.size())});
        for (const auto &m : t.Models) {
            for (size_t i = 0; i < MaximumOrder; ++i) models.push_back(SplitFloat(i < m.ArLsf.size() ? m.ArLsf[i] : 0));
            for (size_t i = 0; i < MaximumOrder; ++i) models.push_back(SplitFloat(i < m.MaLsf.size() ? m.MaLsf[i] : 0));
            models.push_back(SplitFloat(m.Variance));
            models.push_back(SplitFloat(m.Gain));
        }
        for (const auto &triangle : t.Triangles) {
            const auto &a = t.Models[triangle[0]], &b = t.Models[triangle[1]], &c = t.Models[triangle[2]];
            const double determinant = (b.Speed - a.Speed) * (c.Force - a.Force) - (c.Speed - a.Speed) * (b.Force - a.Force);
            const std::array affine{(c.Force - a.Force) / determinant, (a.Speed - c.Speed) / determinant, (c.Speed * a.Force - a.Speed * c.Force) / determinant, (a.Force - b.Force) / determinant, (b.Speed - a.Speed) / determinant, (a.Speed * b.Force - b.Speed * a.Force) / determinant};
            for (double value : affine) weights.push_back(SplitFloat(value));
            triangles.push_back({offset + triangle[0], offset + triangle[1], offset + triangle[2], 0});
        }
    }
    coordinates.reserve(controls.size() * 2);
    const size_t frames = noise.size() / textures.size();
    for (size_t i = 0; i < controls.size(); ++i) {
        const auto &t = textures[i / frames];
        if (!std::isfinite(controls[i].Speed) || !std::isfinite(controls[i].Force) || !std::isfinite(noise[i])) throw std::invalid_argument("Nonfinite HaTT control");
        coordinates.push_back(SplitFloat(Bound(controls[i].Speed, t.MaximumSpeed, upstream_bounds)));
        coordinates.push_back(SplitFloat(Bound(controls[i].Force, t.MaximumForce, upstream_bounds)));
    }
    std::array<ExtendedFloat, 15> cosine;
    double term = 1;
    for (size_t i = 0; i < cosine.size(); ++i) {
        cosine[i] = SplitFloat(term);
        term /= -double((2 * i + 1) * (2 * i + 2));
    }
    const std::array block{uint32_t(frames), uint32_t(textures.size())};
    const auto parameters = Upload(gpu, block), model_buffer = Upload<ExtendedFloat>(gpu, models), coordinate_buffer = Upload<ExtendedFloat>(gpu, coordinates), triangle_buffer = Upload<std::array<uint32_t, 4>>(gpu, triangles), weight_buffer = Upload<ExtendedFloat>(gpu, weights), meta_buffer = Upload<std::array<uint32_t, 4>>(gpu, metadata), cosine_buffer = Upload(gpu, cosine);
    const auto coefficients = CreateBuffer(gpu, noise.size() * FilterStride * sizeof(ExtendedFloat)), input = Upload<float>(gpu, noise), output = CreateBuffer(gpu, noise.size_bytes());
    const auto interpolate = CreateKernel(gpu, "HattInterpolate"), render = CreateKernel(gpu, "HattRender");
    const std::array interpolation_bindings{GpuBinding{parameters, 0}, GpuBinding{model_buffer, 1}, GpuBinding{coordinate_buffer, 2}, GpuBinding{triangle_buffer, 3}, GpuBinding{weight_buffer, 4}, GpuBinding{meta_buffer, 5}, GpuBinding{cosine_buffer, 6}, GpuBinding{coefficients, 7}};
    const std::array render_bindings{GpuBinding{parameters, 0}, GpuBinding{coefficients, 1}, GpuBinding{meta_buffer, 2}, GpuBinding{input, 3}, GpuBinding{output, 4}};
    BeginGpu(gpu);
    DispatchGpu(gpu, interpolate, interpolation_bindings, {uint32_t(noise.size())});
    DispatchGpu(gpu, render, render_bindings, {uint32_t(textures.size())});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto samples = BufferSpan<float>(output);
    for (float sample : samples)
        if (!std::isfinite(sample)) throw std::runtime_error("Invalid or unstable HaTT GPU interpolation");
    return {samples.begin(), samples.end()};
}
}
