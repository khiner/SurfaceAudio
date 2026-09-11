#include "willemsen/Willemsen.h"
#include <array>
#include <numbers>
namespace surface_audio::willemsen {
Model<double> MakeModel(Parameters p) {
    for (double x : {p.SampleRate, p.Fundamental, p.Length, p.Density, p.Radius, p.Young, p.Loss0, p.Loss1, p.Noise})
        if (!std::isfinite(x) || x < 0) throw std::invalid_argument("Invalid Willemsen parameter");
    if (p.SampleRate == 0 || p.Fundamental == 0 || p.Length == 0 || p.Density == 0 || p.Radius == 0 || !std::isfinite(p.BowPosition) || !std::isfinite(p.PickupPosition)) throw std::invalid_argument("Invalid Willemsen dimensions");
    const double density = p.Density * std::numbers::pi * p.Radius * p.Radius;
    const double wave = std::pow(2 * p.Fundamental * p.Length, 2), stiffness = p.Young * p.Radius * p.Radius / (4 * p.Density);
    const double count = std::floor(p.Length / StiffStringMinimumSpacing(std::sqrt(wave), stiffness, p.Loss1, 1 / p.SampleRate));
    if (!std::isfinite(count) || count < 6 || count >= double(MaximumPoints)) throw std::invalid_argument("Willemsen grid exceeds supported dimensions");
    const double spacing = p.Length / count, x = p.BowPosition * count;
    if (x < 2 || x >= count - 2 || p.PickupPosition <= 0 || p.PickupPosition >= 1) throw std::invalid_argument("Willemsen contact is outside interior grid");
    const bool author = p.Discretization == Scheme::AuthorFigure;
    const unsigned intervals = unsigned(count) - author, bow = unsigned(x) - author, pickup = unsigned(p.PickupPosition * count) - author;
    if (author && (bow < 2 || bow > intervals - 2)) throw std::invalid_argument("Author contact is outside clamped interior");
    const double a = x - unsigned(x), dt = 1 / p.SampleRate, h2 = spacing * spacing;
    const std::array weights = author ? std::array{0., 1., 0., 0.} :
        std::array{-a * (a - 1) * (a - 2) / 6, (a - 1) * (a + 1) * (a - 2) / 2, -a * (a + 1) * (a - 2) / 2, a * (a + 1) * (a - 1) / 6};
    const double lambda = wave * dt * dt / h2, mu = stiffness * dt * dt / (h2 * h2);
    const auto coefficients = [&](double scale) {
        const double loss = 2 * p.Loss1 * scale * dt / h2, denominator = 1 + p.Loss0 * scale * dt;
        return std::array{(2 - 2 * lambda - 6 * mu - 2 * loss) / denominator, (lambda + 4 * mu + loss) / denominator,
                          -mu / denominator, (-1 + p.Loss0 * scale * dt + 2 * loss) / denominator, -loss / denominator};
    };
    const auto update = coefficients(1), contact = author ? coefficients(density) : update;
    double coupling = 0;
    if (author) coupling = 1 / (2 * density * spacing * (p.SampleRate + p.Loss0 * density));
    else for (double weight : weights) coupling += weight * weight / (2 * density * spacing * (p.SampleRate + p.Loss0));
    return {.SampleRate = p.SampleRate, .Spacing = spacing, .Density = density, .Loss0 = p.Loss0, .Loss1 = p.Loss1,
            .WaveSpeedSquared = wave, .StiffnessSquared = stiffness, .BristleDamping = author ? 1. : .1, .Noise = p.Noise, .Coupling = coupling,
            .Update = {update[0], update[1], update[2], update[3], update[4]},
            .ContactUpdate = {contact[0], contact[1], contact[2], contact[3], contact[4]},
            .Weight = {weights[0], weights[1], weights[2], weights[3]}, .Intervals = intervals, .BowIndex = bow, .Pickup = pickup, .AuthorFigure = unsigned(author)};
}
Model<float> MakeFloatModel(Parameters p) {
    const auto d = MakeModel(p);
    const auto cast = [](double v) { const float x = static_cast<float>(v); if (!std::isfinite(x) || (v != 0 && x == 0)) throw std::invalid_argument("Willemsen FP32 range"); return x; };
    return {.SampleRate = cast(d.SampleRate), .Spacing = cast(d.Spacing), .Density = cast(d.Density), .Loss0 = cast(d.Loss0), .Loss1 = cast(d.Loss1),
            .WaveSpeedSquared = cast(d.WaveSpeedSquared), .StiffnessSquared = cast(d.StiffnessSquared), .BristleDamping = cast(d.BristleDamping),
            .Noise = cast(d.Noise), .Coupling = cast(d.Coupling),
            .Update = {cast(d.Update[0]), cast(d.Update[1]), cast(d.Update[2]), cast(d.Update[3]), cast(d.Update[4])},
            .ContactUpdate = {cast(d.ContactUpdate[0]), cast(d.ContactUpdate[1]), cast(d.ContactUpdate[2]), cast(d.ContactUpdate[3]), cast(d.ContactUpdate[4])},
            .Weight = {cast(d.Weight[0]), cast(d.Weight[1]), cast(d.Weight[2]), cast(d.Weight[3])},
            .Intervals = d.Intervals, .BowIndex = d.BowIndex, .Pickup = d.Pickup, .AuthorFigure = d.AuthorFigure};
}
Batch RenderGpu(Gpu &gpu, Parameters p, std::span<const Drive> drives, unsigned frames) {
    if (drives.empty() || !frames || uint64_t(drives.size()) * frames > UINT32_MAX) throw std::invalid_argument("Invalid Willemsen batch");
    for (const auto d : drives)
        if (!std::isfinite(d.Velocity) || !std::isfinite(d.NormalForce) || d.NormalForce < 0) throw std::invalid_argument("Invalid Willemsen drive");
    const auto model = MakeFloatModel(p);
    const unsigned voices = static_cast<unsigned>(drives.size()), stride = 2 * (model.Intervals + 1) + 3;
    if (uint64_t(voices) * stride > UINT32_MAX) throw std::invalid_argument("Willemsen state index exceeds uint32 range");
    const auto input = Upload(gpu, model), controls = Upload(gpu, drives), shape = Upload(gpu, std::array{voices, frames});
    const auto wave = CreateBuffer(gpu, size_t(voices) * frames * sizeof(float));
    const auto state = CreateBuffer(gpu, size_t(voices) * stride * sizeof(float));
    const auto residual = CreateBuffer(gpu, voices * sizeof(float)), failed = CreateBuffer(gpu, voices * sizeof(unsigned));
    const auto kernel = CreateKernel(gpu, "WillemsenRender");
    const std::array bindings{GpuBinding{input, 0}, GpuBinding{controls, 1}, GpuBinding{shape, 2}, GpuBinding{wave, 3}, GpuBinding{state, 4}, GpuBinding{residual, 5}, GpuBinding{failed, 6}};
    BeginGpu(gpu);
    DispatchGpu(gpu, kernel, bindings, {voices, 1, 1}, {32, 1, 1});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto w = BufferSpan<float>(wave), s = BufferSpan<float>(state), r = BufferSpan<float>(residual);
    const auto f = BufferSpan<unsigned>(failed);
    return {{w.begin(), w.end()}, {s.begin(), s.end()}, {r.begin(), r.end()}, {f.begin(), f.end()}};
}
}
