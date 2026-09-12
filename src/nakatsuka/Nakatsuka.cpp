#include "Nakatsuka.h"
#include "core/ExtendedFloat.h"
#include "core/GpuMix.h"
#include "core/Random.h"
#include <algorithm>
#include <cmath>
#include <numbers>

namespace surface_audio::nakatsuka {
namespace {
void Validate(const Material &m) {
    if (!(m.Modulus > 0 && std::isfinite(m.Modulus) && std::abs(m.Poisson) < 1 && m.ArealDensity > 0 && std::isfinite(m.ArealDensity) && m.Thickness > 0 && std::isfinite(m.Thickness))) throw std::invalid_argument("Invalid microrectangle material");
}
uint32_t Frames(const Material &m, const AcousticSettings &s, std::span<const Patch> patches, std::span<const Sample> samples) {
    Validate(m);
    if (patches.empty() || samples.empty() || samples.size() % patches.size() || samples.size() > UINT32_MAX || !(s.SampleRate > 0 && std::isfinite(s.SampleRate) && s.ReferenceSpeed > 0 && std::isfinite(s.ReferenceSpeed) && s.AirDensity > 0 && std::isfinite(s.AirDensity) && s.SoundSpeed > 0 && std::isfinite(s.SoundSpeed) && s.Attenuation >= 0 && std::isfinite(s.Attenuation)) || !s.OddModes || s.OddModes > 8) throw std::invalid_argument("Invalid microrectangle render dimensions");
    for (const auto &p : patches)
        if (!(p.Width > 0 && std::isfinite(p.Width) && p.Height > 0 && std::isfinite(p.Height))) throw std::invalid_argument("Invalid microrectangle dimensions");
    for (const auto &p : samples)
        if (!(p.Speed >= 0 && std::isfinite(p.Speed) && std::isfinite(p.Displacement) && p.Distance > 0 && std::isfinite(p.Distance) && p.Contact >= 0 && p.Contact <= 1)) throw std::invalid_argument("Invalid contact trajectory");
    return uint32_t(samples.size() / patches.size());
}
std::vector<double> Frequencies(const Material &m, uint32_t odd_modes, std::span<const Patch> patches) {
    std::vector<double> result;
    result.reserve(patches.size() * odd_modes * odd_modes);
    for (const auto &p : patches)
        for (uint32_t i = 0; i < odd_modes; ++i)
            for (uint32_t j = 0; j < odd_modes; ++j) result.push_back(AngularFrequency(m, p.Width, p.Height, 2 * i + 1, 2 * j + 1));
    return result;
}
}
double AdhesionPotential(double distance, double degree, double effective_distance) {
    if (!(distance >= 0 && std::isfinite(distance) && degree >= 0 && std::isfinite(degree) && effective_distance > 0 && std::isfinite(effective_distance))) throw std::invalid_argument("Invalid adhesion parameters");
    return distance > effective_distance ? degree / distance : degree;
}
double AdhesionCorrection(double distance, double degree, double effective_distance, bool shifted) {
    AdhesionPotential(distance, degree, effective_distance);
    if (degree == 0 || distance <= effective_distance) return 0;
    return shifted ? distance * (1 - distance / effective_distance) : distance;
}
double AngularFrequency(const Material &m, double a, double b, uint32_t i, uint32_t j) {
    Validate(m);
    if (!(a > 0 && b > 0 && std::isfinite(a) && std::isfinite(b)) || !i || !j) throw std::invalid_argument("Invalid rectangle mode");
    return std::numbers::pi * std::numbers::pi * (double(i) * i / (a * a) + double(j) * j / (b * b)) * std::sqrt(m.Modulus * std::pow(m.Thickness, 3) / (12 * m.ArealDensity * (1 - m.Poisson * m.Poisson)));
}
std::vector<Patch> MakePatches(const Material &m, uint32_t count, uint64_t seed) {
    Validate(m);
    if (!count || !(m.Width > 0 && std::isfinite(m.Width) && m.Height > 0 && std::isfinite(m.Height) && m.WidthDeviation >= 0 && std::isfinite(m.WidthDeviation) && m.HeightDeviation >= 0 && std::isfinite(m.HeightDeviation))) throw std::invalid_argument("Invalid patch distribution");
    auto random = MakeRandom(seed);
    const auto draw = [&](double mean, double deviation) {
        for (uint32_t attempt = 0; attempt < 10000; ++attempt) {
            const double x = mean + deviation * Normal(random);
            if (float(x) > 0 && std::isfinite(float(x))) return float(x);
        }
        throw std::runtime_error("No positive rectangle dimension sampled");
    };
    std::vector<Patch> patches;
    patches.reserve(count);
    for (uint32_t i = 0; i < count; ++i) patches.push_back({draw(m.Width, m.WidthDeviation), draw(m.Height, m.HeightDeviation)});
    return patches;
}
Simulation SimulateGpu(Gpu &gpu, const Scene &s, uint32_t frames) {
    const uint64_t count = uint64_t(s.Columns) * s.Rows;
    if (s.Columns < 2 || s.Rows < 2 || count > 1024 || !frames || count * frames > UINT32_MAX || !s.Iterations || s.Iterations > 32 || s.ShiftedAdhesion > 1 || !(s.SampleRate > 0 && s.Spacing > 0 && s.InverseMass > 0 && s.Speed >= 0 && s.Gravity >= 0 && s.Damping >= 0 && s.Stretch >= 0 && s.Stretch <= 1 && s.Adhesion >= 0 && s.Adhesion <= 1 && s.AdhesionDistance > 0 && s.SphereRadius >= 0 && s.InitialHeight >= s.SphereRadius && s.ListenerHeight > s.InitialHeight && s.MotionStart >= 0 && s.MotionDuration >= 0 && s.SettlingTime >= 0 && s.MotionRise >= 0 && s.MotionFall >= 0 && (s.MotionDuration == 0 || s.MotionRise + s.MotionFall <= s.MotionDuration) && double(s.SettlingTime) * s.SampleRate + frames <= UINT32_MAX)) throw std::invalid_argument("Invalid Nakatsuka scene");
    for (float value : std::array{s.SampleRate, s.Spacing, s.InverseMass, s.Speed, s.Gravity, s.Damping, s.Stretch, s.Adhesion, s.AdhesionDistance, s.SphereRadius, s.InitialHeight, s.ListenerHeight, s.MotionStart, s.MotionDuration, s.SettlingTime, s.OriginY, s.MotionRise, s.MotionFall})
        if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite Nakatsuka scene");
    const auto parameters = Upload(gpu, s), length = Upload(gpu, frames), output = CreateBuffer(gpu, count * frames * sizeof(Sample)), positions = CreateBuffer(gpu, count * 4 * sizeof(float));
    const auto kernel = CreateKernel(gpu, "NakatsukaSimulate");
    const std::array bindings{GpuBinding{parameters, 0}, GpuBinding{length, 1}, GpuBinding{output, 2}, GpuBinding{positions, 3}};
    BeginGpu(gpu);
    DispatchGroupsGpu(gpu, kernel, bindings, {1}, {uint32_t((count + 31) / 32 * 32)});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto samples = BufferSpan<Sample>(output);
    const auto final = BufferSpan<float>(positions);
    return {{samples.begin(), samples.end()}, {final.begin(), final.end()}};
}
std::vector<double> Render(const Material &m, const AcousticSettings &s, std::span<const Patch> patches, std::span<const Sample> samples) {
    const auto frames = Frames(m, s, patches, samples);
    const auto frequencies = Frequencies(m, s.OddModes, patches);
    const uint32_t modes = s.OddModes * s.OddModes;
    std::vector<double> output(frames);
    for (size_t patch = 0; patch < patches.size(); ++patch) {
        std::vector<double> phase(modes);
        for (uint32_t frame = 0; frame < frames; ++frame) {
            const auto &sample = samples[patch * frames + frame];
            const double scale = sample.Speed / s.ReferenceSpeed;
            for (uint32_t mode = 0; mode < modes; ++mode) {
                const double omega = frequencies[patch * modes + mode] * scale * scale;
                auto &angle = phase[mode];
                angle = std::remainder(angle + omega / s.SampleRate, 2 * std::numbers::pi);
                if (omega >= std::numbers::pi * s.SampleRate) continue;
                const double amplitude = sample.Displacement * sample.Contact / modes;
                output[frame] -= s.AirDensity * amplitude * omega * omega * std::exp(-s.Attenuation) / (4 * std::numbers::pi * sample.Distance) * std::cos(angle - omega * sample.Distance / s.SoundSpeed);
            }
        }
    }
    return output;
}
std::vector<float> RenderGpu(Gpu &gpu, const Material &m, const AcousticSettings &s, std::span<const Patch> patches, std::span<const Sample> samples) {
    const auto frames = Frames(m, s, patches, samples);
    const uint32_t modes = s.OddModes * s.OddModes;
    std::vector<ExtendedFloat> frequencies;
    frequencies.reserve(patches.size() * modes + 4);
    for (double frequency : Frequencies(m, s.OddModes, patches)) frequencies.push_back(SplitFloat(frequency));
    frequencies.insert(frequencies.end(), {SplitFloat(1 / s.SampleRate), SplitFloat(1 / (s.ReferenceSpeed * s.ReferenceSpeed)), SplitFloat(2 * std::numbers::pi), SplitFloat(std::numbers::pi * s.SampleRate)});
    struct Block {
        uint32_t Frames, Patches, Modes;
        float SampleRate, AirDensity, SoundSpeed, Attenuation;
    };
    const auto parameters = Upload(gpu, Block{frames, uint32_t(patches.size()), modes, float(s.SampleRate), float(s.AirDensity), float(s.SoundSpeed), float(s.Attenuation)}), frequency = Upload<ExtendedFloat>(gpu, frequencies), trajectory = Upload<Sample>(gpu, samples), output = CreateBuffer(gpu, samples.size() * sizeof(float));
    const auto kernel = CreateKernel(gpu, "NakatsukaRender");
    const auto mix = CreateGpuMix(gpu, {uint32_t(patches.size()), frames});
    const std::array bindings{GpuBinding{parameters, 0}, GpuBinding{frequency, 1}, GpuBinding{trajectory, 2}, GpuBinding{output, 3}};
    BeginGpu(gpu);
    DispatchGpu(gpu, kernel, bindings, {uint32_t(patches.size())});
    EncodeMix(gpu, mix, output);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto result = BufferSpan<float>(mix.Output);
    return {result.begin(), result.end()};
}
}
