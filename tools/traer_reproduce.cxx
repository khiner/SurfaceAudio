#include "core/AudioFile.h"
#include "core/ErbNoise.h"
#include "core/GpuConvolution.h"
#include "traer/Traer.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <numeric>

using namespace surface_audio;
using namespace surface_audio::traer;
namespace {
using Clock = std::chrono::steady_clock;
double Milliseconds(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
void Save(const std::filesystem::path &path, std::span<const double> values) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size_bytes()));
    if (!stream) throw std::runtime_error("Cannot write reference samples");
}
void Scrape(Gpu &gpu, const std::filesystem::path &config_path, const std::filesystem::path &profile_path, const std::filesystem::path &output) {
    std::ifstream config(config_path);
    uint32_t taps{}, rate{}, mode_count{}, force_frames{};
    double spacing{}, mass{}, shear{}, gamma{}, low{}, high{}, cycles{};
    config >> taps >> rate >> mode_count >> force_frames >> spacing >> mass >> shear >> gamma >> low >> high >> cycles;
    if (!config || !mode_count || mode_count > 1024 || !taps || taps > 2000000 || !force_frames || force_frames > 2000000)
        throw std::invalid_argument("Invalid scrape configuration");
    std::vector<Resonance> modes(3 * mode_count);
    for (uint32_t i = 0; i < mode_count; ++i) config >> modes[i].Frequency >> modes[i].OnsetDb >> modes[i].DecayDbPerSecond;
    if (!config) throw std::invalid_argument("Truncated scrape modes");
    auto random = MakeRandom(2019);
    for (uint32_t region = 1; region < 3; ++region) {
        std::copy_n(modes.begin(), mode_count, modes.begin() + region * mode_count);
        PerturbOnsets(std::span(modes).subspan(region * mode_count, mode_count), random);
    }
    const auto bytes = std::filesystem::file_size(profile_path);
    if (bytes % sizeof(double)) throw std::invalid_argument("Invalid surface binary size");
    std::vector<double> profile(bytes / sizeof(double));
    std::ifstream stream(profile_path, std::ios::binary);
    stream.read(reinterpret_cast<char *>(profile.data()), std::streamsize(bytes));
    if (!stream) throw std::invalid_argument("Cannot read surface profile");
    std::vector<double> position(force_frames), velocity(force_frames);
    std::vector<float> locations(force_frames);
    const double duration = double(force_frames - 1) / rate, omega = 2 * std::numbers::pi * cycles / duration;
    for (uint32_t frame = 0; frame < force_frames; ++frame) {
        const double time = double(frame) / rate;
        position[frame] = (low + high) / 2 - (high - low) / 2 * std::cos(omega * time);
        velocity[frame] = (high - low) / 2 * omega * std::sin(omega * time);
        locations[frame] = float(position[frame]);
    }
    const auto force = ScrapeExcitation(profile, spacing, position, velocity, mass, shear, gamma);
    auto response = CreateResponseGpu(gpu, taps, 3, float(rate), modes, {}, {});
    BeginGpu(gpu);
    EncodeResponse(gpu, response);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto responses = BufferSpan<float>(response.Output);
    const std::array nodes{float(low), float((low + high) / 2), float(high)};
    const auto varying = ConvolveSpatialGpu(gpu, force, responses, taps, nodes, locations);
    const auto constant = ConvolveFixedGpu(gpu, force, responses.first(taps));
    std::filesystem::create_directories(output);
    WriteWave(output / "varying.wav", rate, 1, varying);
    WriteWave(output / "constant.wav", rate, 1, constant);
    WriteWave(output / "force.wav", rate, 1, force);
    Save(output / "position.f64", position);
    Save(output / "velocity.f64", velocity);
    std::ofstream mode_stream(output / "modes.txt");
    mode_stream << std::setprecision(17);
    for (const auto &mode : modes) mode_stream << mode.Frequency << ' ' << mode.OnsetDb << ' ' << mode.DecayDbPerSecond << '\n';
    if (!mode_stream) throw std::runtime_error("Cannot write scrape parameters");
}
void Benchmark(Gpu &gpu) {
    constexpr uint32_t frames = 44100, voices = 32, rate = 44100;
    std::vector<Resonance> modes(voices * PaperModeCount);
    std::vector<Transient> bands(voices * PaperNoiseBandCount);
    const auto noise_one = CreateErbNoise(gpu, PaperNoiseBandCount, frames, rate);
    std::vector<float> noise(size_t(voices) * noise_one.size()), cpu(size_t(voices) * frames);
    for (uint32_t voice = 0; voice < voices; ++voice) {
        std::copy(noise_one.begin(), noise_one.end(), noise.begin() + voice * noise_one.size());
        for (uint32_t i = 0; i < PaperModeCount; ++i) modes[voice * PaperModeCount + i] = {float(100 + 479.13 * i + voice), -25.f - i, 40.f + i};
        for (uint32_t i = 0; i < PaperNoiseBandCount; ++i) bands[voice * PaperNoiseBandCount + i] = {-40.f - i, 400.f + i};
    }
    auto response = CreateResponseGpu(gpu, frames, voices, rate, modes, bands, noise);
    const auto render_cpu = [&] {
        for (uint32_t voice = 0; voice < voices; ++voice)
            EvaluateResponseFloat(std::span(modes).subspan(voice * PaperModeCount, PaperModeCount), std::span(bands).subspan(voice * PaperNoiseBandCount, PaperNoiseBandCount), frames, rate, std::span(noise).subspan(voice * noise_one.size(), noise_one.size()), std::span(cpu).subspan(voice * frames, frames));
    };
    const auto render_gpu = [&] { BeginGpu(gpu); EncodeResponse(gpu, response); SubmitGpu(gpu); WaitGpu(gpu); };
    render_cpu();
    render_gpu();
    std::vector<double> cpu_ms, gpu_ms;
    for (int repeat = 0; repeat < 3; ++repeat) {
        auto start = Clock::now();
        render_cpu();
        cpu_ms.push_back(Milliseconds(start));
        start = Clock::now();
        render_gpu();
        gpu_ms.push_back(Milliseconds(start));
    }
    std::ranges::sort(cpu_ms);
    std::ranges::sort(gpu_ms);
    double error = 0, energy = 0, maximum = 0;
    const auto actual = BufferSpan<float>(response.Output);
    for (size_t i = 0; i < cpu.size(); ++i) {
        const double delta = double(actual[i]) - cpu[i];
        error += delta * delta;
        energy += double(cpu[i]) * cpu[i];
        maximum = std::max(maximum, std::abs(delta));
    }
    std::cout << "{\"device\":\"" << DeviceName(gpu) << "\",\"voices\":" << voices << ",\"frames\":" << frames
              << ",\"modes\":15,\"bands\":30,\"precision\":\"float32\",\"cpu_median_ms\":" << cpu_ms[1]
              << ",\"gpu_submit_wait_median_ms\":" << gpu_ms[1] << ",\"relative_error\":" << std::sqrt(error / energy)
              << ",\"maximum_error\":" << maximum << "}\n";
    if (std::sqrt(error / energy) > 3e-4 || !std::isfinite(error)) throw std::runtime_error("Batch waveform mismatch");
}
}
int main(int argc, char **argv) {
    try {
        auto gpu = CreateGpu();
        std::cout << std::setprecision(12);
        if (argc == 5 && std::string_view(argv[1]) == "scrape") {
            Scrape(gpu, argv[2], argv[3], argv[4]);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "benchmark") {
            Benchmark(gpu);
            return 0;
        }
        if (argc != 4) throw std::invalid_argument("Usage: traerReproduce config.txt force.wav output-directory | benchmark");
        std::ifstream input(argv[1]);
        uint32_t frames{}, rate{}, mode_count{}, band_count{};
        input >> frames >> rate >> mode_count >> band_count;
        if (!input || mode_count > 1024 || band_count > 1024 || frames > 2000000) throw std::invalid_argument("Invalid render configuration");
        std::vector<Resonance> modes(mode_count);
        std::vector<Transient> bands(band_count);
        for (auto &mode : modes) input >> mode.Frequency >> mode.OnsetDb >> mode.DecayDbPerSecond >> mode.EndFrame;
        for (auto &band : bands) input >> band.OnsetDb >> band.DecayDbPerSecond;
        if (!input) throw std::invalid_argument("Truncated response configuration");
        auto force = ReadWave(argv[2]);
        if (force.SampleRate != rate || force.Channels != 1) throw std::invalid_argument("Force must be mono at response sample rate");
        const auto noise = bands.empty() ? std::vector<float>{} : CreateErbNoise(gpu, band_count, frames, rate, {.Seed = 2019});
        std::vector<double> reference(frames);
        EvaluateResponse(modes, bands, frames, rate, noise, reference);
        auto response = CreateResponseGpu(gpu, frames, 1, float(rate), modes, bands, noise);
        BeginGpu(gpu);
        EncodeResponse(gpu, response);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        const auto actual = BufferSpan<float>(response.Output);
        double error = 0, energy = 0;
        for (size_t i = 0; i < reference.size(); ++i) {
            error += std::pow(double(actual[i]) - reference[i], 2);
            energy += reference[i] * reference[i];
        }
        if (!std::isfinite(error) || error / std::max(energy, 1e-30) > 1e-6) throw std::runtime_error("GPU versus FP64 response mismatch");
        const auto output = std::filesystem::path(argv[3]);
        std::filesystem::create_directories(output);
        Save(output / "response.f64", reference);
        WriteWave(output / "response.wav", rate, 1, actual);
        auto result = ConvolveFixedGpu(gpu, force.Samples, actual);
        WriteWave(output / "synthesis.wav", rate, 1, result);
        std::cout << "{\"response_relative_error\":" << std::sqrt(error / std::max(energy, 1e-30)) << "}\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
