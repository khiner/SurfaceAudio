#include "core/AudioFile.h"
#include "lagrange/Lagrange.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace surface_audio;
using namespace surface_audio::lagrange;
namespace {
void Benchmark() {
    auto gpu = CreateGpu();
    std::vector<Resonance> modes;
    for (uint32_t n = 0; n < 80; ++n) modes.push_back({101.137 + 251.113 * n, .3 + .17 * n, .1 / (n + 1), -.05 / (n + 1)});
    constexpr uint32_t frames = 88200, rate = 44100;
    const auto reference = ModalResponse(modes, frames, rate), warm = ModalResponseGpu(gpu, modes, frames, rate);
    double error = 0, energy = 0;
    for (size_t n = 0; n < frames; ++n) {
        error += std::pow(double(reference[n]) - warm[n], 2);
        energy += double(reference[n]) * reference[n];
    }
    std::array<double, 3> cpu{}, metal{};
    for (size_t repeat = 0; repeat < cpu.size(); ++repeat) {
        const auto begin_cpu = std::chrono::steady_clock::now();
        const auto cpu_result = ModalResponse(modes, frames, rate);
        cpu[repeat] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin_cpu).count();
        const auto begin_metal = std::chrono::steady_clock::now();
        const auto gpu_result = ModalResponseGpu(gpu, modes, frames, rate);
        metal[repeat] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin_metal).count();
        if (cpu_result.size() != gpu_result.size()) throw std::runtime_error("Benchmark dimensions changed");
    }
    std::ranges::sort(cpu);
    std::ranges::sort(metal);
    std::cout << std::setprecision(12) << "{\"frames\":" << frames << ",\"modes\":" << modes.size()
              << ",\"cpu_fp64_ms\":" << cpu[1] << ",\"metal_fp32_ms\":" << metal[1]
              << ",\"relative_error\":" << std::sqrt(error / energy) << ",\"scope\":\"Complete Eq2 response, allocations and GPU transfer included\"}\n";
}
}
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "benchmark") {
            Benchmark();
            return 0;
        }
        if (argc != 4) throw std::invalid_argument("Usage: lagrangeReproduce input.wav settings.txt output_directory");
        const auto wave = ReadWave(argv[1]);
        if (wave.Channels != 1) throw std::invalid_argument("Lagrange analysis requires mono input");
        AnalysisOptions options;
        std::ifstream settings(argv[2]);
        settings >> options.ModalInterval.Begin >> options.ModalInterval.End >> options.ImpactInterval.Begin >> options.ImpactInterval.End;
        if (!settings) throw std::invalid_argument("Settings require modal_begin modal_end impact_begin impact_end samples");
        uint32_t gains = 0;
        settings >> gains;
        options.Gains = ModalGain(gains);
        uint32_t fit = 0;
        settings >> fit >> options.CenterImpact;
        settings >> options.AudioWeight;
        options.Fit = TriggerFit(fit);
        const std::filesystem::path output(argv[3]);
        std::filesystem::create_directories(output);
        auto gpu = CreateGpu();
        const auto start = std::chrono::steady_clock::now();
        const auto analysis = Analyze(gpu, wave.Samples, wave.SampleRate, options);
        const double analysis_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const auto synthesis_start = std::chrono::steady_clock::now();
        const auto synthesis = Synthesize(gpu, analysis, analysis.Triggers);
        const double synthesis_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - synthesis_start).count();
        const auto replay = FilterModalGpu(gpu, analysis.Excitation, analysis.Modes, 2 * analysis.Frames - 1, wave.SampleRate);
        WriteWave(output / "synthesis.wav", wave.SampleRate, 1, synthesis);
        WriteWave(output / "source_replay.wav", wave.SampleRate, 1, replay);
        WriteWave(output / "excitation.wav", wave.SampleRate, 1, analysis.Excitation);
        WriteWave(output / "impact.wav", wave.SampleRate, 1, analysis.Impact);
        WriteWave(output / "envelope.wav", wave.SampleRate, 1, analysis.Envelope);
        WriteWave(output / "impact_envelope.wav", wave.SampleRate, 1, analysis.ImpactEnvelope);
        WriteWave(output / "deconvolved.wav", wave.SampleRate, 1, analysis.Deconvolved);
        std::ofstream modes(output / "modes.txt"), triggers(output / "triggers.txt");
        modes << std::setprecision(17);
        triggers << std::setprecision(17);
        for (auto mode : analysis.Modes) modes << mode.Frequency << ' ' << mode.Damping << ' ' << mode.RealGain << ' ' << mode.ImaginaryGain << '\n';
        for (auto trigger : analysis.Triggers) triggers << trigger.Sample << ' ' << trigger.Amplitude << '\n';
        std::cout << std::setprecision(12) << "{\"analysis_ms\":" << analysis_ms << ",\"synthesis_ms\":" << synthesis_ms
                  << ",\"source_fit_kkt\":" << analysis.SourceFitKkt
                  << ",\"modes\":" << analysis.Modes.size() << ",\"triggers\":" << analysis.Triggers.size() << ",\"device\":\"" << DeviceName(gpu) << "\"}\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
