#include "core/AudioFile.h"
#include "poirot/Poirot.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
using namespace surface_audio;
using namespace surface_audio::poirot;
static void RenderGpu(Gpu &gpu, GpuSignal &bank, std::span<float> output) {
    const size_t frames = output.size() / bank.Voices;
    for (size_t offset = 0; offset < frames;) {
        const uint32_t count = std::min<size_t>(bank.Capacity, frames - offset);
        BeginGpu(gpu);
        EncodeSignal(gpu, bank, count);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        const auto block = BufferSpan<float>(bank.Output);
        for (uint32_t voice = 0; voice < bank.Voices; ++voice) std::copy_n(block.begin() + size_t(voice) * count, count, output.begin() + size_t(voice) * frames + offset);
        offset += count;
    }
}
static void Benchmark() {
    auto gpu = CreateGpu();
    std::cout << "Device " << DeviceName(gpu) << '\n';
    auto modes = StringModes({});
    for (size_t i = 0; i < modes.size(); ++i) modes[i].Amplitude = 1.f / (i + 1);
    const SignalParameters parameters{.SplitThreshold = 0, .SplitSlope = 10, .PowerScale = 10000, .ShapeWeightedSplit = true};
    constexpr uint32_t frames = 44100, capacity = 4096;
    for (uint32_t voices : {1u, 64u}) {
        const std::vector<SignalState> initial(voices, MakeSignal(parameters, modes));
        std::vector<float> cpu(size_t(voices) * frames), device(cpu.size());
        std::array<double, 3> cpu_times{}, gpu_times{};
        double relative_error = 0;
        for (unsigned repeat = 0; repeat < 4; ++repeat) {
            auto states = initial;
            auto bank = CreateGpuSignal(gpu, initial, capacity);
            const auto start_cpu = std::chrono::steady_clock::now();
            for (uint32_t v = 0; v < voices; ++v) RenderSignal(states[v], std::span(cpu).subspan(size_t(v) * frames, frames));
            const auto stop_cpu = std::chrono::steady_clock::now();
            RenderGpu(gpu, bank, device);
            const auto stop_gpu = std::chrono::steady_clock::now();
            if (repeat) {
                cpu_times[repeat - 1] = std::chrono::duration<double, std::milli>(stop_cpu - start_cpu).count();
                gpu_times[repeat - 1] = std::chrono::duration<double, std::milli>(stop_gpu - stop_cpu).count();
            }
            double error = 0, norm = 0;
            for (size_t i = 0; i < cpu.size(); ++i) {
                error += std::pow(cpu[i] - device[i], 2);
                norm += cpu[i] * cpu[i];
            }
            relative_error = std::sqrt(error / norm);
            if (!std::isfinite(relative_error) || relative_error > .005) throw std::runtime_error("Benchmark GPU waveform mismatch");
        }
        std::sort(cpu_times.begin(), cpu_times.end());
        std::sort(gpu_times.begin(), gpu_times.end());
        std::cout << "voices " << voices << " modes " << modes.size() << " frames " << frames << " CPU_median_ms " << cpu_times[1] << " GPU_median_ms " << gpu_times[1] << " relative_error " << relative_error << '\n';
    }
}
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--benchmark") {
            Benchmark();
            return 0;
        }
        if (argc != 5) throw std::invalid_argument("Usage: poirotReproduce signal|physical CONFIG OUTPUT.wav SECONDS");
        const double seconds = std::stod(argv[4]);
        if (!(seconds > 0 && seconds <= 60)) throw std::invalid_argument("Invalid duration");
        std::ifstream config(argv[2]);
        if (!config) throw std::runtime_error("Cannot open config");
        std::vector<float> output(size_t(seconds * 44100));
        if (std::string(argv[1]) == "signal") {
            SignalParameters p;
            unsigned shape{}, reset{}, count{};
            config >> p.Activation >> p.Lambda >> p.Height >> p.Position >> p.SplitThreshold >> p.SplitSlope >> p.PowerScale >> p.ReturnGain >> shape >> reset >> count;
            p.ShapeWeightedSplit = shape;
            p.ResetPhaseAtActivation = reset;
            if (!count || count > 1024) throw std::invalid_argument("Invalid mode count");
            std::vector<surface_audio::poirot::Mode> modes(count);
            for (auto &m : modes) config >> m.Frequency >> m.Damping >> m.Amplitude >> m.Phase;
            if (!config) throw std::invalid_argument("Invalid signal config");
            const auto state = MakeSignal(p, modes);
            auto gpu = CreateGpu();
            auto synth = CreateGpuSignal(gpu, std::span(&state, 1), 4096);
            RenderGpu(gpu, synth, output);
        } else if (std::string(argv[1]) == "physical") {
            StringParameters p;
            double height_ratio{};
            config >> p.WaveSpeed >> p.Stiffness >> p.Loss0 >> p.Loss1 >> p.ObstaclePosition >> height_ratio >> p.ReadoutPosition;
            if (!config || !std::isfinite(height_ratio) || height_ratio < 0) throw std::invalid_argument("Invalid physical config");
            auto probe_parameters = p;
            probe_parameters.ContactStiffness = 0;
            probe_parameters.ReadoutPosition = p.ObstaclePosition;
            auto probe = MakeString(probe_parameters);
            std::vector<float> before(size_t((p.Activation + .005) * p.SampleRate));
            RenderString(probe, before);
            double envelope = 0;
            for (size_t i = size_t((p.Activation - .5 / 404) * p.SampleRate); i < size_t((p.Activation + .5 / 404) * p.SampleRate); ++i) envelope = std::max(envelope, double(std::abs(before[i])));
            p.ObstacleHeight = height_ratio * envelope;
            auto state = MakeString(p);
            RenderString(state, output);
            std::cout << "obstacle_height_m " << p.ObstacleHeight << " maximum_penetration_m " << state.MaximumPenetration << '\n';
        } else throw std::invalid_argument("Unknown model");
        if (!std::ranges::all_of(output, [](float x) { return std::isfinite(x); })) throw std::runtime_error("Nonfinite synthesis");
        WriteWave(argv[3], 44100, 1, output);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
