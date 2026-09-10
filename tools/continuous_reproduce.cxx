#include "continuous/Continuous.h"
#include "core/AudioFile.h"
#include "core/GpuConvolution.h"
#include "core/GpuModal.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <numeric>
#include <sstream>
#include <vector>

using namespace surface_audio;
using namespace surface_audio::continuous;
namespace {
void Benchmark() {
    constexpr uint32_t frames = 44100;
    auto gpu = CreateGpu();
    std::cout << std::setprecision(12) << "{\"scope\":\"FP32 continuous source only; 44100 frames per voice; construction and one GPU warmup excluded; GPU includes dispatch, completion and output copy; full waveform checked\",\"device\":" << std::quoted(DeviceName(gpu)) << ",\"runs\":[";
    for (const uint32_t voices : {1u, 64u}) {
        std::vector<Parameters> parameters;
        std::vector<State> states;
        for (uint32_t voice = 0; voice < voices; ++voice) {
            parameters.push_back(MakeParameters({.Angle = (voice % 4 ? voice % 4 - 1 : 0) * 2 * std::numbers::pi / 3, .Radius = voice % 4 ? 1. : 0.}));
            states.push_back(MakeState(7000 + voice));
        }
        const auto source = CreateGpuSource(gpu, parameters, states, frames);
        std::vector<float> cpu_output(size_t(voices) * frames), gpu_output(cpu_output.size());
        BeginGpu(gpu);
        EncodeSource(gpu, source);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        std::ranges::copy(states, BufferSpan<State>(source.States).begin());
        const auto cpu_begin = std::chrono::steady_clock::now();
        for (uint32_t voice = 0; voice < voices; ++voice) Render(parameters[voice], states[voice], std::span(cpu_output).subspan(size_t(voice) * frames, frames));
        const double cpu_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - cpu_begin).count();
        const auto gpu_begin = std::chrono::steady_clock::now();
        BeginGpu(gpu);
        EncodeSource(gpu, source);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        std::ranges::copy(BufferSpan<float>(source.Output), gpu_output.begin());
        const double gpu_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - gpu_begin).count();
        double square = 0, error = 0, maximum = 0;
        for (size_t index = 0; index < cpu_output.size(); ++index) {
            const double delta = double(gpu_output[index]) - cpu_output[index];
            if (!std::isfinite(delta)) throw std::runtime_error("Nonfinite continuous benchmark output");
            square += double(cpu_output[index]) * cpu_output[index];
            error += delta * delta;
            maximum = std::max(maximum, std::abs(delta));
        }
        const double relative = std::sqrt(error / std::max(square, 1e-30));
        if (relative >= .003 || maximum >= .025) throw std::runtime_error("Continuous benchmark waveform mismatch");
        if (voices != 1) std::cout << ',';
        std::cout << "{\"voices\":" << voices << ",\"frames_per_voice\":" << frames << ",\"cpu_seconds\":" << cpu_seconds << ",\"gpu_seconds\":" << gpu_seconds << ",\"relative_l2\":" << relative << ",\"max_absolute\":" << maximum << '}';
    }
    std::cout << "]}\n";
}
struct Segment {
    uint32_t Frames;
    Parameters Config;
};
std::vector<Segment> ReadControls(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open continuous controls");
    std::vector<Segment> result;
    uint64_t total = 0;
    for (std::string line; std::getline(input, line);) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        Controls c;
        uint32_t frames{};
        if (!(row >> frames >> c.Angle >> c.Radius >> c.Size >> c.Velocity >> c.Roughness >> c.Asymmetry >> c.ScratchDensity >> c.FrictionSigma >> c.FrictionDurationSamples >> c.CutoffHz >> c.Gain) || !frames) throw std::invalid_argument("Expected frames, angle, radius, size, velocity, roughness, asymmetry, scratch density, friction sigma, pulse samples, cutoff, gain");
        row >> std::ws;
        if (!row.eof()) throw std::invalid_argument("Extra continuous control field");
        total += frames;
        if (total > 120 * 44100) throw std::invalid_argument("Continuous render exceeds 120 seconds");
        result.push_back({frames, MakeParameters(c)});
    }
    if (result.empty()) throw std::invalid_argument("Empty continuous controls");
    return result;
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--benchmark") {
            Benchmark();
            return 0;
        }
        if (argc != 5) throw std::invalid_argument("Usage: continuousReproduce CONTROLS.txt RESPONSE.wav|MODES.txt OUTPUT.wav SEED; 44100 Hz; mode rows: frequency decay amplitude");
        const auto segments = ReadControls(argv[1]);
        const std::filesystem::path response_path = argv[2], output_path = argv[3];
        const uint64_t seed = std::stoull(argv[4]);
        constexpr uint32_t block = 256, rate = 44100;
        auto gpu = CreateGpu();
        const std::array initial{segments.front().Config};
        const std::array states{MakeState(seed)};
        const auto source = CreateGpuSource(gpu, initial, states, block);
        const bool modal = response_path.extension() == ".txt";
        GpuModal resonator;
        uint32_t tail_frames = 0;
        if (modal) {
            std::ifstream file(response_path);
            if (!file) throw std::runtime_error("Cannot open continuous modes");
            std::vector<Mode> modes;
            for (Mode mode; file >> mode.Frequency;) {
                if (!(file >> mode.Decay >> mode.Amplitude)) throw std::invalid_argument("Incomplete continuous mode row");
                modes.push_back(mode);
            }
            if (!file.eof()) throw std::invalid_argument("Invalid continuous mode row");
            for (const auto &mode : modes) {
                const double frames = std::ceil(std::log(1000.) * mode.Decay * rate);
                if (!std::isfinite(frames) || frames <= 0 || frames > 30 * rate) throw std::invalid_argument("Invalid continuous modal tail or decay exceeds 30 seconds");
                tail_frames = std::max(tail_frames, uint32_t(frames));
            }
            resonator = CreateGpuModal(gpu, MakeModalBank(modes, 1, rate), block);
        }
        const size_t source_frames = std::accumulate(segments.begin(), segments.end(), size_t{}, [](size_t total, const Segment &segment) { return total + segment.Frames; });
        std::vector<float> force, audio;
        force.reserve(source_frames);
        if (modal) audio.reserve(source_frames + tail_frames);
        const auto render = [&](uint32_t count, bool excite) {
            for (uint32_t begin = 0; begin < count; begin += block) {
                const uint32_t frames = std::min(block, count - begin);
                BufferSpan<uint32_t>(source.Block)[0] = frames;
                if (modal) BufferSpan<ModalBlock>(resonator.Parameters)[0].Frames = frames;
                BeginGpu(gpu);
                if (excite) EncodeSource(gpu, source);
                if (modal) EncodeModal(gpu, resonator, source.Output);
                SubmitGpu(gpu);
                WaitGpu(gpu);
                if (excite) {
                    const auto samples = BufferSpan<float>(source.Output).first(frames);
                    force.insert(force.end(), samples.begin(), samples.end());
                }
                if (modal) {
                    const auto sound = BufferSpan<float>(resonator.Output).first(frames);
                    audio.insert(audio.end(), sound.begin(), sound.end());
                }
            }
        };
        for (const auto &segment : segments) {
            BufferSpan<Parameters>(source.Parameters)[0] = segment.Config;
            render(segment.Frames, true);
        }
        if (modal) {
            std::ranges::fill(BufferSpan<float>(source.Output), 0);
            render(tail_frames, false);
        } else {
            const auto response = ReadWave(response_path);
            if (response.SampleRate != rate || response.Channels != 1) throw std::invalid_argument("Continuous response must be mono 44100 Hz");
            audio = ConvolveFixedGpu(gpu, force, response.Samples);
        }
        if (!output_path.parent_path().empty()) std::filesystem::create_directories(output_path.parent_path());
        WriteWave(output_path, rate, 1, audio);
        WriteWave(output_path.parent_path() / (output_path.stem().string() + "-force.wav"), rate, 1, force);
        const auto &state = BufferSpan<State>(source.States)[0];
        std::ofstream metadata(output_path.parent_path() / (output_path.stem().string() + ".json"));
        metadata << std::setprecision(12) << "{\"sample_rate\":44100,\"source_frames\":" << force.size() << ",\"output_frames\":" << audio.size() << ",\"seed\":" << seed << ",\"events\":" << state.Events << ",\"interval_clamps\":" << state.IntervalClamps << ",\"duration_clamps\":" << state.DurationClamps << ",\"lookahead_samples\":" << std::ceil(.5f * initial[0].MaximumDuration * rate) << ",\"object\":" << std::quoted(modal ? "exponential sinusoidal modal bank" : "supplied measured impulse response") << ",\"device\":" << std::quoted(DeviceName(gpu)) << "}\n";
        if (!metadata) throw std::runtime_error("Cannot write continuous metadata");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
