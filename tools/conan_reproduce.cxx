#include "conan/Conan.h"
#include "core/AudioFile.h"
#include "core/Gpu.h"
#include "core/GpuConvolution.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace surface_audio;
using namespace surface_audio::conan;
namespace fs = std::filesystem;

static std::vector<Impact> ReadImpacts(const fs::path &path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot read " + path.string());
    std::vector<Impact> result;
    for (std::string line; std::getline(input, line);) {
        std::ranges::replace(line, ',', ' ');
        std::istringstream values(line);
        Impact impact;
        if (!(values >> impact.Time >> impact.Amplitude >> impact.Duration)) throw std::runtime_error("Invalid impact CSV");
        result.push_back(impact);
    }
    return result;
}

static void WriteProcess(std::ostream &output, const Process &process) {
    output << "{\"mean\":" << process.Mean << ",\"sigma\":" << process.Sigma << ",\"a1\":" << process.A1 << ",\"b1\":" << process.B1 << ",\"empirical\":" << process.Empirical << ",\"quantiles\":[";
    for (unsigned index = 0; index < QuantileCount; ++index) output << (index ? "," : "") << process.Quantiles[index];
    output << "]}";
}

static std::vector<float> Synthesize(Gpu &gpu, const Parameters &parameters, const fs::path &output, const std::string &name, std::ostream &metadata) {
    constexpr unsigned voices = 8, warmup = 88200, samples = 441000, frames = warmup + samples;
    std::array<Parameters, voices> models;
    models.fill(parameters);
    std::array<State, voices> states;
    for (unsigned voice = 0; voice < voices; ++voice) states[voice] = MakeState(2014 + voice, 1);
    auto cpu = states[0];
    const auto models_buffer = Upload<Parameters>(gpu, models), states_buffer = Upload<State>(gpu, states), frames_buffer = Upload(gpu, frames);
    const auto output_buffer = CreateBuffer(gpu, size_t(voices) * frames * sizeof(float));
    const std::array bindings{GpuBinding{models_buffer, 0}, GpuBinding{states_buffer, 1}, GpuBinding{output_buffer, 2}, GpuBinding{frames_buffer, 3}};
    BeginGpu(gpu);
    DispatchGpu(gpu, CreateKernel(gpu, "ConanSynthesize"), bindings, {voices}, {voices});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto generated = BufferSpan<float>(output_buffer);
    const auto terminal = BufferSpan<State>(states_buffer);
    for (float sample : generated)
        if (!std::isfinite(sample)) throw std::runtime_error("Nonfinite Conan GPU output");
    double difference = 0, energy = 0, maximum = 0;
    for (unsigned sample = 0; sample < frames; ++sample) {
        const float reference = Step(parameters, cpu);
        if (!std::isfinite(reference)) throw std::runtime_error("Nonfinite Conan CPU output");
        const double error = generated[sample] - reference;
        difference += error * error;
        energy += double(reference) * reference;
        maximum = std::max(maximum, std::abs(error));
    }
    const double relative = std::sqrt(difference / energy);
    if (!(energy > 0) || !std::isfinite(energy) || !std::isfinite(relative) || !std::isfinite(maximum) || relative > .002 || maximum > .004) throw std::runtime_error("Conan CPU/GPU full-record disagreement");
    metadata << "{\"amplitude\":";
    WriteProcess(metadata, parameters.Amplitude);
    metadata << ",\"interval\":";
    WriteProcess(metadata, parameters.Interval);
    metadata << ",\"duration_scale\":" << parameters.DurationScale << ",\"duration_exponent\":" << parameters.DurationExponent
             << ",\"modulation_depth\":" << parameters.ModulationDepth << ",\"warmup_seconds\":2,\"seconds_per_seed\":10,\"seed_base\":2014,\"cpu_gpu_relative_rms_error\":" << relative << ",\"cpu_gpu_max_error\":" << maximum << ",\"seeds\":[";
    for (unsigned voice = 0; voice < voices; ++voice) {
        const auto &state = terminal[voice];
        WriteWave(output / (name + "-seed" + std::to_string(voice) + ".wav"), 44100, 1, generated.subspan(size_t(voice) * frames + warmup, samples));
        if (voice) metadata << ',';
        metadata << "{\"seed\":" << 2014 + voice << ",\"events_including_warmup\":" << state.Events << ",\"amplitude_clamps\":" << state.AmplitudeClamps << ",\"interval_clamps\":" << state.IntervalClamps << ",\"duration_clamps\":" << state.DurationClamps << ",\"pulse_overflows\":" << state.PulseOverflows << '}';
        if (state.PulseOverflows) throw std::runtime_error("Pulse capacity overflow");
    }
    metadata << "]}";
    std::cout << name << " CPU/GPU relative RMS " << relative << ", duration scale " << parameters.DurationScale << '\n';
    return {generated.begin() + warmup, generated.begin() + frames};
}

int main(int argc, char **argv) {
    try {
        if (argc > 2 && (argc != 4 || std::string(argv[2]) != "--parameters")) throw std::invalid_argument("Usage: conanReproduce [OUTPUT_DIR [--parameters FILE]]");
        const fs::path output = argc > 1 ? argv[1] : "outputs/reproduction/conan";
        fs::create_directories(output);
        auto gpu = CreateGpu();
        std::ofstream metadata(output / "parameters.json");
        if (!metadata) throw std::runtime_error("Cannot write reproduction metadata");
        metadata << std::setprecision(12) << "{\"device\":\"" << DeviceName(gpu) << "\",\"cases\":{";
        bool first = true;
        if (argc == 4) {
            std::ifstream input(argv[3]);
            if (!input) throw std::runtime_error("Cannot read supplied Conan parameters");
            std::vector<std::string> names;
            for (std::string line; std::getline(input, line);) {
                if (line.empty() || line.front() == '#') continue;
                std::istringstream values(line);
                std::string name, extra;
                auto parameters = MakeParameters({.Asymmetry = 0}, 44100);
                auto &a = parameters.Amplitude;
                auto &t = parameters.Interval;
                if (!(values >> name >> a.Mean >> a.Sigma >> a.A1 >> a.B1 >> t.Mean >> t.Sigma >> t.A1 >> t.B1 >> parameters.DurationScale) || values >> extra || name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") != std::string::npos) throw std::runtime_error("Invalid supplied Conan parameter row");
                if (std::ranges::find(names, name) != names.end()) throw std::runtime_error("Duplicate supplied Conan parameter name");
                names.push_back(name);
                Validate(parameters);
                if (!first) metadata << ',';
                first = false;
                metadata << '"' << name << "\":";
                Synthesize(gpu, parameters, output, name, metadata);
            }
            if (first) throw std::runtime_error("No supplied Conan parameter rows");
            metadata << "}}\n";
            return 0;
        }
        const std::array responses{ReadWave(output / "ir1-calibrated.wav"), ReadWave(output / "ir2-calibrated.wav")};
        for (const auto &response : responses)
            if (response.SampleRate != 44100 || response.Channels != 1) throw std::runtime_error("Response must be mono 44100 Hz");
        const auto render_responses = [&](std::span<const float> force, const std::string &name) {
            for (size_t index = 0; index < responses.size(); ++index) {
                // Four seconds of newly generated force followed by the complete response tail.
                const auto sound = ConvolveFixedGpu(gpu, force.first(4 * 44100), responses[index].Samples);
                WriteWave(output / (name + "-ir" + std::to_string(index + 1) + ".wav"), 44100, 1, sound);
            }
        };
        for (const std::string source : {"physical", "force1", "force2", "physical-full", "force1-full", "force2-full"}) {
            const auto impacts = ReadImpacts(output / (source + (source.ends_with("-full") ? ".csv" : "-train.csv")));
            for (bool empirical : {false, true}) {
                const std::string name = source + (empirical ? "-empirical" : "-gaussian");
                auto parameters = Calibrate(impacts, 44100, empirical);
                if (!first) metadata << ',';
                first = false;
                metadata << '"' << name << "\":";
                const auto force = Synthesize(gpu, parameters, output, name, metadata);
                if (source.starts_with("physical")) render_responses(force, name);
            }
        }
        metadata << ",\"paper-default\":";
        const auto paper = MakeParameters({.Size = .5, .Velocity = .5, .Roughness = .5, .Asymmetry = 0}, 44100);
        const auto force = Synthesize(gpu, paper, output, "paper-default", metadata);
        render_responses(force, "paper-default");
        metadata << "}}\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
