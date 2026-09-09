#include "agarwal/ContactFit.h"
#include "core/Adam.h"
#include "core/AudioFile.h"
#include "core/BinaryFile.h"
#include "core/GpuSpectral.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

using namespace surface_audio;
using namespace surface_audio::agarwal;
namespace {
using Clock = std::chrono::steady_clock;

std::vector<ContactFitMode> ReadModes(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open contact mode file");
    std::vector<ContactFitMode> modes;
    for (std::string line; std::getline(input, line);) {
        if (line.find_first_not_of(" \t\r") == std::string::npos || line[line.find_first_not_of(" \t\r")] == '#') continue;
        std::istringstream row(line);
        float frequency, decay, amplitude;
        if (!(row >> frequency >> decay >> amplitude)) throw std::invalid_argument("Expected frequency, decay seconds, amplitude in every mode row");
        // The optional second endpoint amplitude is unused for this fixed response.
        float ignored;
        row >> std::ws;
        if (!row.eof() && !(row >> ignored)) throw std::invalid_argument("Invalid optional mode endpoint amplitude");
        row >> std::ws;
        if (!row.eof()) throw std::invalid_argument("Too many mode columns");
        modes.push_back({frequency, decay, amplitude});
    }
    if (modes.empty() || modes.size() > 50) throw std::invalid_argument("Contact fitting requires one to fifty modes");
    return modes;
}

void WriteModes(const std::filesystem::path &path, std::span<const ContactFitMode> initial, std::span<const float> parameters) {
    std::ofstream output(path);
    output << std::setprecision(10);
    for (size_t index = 0; index < initial.size(); ++index) {
        const float amplitude = std::exp(parameters[index]), decay = std::exp(parameters[initial.size() + index]);
        output << initial[index].Frequency << ' ' << decay << ' ' << amplitude << ' ' << amplitude << '\n';
    }
    if (!output) throw std::runtime_error("Cannot write fitted contact modes");
}

uint32_t Integer(const char *text) {
    size_t consumed = 0;
    const unsigned long long value = std::stoull(text, &consumed);
    if (consumed != std::char_traits<char>::length(text) || value > UINT32_MAX || text[0] == '-') throw std::invalid_argument("Invalid nonnegative integer argument");
    return uint32_t(value);
}

void Fit(const std::filesystem::path &force_path, const std::filesystem::path &mode_path, const std::filesystem::path &target_path, const std::filesystem::path &directory, uint32_t steps, uint32_t taps, double learning_rate, std::string_view scale) {
    const auto force = ReadBinary<float>(force_path);
    const auto modes = ReadModes(mode_path);
    const auto target = ReadWave(target_path);
    if (target.Channels != 1 || !target.SampleRate || target.Samples.empty() || !taps || force.empty() || target.Samples.size() > force.size() + size_t(taps) - 1 || !std::isfinite(learning_rate) || learning_rate <= 0) throw std::invalid_argument("Invalid target extent, taps or optimizer settings");
    const auto target_samples = [&] {
        std::vector<float> padded(force.size() + size_t(taps) - 1, 0);
        std::ranges::copy(target.Samples, padded.begin());
        return padded;
    }();
    const SpectralLossOptions options{.Scale = scale == "linear" ? SpectralMagnitudeScale::Linear : scale == "ln" ? SpectralMagnitudeScale::NaturalLog :
                                                                                                                    SpectralMagnitudeScale::Decibels};
    auto gpu = CreateGpu();
    const auto contact = CreateContactFitGpu(gpu, force, modes, target.SampleRate, taps);
    const auto spectral = CreateSpectralLossGpu(gpu, target_samples, target.SampleRate, options);
    const auto parameters = BufferSpan<float>(contact.Parameters);
    const auto decay_bounds = ContactFitLogDecayBounds();
    const auto bounds = std::views::iota(size_t(0), parameters.size()) | std::views::transform([&](size_t index) { return index < modes.size() ? std::array{double(ContactFitMinimumLogAmplitude), double(ContactFitMaximumLogAmplitude)} : std::array{double(decay_bounds[0]), double(decay_bounds[1])}; }) | std::ranges::to<std::vector>();
    auto optimizer = CreateAdam(parameters, bounds);
    std::vector<float> best(parameters.begin(), parameters.end());
    double best_loss = std::numeric_limits<double>::infinity(), initial_loss = 0, final_loss = 0;
    uint32_t best_step = 0;
    uint64_t bound_updates = 0;
    std::filesystem::create_directories(directory);
    WriteWave(directory / "target-padded.wav", target.SampleRate, 1, target_samples);
    std::ofstream trace(directory / "loss.csv");
    trace << "step,total,fft4096,fft1024,fft256,fft64,seconds\n"
          << std::setprecision(12);
    const auto start = Clock::now();
    for (uint32_t step = 0;; ++step) {
        BeginGpu(gpu);
        EncodeContactFit(gpu, contact);
        EncodeSpectralLoss(gpu, spectral, contact.Output);
        if (step < steps) EncodeContactFitGradient(gpu, contact, spectral.Gradient);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        const auto loss = BufferSpan<float>(spectral.Loss);
        final_loss = loss[0];
        if (!std::isfinite(final_loss)) throw std::runtime_error("Nonfinite contact fitting loss");
        if (final_loss < best_loss) {
            best_loss = final_loss;
            best_step = step;
            std::ranges::copy(parameters, best.begin());
        }
        if (!step) {
            initial_loss = final_loss;
            WriteWave(directory / "initial.wav", target.SampleRate, 1, BufferSpan<float>(contact.Output));
        }
        if (step % 100 == 0 || step == steps) {
            trace << step;
            for (float value : loss) trace << ',' << value;
            trace << ',' << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
            trace.flush();
        }
        if (step % 1000 == 0 || step == steps) std::cout << "contact step " << step << " loss " << final_loss << " best " << best_loss << std::endl;
        if (step == steps) break;
        bound_updates += UpdateAdam(optimizer, BufferSpan<float>(contact.Gradient), parameters, learning_rate);
    }
    WriteModes(directory / "last-modes.txt", modes, parameters);
    WriteBinary<float>(directory / "last-parameters.f32", parameters);
    WriteModes(directory / "fitted-modes.txt", modes, best);
    WriteBinary<float>(directory / "fitted-parameters.f32", best);
    std::ranges::copy(best, parameters.begin());
    BeginGpu(gpu);
    EncodeContactFit(gpu, contact);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    WriteWave(directory / "fitted.wav", target.SampleRate, 1, BufferSpan<float>(contact.Output));
    std::ofstream metadata(directory / "fit.json");
    metadata << std::setprecision(12) << "{\n  \"force_file\": " << std::quoted(std::filesystem::absolute(force_path).string())
             << ",\n  \"mode_file\": " << std::quoted(std::filesystem::absolute(mode_path).string()) << ",\n  \"target_file\": " << std::quoted(std::filesystem::absolute(target_path).string())
             << ",\n  \"modes\": " << modes.size() << ",\n  \"sample_rate\": " << target.SampleRate << ",\n  \"force_frames\": " << force.size() << ",\n  \"target_frames\": " << target.Samples.size()
             << ",\n  \"taps\": " << taps << ",\n  \"output_frames\": " << contact.Frames << ",\n  \"target_padding_frames\": " << target_samples.size() - target.Samples.size()
             << ",\n  \"steps\": " << steps << ",\n  \"learning_rate\": " << learning_rate << ",\n  \"initial_loss\": " << initial_loss << ",\n  \"final_loss\": " << final_loss
             << ",\n  \"best_loss\": " << best_loss << ",\n  \"best_step\": " << best_step << ",\n  \"bound_updates\": " << bound_updates
             << ",\n  \"seconds\": " << std::chrono::duration<double>(Clock::now() - start).count() << ",\n  \"device\": " << std::quoted(DeviceName(gpu))
             << ",\n  \"optimizer\": \"Adam beta1=.9 beta2=.999 epsilon=1e-8; double master log amplitudes and log decay seconds\""
             << ",\n  \"log_amplitude_bounds\": [-30,20],\n  \"decay_seconds_bounds\": [" << ContactFitMinimumDecay << ',' << ContactFitMaximumDecay << ']'
             << ",\n  \"log_decay_bounds\": [" << decay_bounds[0] << ',' << decay_bounds[1] << ']'
             << ",\n  \"decay_quantization\": \"Physical bounds are float constants; natural-log bounds round inward to float so exponentiated parameters remain accepted on mode-file reload\""
             << ",\n  \"loss_convention\": \"sum of four mean Huber losses; delta1; periodic Hann; quarter-window hop; centered zero padding; unnormalized one-sided " << scale << " magnitude; floor1e-6\""
             << ",\n  \"target_tail_convention\": \"Reference samples followed by explicit zeros to forceFrames+taps-1; the complete rendered tail participates in fitting\""
             << ",\n  \"scope\": \"Fixed finite exponential-sine contact IR; supplied force and frequencies fixed; target supplies loss only\"\n}\n";
    if (!metadata || !trace) throw std::runtime_error("Cannot write contact fitting metadata");
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc < 7 || argc > 9) throw std::invalid_argument("Usage: agarwalContactFit FORCE.f32 MODES.txt TARGET.wav OUTPUT STEPS TAPS [LEARNING_RATE [linear|ln|db]]");
        const std::string_view scale = argc == 9 ? argv[8] : "linear";
        if (scale != "linear" && scale != "ln" && scale != "db") throw std::invalid_argument("Loss scale must be linear, ln, or db");
        Fit(argv[1], argv[2], argv[3], argv[4], Integer(argv[5]), Integer(argv[6]), argc >= 8 ? std::stod(argv[7]) : .01, scale);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
