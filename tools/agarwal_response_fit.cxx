#include "agarwal/Response.h"
#include "core/AudioFile.h"
#include "core/BinaryFile.h"
#include "core/GpuSpectral.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <ranges>
#include <string>

using namespace surface_audio;
using namespace surface_audio::agarwal;
namespace {
using Clock = std::chrono::steady_clock;

std::vector<float> ReadParameters(const std::filesystem::path &path) {
    auto values = ReadBinary<float>(path);
    if (values.size() != ResponseParameterCount) throw std::invalid_argument("Expected 50 response parameters");
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) throw std::invalid_argument("Nonfinite response parameter");
        if ((i < 10 && (values[i] <= 0 || values[i] >= 22050)) || (((i >= 20 && i < 30) || i >= 40) && values[i] <= 0)) throw std::invalid_argument("Invalid response frequency or RT60");
    }
    return values;
}

void SetParameters(const ResponseGpu &response, std::span<const float> values) { std::ranges::copy(values, BufferSpan<float>(response.Parameters).begin()); }

void Fit(const std::filesystem::path &input, const std::filesystem::path &initial_path, const std::filesystem::path &output, uint32_t steps, uint64_t seed, double learning_rate, std::string_view scale_mode, std::string_view loss_scale) {
    const auto target = ReadWave(input);
    if (target.Channels != 1 || target.SampleRate != 44100 || target.Samples.empty() || target.Samples.size() > UINT32_MAX) throw std::invalid_argument("Fit requires nonempty mono 44100 Hz WAV");
    for (float x : target.Samples)
        if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite target waveform");
    if (!steps || !std::isfinite(learning_rate) || learning_rate <= 0) throw std::invalid_argument("Invalid optimizer settings");
    auto parameters = ReadParameters(initial_path), best = parameters;
    std::filesystem::create_directories(output);
    auto gpu = CreateGpu();
    const ResponseNoiseSettings noise_settings{.Seed = seed};
    const auto noise = CreateResponseNoise(gpu, uint32_t(target.Samples.size()), float(target.SampleRate), noise_settings);
    const auto response = CreateResponseGpu(gpu, uint32_t(target.Samples.size()), float(target.SampleRate), noise);
    const SpectralLossOptions loss_options{.Scale = loss_scale == "linear" ? SpectralMagnitudeScale::Linear : loss_scale == "ln" ? SpectralMagnitudeScale::NaturalLog :
                                                                                                                                   SpectralMagnitudeScale::Decibels};
    const auto spectral = CreateSpectralLossGpu(gpu, target.Samples, target.SampleRate, loss_options);
    const bool scaled = scale_mode != "physical", log_decay = scale_mode == "log-decay";
    auto master = std::views::iota(size_t{0}, parameters.size()) | std::views::transform([&](size_t i) { return log_decay && ((i >= 20 && i < 30) || i >= 40) ? std::log(double(parameters[i])) : double(parameters[i]); }) | std::ranges::to<std::vector>();
    std::array<double, ResponseParameterCount> first{}, second{};
    double best_loss = std::numeric_limits<double>::infinity(), initial_loss = 0, final_loss = 0;
    uint32_t best_step = 0;
    uint64_t bound_updates = 0;
    const auto start = Clock::now();
    std::ofstream trace(output / "loss.csv");
    trace << "step,total,fft4096,fft1024,fft256,fft64,seconds\n"
          << std::setprecision(10);
    for (uint32_t step = 0; step <= steps; ++step) {
        SetParameters(response, parameters);
        BeginGpu(gpu);
        EncodeResponse(gpu, response);
        EncodeSpectralLoss(gpu, spectral, response.Output);
        if (step < steps) EncodeResponseGradient(gpu, response, spectral.Gradient);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        const auto losses = BufferSpan<float>(spectral.Loss);
        final_loss = losses[0];
        if (!std::isfinite(final_loss)) throw std::runtime_error("Nonfinite fitting loss");
        if (final_loss < best_loss) {
            best_loss = final_loss;
            best = parameters;
            best_step = step;
        }
        if (!step) {
            initial_loss = final_loss;
            WriteWave(output / "initial.wav", target.SampleRate, 1, BufferSpan<float>(response.Output));
        }
        if (step % 100 == 0 || step == steps) {
            trace << step;
            for (float value : losses) trace << ',' << value;
            trace << ',' << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
            trace.flush();
        }
        if (step % 1000 == 0 || step == steps) std::cout << input.filename().string() << " step " << step << " loss " << final_loss << " best " << best_loss << std::endl;
        if (step == steps) break;
        const auto gradient = BufferSpan<float>(response.Gradient);
        const double correction1 = 1 - std::pow(.9, double(step + 1)), correction2 = 1 - std::pow(.999, double(step + 1));
        for (size_t i = 0; i < parameters.size(); ++i) {
            const double scale = scaled && (i < 20 || (i >= 30 && i < 40)) ? 100 : 1;
            const bool decay = (i >= 20 && i < 30) || i >= 40;
            const double derivative = gradient[i] * scale * (log_decay && decay ? parameters[i] : 1);
            if (!std::isfinite(derivative)) throw std::runtime_error("Nonfinite fitting gradient");
            first[i] = .9 * first[i] + .1 * derivative;
            second[i] = .999 * second[i] + .001 * derivative * derivative;
            const double proposal = master[i] - learning_rate * scale * (first[i] / correction1) / (std::sqrt(second[i] / correction2) + 1e-8);
            // Bounds constrain numerical inference, not the forward response equations.
            const double lower = i < 10 ? 1 : decay ? 1e-5 :
                                                      -160;
            const double upper = i < 10 ? target.SampleRate * .5 - 1 : decay ? 20 :
                                                                               60;
            master[i] = std::clamp(proposal, log_decay && decay ? std::log(lower) : lower, log_decay && decay ? std::log(upper) : upper);
            bound_updates += master[i] != proposal;
            parameters[i] = float(log_decay && decay ? std::exp(master[i]) : master[i]);
        }
    }
    WriteBinary<float>(output / "last.f32", parameters);
    WriteBinary<float>(output / "fitted.f32", best);
    SetParameters(response, best);
    BeginGpu(gpu);
    EncodeResponse(gpu, response);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    WriteWave(output / "fitted.wav", target.SampleRate, 1, BufferSpan<float>(response.Output));
    std::ofstream metadata(output / "fit.json");
    metadata << std::setprecision(12) << "{\n  \"steps\": " << steps << ",\n  \"seed\": " << seed
             << ",\n  \"learning_rate\": " << learning_rate << ",\n  \"parameter_scale_mode\": \"" << scale_mode
             << "\",\n  \"initial_loss\": " << initial_loss << ",\n  \"final_loss\": " << final_loss << ",\n  \"best_loss\": " << best_loss
             << ",\n  \"best_step\": " << best_step << ",\n  \"bound_updates\": " << bound_updates
             << ",\n  \"seconds\": " << std::chrono::duration<double>(Clock::now() - start).count()
             << ",\n  \"device\": \"" << DeviceName(gpu) << "\",\n  \"optimizer\": \"Adam beta1=0.9 beta2=0.999 epsilon=1e-8; double master parameters\",\n"
             << "  \"parameter_bounds\": {\"frequency_hz\": [1,22049], \"amplitude_db\": [-160,60], \"rt60_seconds\": [0.00001,20]},\n"
             << "  \"loss_convention\": \"sum of four mean Huber losses, delta 1, periodic Hann, quarter-window hop, centered zero padding, unnormalized one-sided " << loss_scale << " magnitude, magnitude floor 1e-6\",\n"
             << "  \"noise_convention\": \"fixed shared Gaussian excitation, 513-tap centered Hamming FIR, ten ERB bands from zero to Nyquist, unit expected RMS per band\",\n"
             << "  \"scope\": \"known-recording reconstruction; not author parameters or a perceptual equivalence score\"\n}\n";
    if (!metadata || !trace) throw std::runtime_error("Cannot write fitting metadata");
}

void Sample(const std::filesystem::path &parameters_path, const std::filesystem::path &output, uint32_t frames, uint64_t seed) {
    const auto parameters = ReadParameters(parameters_path);
    auto gpu = CreateGpu();
    const ResponseNoiseSettings settings{.Seed = seed};
    const auto noise = CreateResponseNoise(gpu, frames, 44100, settings);
    const auto response = CreateResponseGpu(gpu, frames, 44100, noise);
    SetParameters(response, parameters);
    BeginGpu(gpu);
    EncodeResponse(gpu, response);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
    WriteWave(output, 44100, 1, BufferSpan<float>(response.Output));
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc >= 7 && argc <= 10 && std::string_view(argv[1]) == "fit") {
            const std::string_view scale = argc >= 9 ? argv[8] : "physical";
            if (scale != "physical" && scale != "scaled" && scale != "log-decay") throw std::invalid_argument("Parameter scale must be physical, scaled, or log-decay");
            const std::string_view loss = argc == 10 ? argv[9] : "db";
            if (loss != "db" && loss != "ln" && loss != "linear") throw std::invalid_argument("Loss scale must be db, ln, or linear");
            Fit(argv[2], argv[3], argv[4], uint32_t(std::stoul(argv[5])), std::stoull(argv[6]), argc >= 8 ? std::stod(argv[7]) : 2e-6, scale, loss);
        } else if (argc == 6 && std::string_view(argv[1]) == "sample") {
            Sample(argv[2], argv[3], uint32_t(std::stoul(argv[4])), std::stoull(argv[5]));
        } else throw std::invalid_argument("Usage: agarwalResponseFit fit input.wav initial.f32 output steps seed [learning_rate [physical|scaled|log-decay [db|ln|linear]]] | sample parameters.f32 output.wav frames seed");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
