#include "core/Adam.h"
#include "core/AudioFile.h"
#include "core/BinaryFile.h"
#include "core/GpuEndpointMix.h"
#include "core/GpuPooledSpectral.h"
#include "core/GpuSpectral.h"
#include <algorithm>
#include <array>
#include <charconv>
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
namespace {
using Clock = std::chrono::steady_clock;

std::vector<float> ReadEndpoints(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open initial endpoint file");
    std::vector<std::array<float, 2>> rows;
    for (std::string line; std::getline(input, line);) {
        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') continue;
        std::istringstream row(line);
        std::array<float, 2> endpoints;
        if (!(row >> endpoints[0] >> endpoints[1])) throw std::invalid_argument("Expected two natural-log amplitudes in each initial endpoint row");
        row >> std::ws;
        if (!row.eof()) throw std::invalid_argument("Too many initial endpoint columns");
        rows.push_back(endpoints);
    }
    if (rows.empty() || rows.size() > 100) throw std::invalid_argument("Endpoint fitting requires one to one hundred modes");
    return std::views::iota(size_t(0), 2 * rows.size()) | std::views::transform([&](size_t index) { return rows[index % rows.size()][index / rows.size()]; }) | std::ranges::to<std::vector>();
}

void WriteEndpoints(const std::filesystem::path &path, std::span<const float> parameters) {
    std::ofstream output(path);
    output << std::setprecision(17);
    for (size_t mode = 0; mode < parameters.size() / 2; ++mode) output << std::exp(double(parameters[mode])) << ' ' << std::exp(double(parameters[parameters.size() / 2 + mode])) << '\n';
    if (!output) throw std::runtime_error("Cannot write fitted endpoint amplitudes");
}

uint32_t Integer(std::string_view text) {
    uint32_t value;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) throw std::invalid_argument("Invalid nonnegative integer argument");
    return value;
}

double LearningRate(const char *text) {
    size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != std::char_traits<char>::length(text) || !std::isfinite(value) || value <= 0) throw std::invalid_argument("Invalid endpoint learning rate");
    return value;
}

void Fit(const std::filesystem::path &basis_path, const std::filesystem::path &location_path, const std::filesystem::path &initial_path, const std::filesystem::path &target_path, const std::filesystem::path &directory, uint32_t steps, double learning_rate, std::string_view scale, double pooled_weight) {
    const auto basis = ReadBinary<float>(basis_path), location = ReadBinary<float>(location_path), initial = ReadEndpoints(initial_path);
    const auto target = ReadWave(target_path);
    if (location.empty() || location.size() > (1u << 22) || basis.size() != location.size() * (initial.size() / 2) || target.Channels != 1 || !target.SampleRate || target.Samples.empty() || target.Samples.size() > location.size()) throw std::invalid_argument("Invalid endpoint basis, location or target dimensions");
    const auto target_samples = [&] {
        std::vector<float> padded(location.size(), 0);
        std::ranges::copy(target.Samples, padded.begin());
        return padded;
    }();
    const SpectralLossOptions options{.Scale = scale == "linear" ? SpectralMagnitudeScale::Linear : scale == "ln" ? SpectralMagnitudeScale::NaturalLog :
                                                                                                                    SpectralMagnitudeScale::Decibels};
    auto gpu = CreateGpu();
    const auto endpoint = CreateGpuEndpointMix(gpu, basis, location, initial);
    const auto spectral = CreateSpectralLossGpu(gpu, target_samples, target.SampleRate, options);
    constexpr std::array pool_bands{SpectralPoolBand{80, 3000}, SpectralPoolBand{3000, 6000}, SpectralPoolBand{6000, 9000}, SpectralPoolBand{9000, 12000}, SpectralPoolBand{12000, 14000}};
    if (pooled_weight > 0 && target.SampleRate % 10) throw std::invalid_argument("Pooled 100ms preset requires a sample rate divisible by ten");
    const auto pool = CreatePooledSpectral(gpu, spectral, pool_bands, {.Weight = float(pooled_weight), .Resolution = 0, .EventSamples = uint32_t(target.Samples.size()), .TimePoolSamples = target.SampleRate / 10});
    const auto parameters = BufferSpan<float>(endpoint.Parameters);
    const std::vector bounds(parameters.size(), std::array{double(EndpointMixMinimumLogAmplitude), double(EndpointMixMaximumLogAmplitude)});
    auto optimizer = CreateAdam(parameters, bounds);
    std::vector<float> best(parameters.begin(), parameters.end());
    double best_loss = std::numeric_limits<double>::infinity(), initial_loss = 0, final_loss = 0;
    double initial_base = 0, initial_pooled = 0, final_base = 0, final_pooled = 0, best_base = 0, best_pooled = 0;
    uint32_t best_step = 0;
    uint64_t bound_updates = 0;
    std::filesystem::create_directories(directory);
    WriteWave(directory / "target-padded.wav", target.SampleRate, 1, target_samples);
    std::ofstream trace(directory / "loss.csv");
    if (!trace) throw std::runtime_error("Cannot write endpoint loss trace");
    trace << "step,total,fft4096,fft1024,fft256,fft64" << (pooled_weight > 0 ? ",pooled" : "") << ",seconds\n"
          << std::setprecision(12);
    const auto start = Clock::now();
    for (uint32_t step = 0;; ++step) {
        BeginGpu(gpu);
        EncodeEndpointMix(gpu, endpoint);
        EncodeSpectralLoss(gpu, spectral, endpoint.Output);
        EncodePooledSpectral(gpu, spectral, pool);
        if (step < steps) EncodeEndpointMixGradient(gpu, endpoint, spectral.Gradient);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        const auto loss = BufferSpan<float>(spectral.Loss);
        final_loss = loss[0];
        final_base = double(loss[1]) + loss[2] + loss[3] + loss[4];
        final_pooled = pooled_weight > 0 ? BufferSpan<float>(pool.Loss)[0] : 0;
        if (!std::isfinite(final_loss)) throw std::runtime_error("Nonfinite endpoint fitting loss");
        if (final_loss < best_loss) {
            best_loss = final_loss;
            best_base = final_base;
            best_pooled = final_pooled;
            best_step = step;
            std::ranges::copy(parameters, best.begin());
        }
        if (!step) {
            initial_loss = final_loss;
            initial_base = final_base;
            initial_pooled = final_pooled;
            WriteWave(directory / "initial.wav", target.SampleRate, 1, BufferSpan<float>(endpoint.Output));
        }
        if (step % 100 == 0 || step == steps) {
            trace << step;
            for (float value : loss) trace << ',' << value;
            if (pooled_weight > 0) trace << ',' << final_pooled;
            trace << ',' << std::chrono::duration<double>(Clock::now() - start).count() << '\n';
            trace.flush();
        }
        if (step % 1000 == 0 || step == steps) std::cout << "endpoint step " << step << " loss " << final_loss << " best " << best_loss << std::endl;
        if (step == steps) break;
        bound_updates += UpdateAdam(optimizer, BufferSpan<float>(endpoint.Gradient), parameters, learning_rate);
    }
    WriteBinary<float>(directory / "last-parameters.f32", parameters);
    WriteEndpoints(directory / "fitted-endpoints.txt", best);
    WriteBinary<float>(directory / "fitted-parameters.f32", best);
    std::ranges::copy(best, parameters.begin());
    BeginGpu(gpu);
    EncodeEndpointMix(gpu, endpoint);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    WriteWave(directory / "fitted.wav", target.SampleRate, 1, BufferSpan<float>(endpoint.Output));
    std::ofstream metadata(directory / "fit.json");
    metadata << std::setprecision(12) << "{\n  \"basis_file\": " << std::quoted(std::filesystem::absolute(basis_path).string())
             << ",\n  \"location_file\": " << std::quoted(std::filesystem::absolute(location_path).string()) << ",\n  \"initial_file\": " << std::quoted(std::filesystem::absolute(initial_path).string())
             << ",\n  \"target_file\": " << std::quoted(std::filesystem::absolute(target_path).string()) << ",\n  \"modes\": " << endpoint.Modes << ",\n  \"sample_rate\": " << target.SampleRate
             << ",\n  \"output_frames\": " << endpoint.Frames << ",\n  \"target_frames\": " << target.Samples.size() << ",\n  \"target_padding_frames\": " << target_samples.size() - target.Samples.size()
             << ",\n  \"steps\": " << steps << ",\n  \"learning_rate\": " << learning_rate << ",\n  \"initial_loss\": " << initial_loss << ",\n  \"final_loss\": " << final_loss
             << ",\n  \"best_loss\": " << best_loss << ",\n  \"best_step\": " << best_step << ",\n  \"bound_updates\": " << bound_updates
             << ",\n  \"seconds\": " << std::chrono::duration<double>(Clock::now() - start).count() << ",\n  \"device\": " << std::quoted(DeviceName(gpu))
             << ",\n  \"optimizer\": \"Adam beta1=.9 beta2=.999 epsilon=1e-8; double master natural-log endpoint amplitudes\""
             << ",\n  \"log_amplitude_bounds\": [" << EndpointMixMinimumLogAmplitude << ',' << EndpointMixMaximumLogAmplitude << ']'
             << ",\n  \"parameter_layout\": \"All endpoint-0 natural logs, then all endpoint-1 natural logs; fitted-endpoints.txt contains physical positive amplitudes in two columns\""
             << ",\n  \"formula\": \"sum_m basis[m,n]*exp((1-location[n])*u[m]+location[n]*v[m])\""
             << ",\n  \"loss_convention\": \"Sum of four mean Huber losses; delta1; periodic Hann; quarter-window hop; centered zero padding; unnormalized one-sided " << scale << " magnitude; smoothed magnitude sqrt(power+1e-12)\""
             << ",\n  \"target_tail_convention\": \"Reference samples followed by explicit zeros to location length; all supplied basis samples and rendered tail participate in fitting\""
             << ",\n  \"scope\": \"Fixed supplied basis and location; all endpoint amplitudes variable; target supplies spectral loss only; inferred endpoint weights, not recovered author parameters\"";
    if (pooled_weight > 0) {
        metadata << ",\n  \"pooled_spectral\": {\n    \"weight\": " << pooled_weight
                 << ", \"effective_float_weight\": " << pool.Options.Weight << ", \"fft_size\": " << spectral.Resolutions[0].Size << ", \"hop\": " << spectral.Resolutions[0].Hop
                 << ", \"event_samples\": " << pool.Options.EventSamples << ", \"time_pool_samples\": " << pool.Options.TimePoolSamples
                 << ", \"magnitude_floor\": " << spectral.Options.MagnitudeFloor << ", \"relative_floor\": " << pool.Options.RelativeFloor
                 << ", \"cell_count\": " << pool.Pools
                 << ",\n    \"formula\": \"weight * mean_cells(0.5 * ((sqrt(mean_abs_X_squared+epsilon_squared)-sqrt(mean_abs_Y_squared+epsilon_squared))/D_band)^2); D_band=max(whole_event_target_band_spectral_RMS,relative_floor*max_band_RMS,epsilon)\""
                 << ",\n    \"convention\": \"Unnormalized one-sided spectral-magnitude RMS with equal bin weights, not calibrated waveform RMS. Bands are lower-inclusive and upper-exclusive: [80,3000),[3000,6000),[6000,9000),[9000,12000),[12000,14000) Hz. Integer 100ms spans group frame centers before the unpadded target end. Empty spans omitted; final nonempty partial span retained. Centered Hann windows can overlap the rendered tail. Target-only normalization weights all event frame/bin samples equally. Original linear loss still covers every padded-tail sample.\""
                 << ",\n    \"scope\": \"Optional inference objective; not an author or paper loss claim\""
                 << ",\n    \"initial_base_loss\": " << initial_base << ", \"initial_weighted_pool_loss\": " << initial_pooled
                 << ", \"final_base_loss\": " << final_base << ", \"final_weighted_pool_loss\": " << final_pooled
                 << ", \"best_base_loss\": " << best_base << ", \"best_weighted_pool_loss\": " << best_pooled << ",\n    \"cells\": [";
        const auto regions = BufferSpan<SpectralPoolRegion>(pool.Regions);
        const auto scales = BufferSpan<std::array<float, 2>>(pool.TargetScale);
        for (uint32_t index = 0; index < pool.Pools; ++index) {
            const auto &region = regions[index];
            if (index) metadata << ',';
            metadata << "{\"band\":" << region.Band << ",\"first_bin\":" << region.FirstBin << ",\"end_bin\":" << region.EndBin
                     << ",\"first_frame\":" << region.FirstFrame << ",\"end_frame\":" << region.EndFrame
                     << ",\"target_mean_power\":" << scales[index][0] << ",\"denominator_squared\":" << scales[index][1] << '}';
        }
        metadata << "]\n  }";
    }
    metadata << "\n}\n";
    if (!metadata || !trace) throw std::runtime_error("Cannot write endpoint fitting metadata");
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc < 7 || argc > 10) throw std::invalid_argument("Usage: agarwalEndpointFit BASIS.f32 LOCATION.f32 INITIAL.txt TARGET.wav OUTPUT STEPS [LEARNING_RATE [linear|ln|db [POOLED_WEIGHT]]]");
        const std::string_view scale = argc >= 9 ? argv[8] : "linear";
        if (scale != "linear" && scale != "ln" && scale != "db") throw std::invalid_argument("Loss scale must be linear, ln, or db");
        const double pooled_weight = [&] {
            if (argc < 10) return 0.;
            size_t consumed = 0;
            const double value = std::stod(argv[9], &consumed);
            if (consumed != std::char_traits<char>::length(argv[9]) || !std::isfinite(value) || value < 0 || value > std::numeric_limits<float>::max() || (value > 0 && float(value) < std::numeric_limits<float>::min())) throw std::invalid_argument("Invalid pooled spectral weight");
            return value;
        }();
        if (pooled_weight > 0 && scale != "linear") throw std::invalid_argument("Pooled spectral term requires linear base loss");
        Fit(argv[1], argv[2], argv[3], argv[4], argv[5], Integer(argv[6]), argc >= 8 ? LearningRate(argv[7]) : .01, scale, pooled_weight);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
