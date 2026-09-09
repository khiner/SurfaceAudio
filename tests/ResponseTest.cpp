#include "agarwal/Response.h"
#include "core/GpuSpectral.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <numeric>
#include <stdexcept>

using namespace surface_audio;
using namespace surface_audio::agarwal;

namespace {
void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
std::array<double, ResponseParameterCount> Parameters() {
    std::array<double, ResponseParameterCount> result{};
    for (uint32_t mode = 0; mode < ResponseModeCount; ++mode) {
        result[mode] = float(137 + 871.13 * mode);
        result[10 + mode] = float(-9 - 2.3 * mode);
        result[20 + mode] = float(.031 + .042 * mode);
        result[30 + mode] = float(-25 - 1.7 * mode);
        result[40 + mode] = float(.019 + .027 * mode);
    }
    return result;
}
std::vector<float> Noise(uint32_t frames) {
    std::vector<float> noise(size_t(frames) * ResponseModeCount);
    for (uint32_t mode = 0; mode < ResponseModeCount; ++mode) {
        for (uint32_t frame = 0; frame < frames; ++frame) noise[size_t(mode) * frames + frame] = float(std::sin(.37 * frame + 1.9 * mode) + .31 * std::cos(.071 * frame * (mode + 1)));
    }
    return noise;
}
void TestEquations() {
    constexpr uint32_t frames = 401;
    std::array<double, ResponseParameterCount> parameters{};
    for (uint32_t mode = 0; mode < ResponseModeCount; ++mode) {
        parameters[10 + mode] = parameters[30 + mode] = -600;
        parameters[20 + mode] = parameters[40 + mode] = .2;
    }
    std::vector<float> noise(frames * ResponseModeCount);
    std::vector<double> output(frames);
    parameters[0] = 5;
    parameters[10] = -6;
    EvaluateResponse(parameters, frames, 1000, noise, output);
    Require(output[0] == 0, "Zero-phase modes vanish at onset");
    Require(std::abs(output[50] - std::pow(10., -.3 - .75)) < 1e-14, "Eq. 2 amplitude in dB and seconds");
    Require(std::abs(output[250] / output[50] - .001) < 1e-15, "RT60 attenuates amplitude by 60 dB");
    parameters[10] = -600;
    parameters[33] = -20;
    std::fill_n(noise.begin() + 3 * frames, frames, 2.f);
    EvaluateResponse(parameters, frames, 1000, noise, output);
    Require(std::abs(output[0] - .2) < 1e-15, "Eq. 3 band-major fixed noise and dB amplitude");
    Require(std::abs(output[200] / output[0] - .001) < 1e-15, "Noise RT60 has the same amplitude decay");
    parameters[43] = 0;
    bool rejected = false;
    try {
        EvaluateResponse(parameters, frames, 1000, noise, output);
    } catch (const std::invalid_argument &) { rejected = true; }
    Require(rejected, "Nonpositive RT60 is rejected by reference");
}
void TestGradient() {
    constexpr uint32_t frames = 777;
    constexpr double sample_rate = 48000;
    auto parameters = Parameters();
    const auto noise = Noise(frames);
    std::vector<double> adjoint(frames), plus(frames), minus(frames), output(frames);
    for (uint32_t frame = 0; frame < frames; ++frame) adjoint[frame] = std::cos(.29 * frame) + .7 * std::sin(.031 * frame);
    std::array<double, ResponseParameterCount> gradient{};
    EvaluateResponse(parameters, frames, sample_rate, noise, output);
    for (uint32_t frame = 0; frame < frames; ++frame) {
        const double time = frame / sample_rate;
        double expected = 0;
        for (uint32_t mode = 0; mode < ResponseModeCount; ++mode) {
            const double amplitude = std::exp(parameters[10 + mode] * std::log(10.) / 20);
            const double noise_amplitude = std::exp(parameters[30 + mode] * std::log(10.) / 20);
            expected += amplitude * std::exp(-std::log(1000.) * time / parameters[20 + mode]) * std::sin(2 * std::numbers::pi * parameters[mode] * time);
            expected += noise_amplitude * std::exp(-std::log(1000.) * time / parameters[40 + mode]) * noise[size_t(mode) * frames + frame];
        }
        Require(std::abs(output[frame] - expected) < 1e-14, "Independent exponential form matches all twenty components");
    }
    EvaluateResponseGradient(parameters, frames, sample_rate, noise, adjoint, gradient);
    for (uint32_t parameter = 0; parameter < ResponseParameterCount; ++parameter) {
        const double value = parameters[parameter];
        const double step = parameter < 10 ? .001 : (parameter >= 20 && parameter < 30) || parameter >= 40 ? 1e-6 * value :
                                                                                                             1e-5;
        parameters[parameter] = value + step;
        EvaluateResponse(parameters, frames, sample_rate, noise, plus);
        parameters[parameter] = value - step;
        EvaluateResponse(parameters, frames, sample_rate, noise, minus);
        parameters[parameter] = value;
        double finite_difference = 0;
        for (uint32_t frame = 0; frame < frames; ++frame) finite_difference += adjoint[frame] * (plus[frame] - minus[frame]) / (2 * step);
        if (std::abs(gradient[parameter] - finite_difference) > 2e-7 * (1 + std::abs(finite_difference))) {
            std::cerr << "parameter " << parameter << " analytic " << gradient[parameter] << " finite difference " << finite_difference << '\n';
            throw std::runtime_error("All fifty waveform adjoints match central finite differences");
        }
    }
}
void TestNoise(Gpu &gpu) {
    constexpr uint32_t frames = 2049;
    const ResponseNoiseSettings settings{.TapCount = 129, .Seed = 993};
    const auto reference = ResponseNoiseReference(frames, 48000, settings);
    const auto actual = CreateResponseNoise(gpu, frames, 48000, settings);
    const auto repeated = CreateResponseNoise(gpu, frames, 48000, settings);
    Require(actual == repeated, "Fixed seed gives bit-exact repeated GPU noise");
    double peak_error = 0;
    for (size_t index = 0; index < actual.size(); ++index) peak_error = std::max(peak_error, std::abs(actual[index] - reference[index]));
    Require(peak_error < 4e-6, "GPU FIR matches double accumulation of identical Gaussian and coefficients");
    for (uint32_t band = 0; band < ResponseModeCount; ++band) {
        double energy = 0;
        for (uint32_t frame = 0; frame < frames; ++frame) energy += reference[size_t(band) * frames + frame] * reference[size_t(band) * frames + frame];
        Require(energy / frames > .3 && energy / frames < 2, "Each normalized ERB filter retains finite noise energy");
    }
    auto changed_settings = settings;
    changed_settings.Seed++;
    Require(CreateResponseNoise(gpu, frames, 48000, changed_settings) != actual, "Noise seed changes the fixed realization");
}
void TestGpu(Gpu &gpu, uint32_t frames) {
    constexpr float sample_rate = 48000;
    auto parameters = Parameters();
    const auto noise = Noise(frames);
    auto state = CreateResponseGpu(gpu, frames, sample_rate, noise);
    auto gpu_parameters = BufferSpan<float>(state.Parameters);
    for (uint32_t parameter = 0; parameter < ResponseParameterCount; ++parameter) gpu_parameters[parameter] = float(parameters[parameter]);
    std::vector<float> adjoint(frames);
    for (uint32_t frame = 0; frame < frames; ++frame) adjoint[frame] = float(std::cos(.29 * frame) + .7 * std::sin(.031 * frame));
    const auto gpu_adjoint = Upload<float>(gpu, adjoint);
    std::vector<double> reference(frames), reference_adjoint(adjoint.begin(), adjoint.end());
    std::array<double, ResponseParameterCount> reference_gradient{};
    EvaluateResponse(parameters, frames, sample_rate, noise, reference);
    EvaluateResponseGradient(parameters, frames, sample_rate, noise, reference_adjoint, reference_gradient);
    BeginGpu(gpu);
    EncodeResponse(gpu, state);
    EncodeResponseGradient(gpu, state, gpu_adjoint);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto output = BufferSpan<float>(state.Output), gradient = BufferSpan<float>(state.Gradient);
    double peak_error = 0;
    for (uint32_t frame = 0; frame < frames; ++frame) peak_error = std::max(peak_error, std::abs(output[frame] - reference[frame]));
    Require(peak_error < 2e-5, "Raw GPU waveform matches double reference including partial groups");
    for (uint32_t parameter = 0; parameter < ResponseParameterCount; ++parameter) {
        if (std::abs(gradient[parameter] - reference_gradient[parameter]) > 1e-4 * (1 + std::abs(reference_gradient[parameter]))) {
            std::cerr << "GPU parameter " << parameter << " gradient " << gradient[parameter] << " reference " << reference_gradient[parameter] << '\n';
            throw std::runtime_error("All fifty GPU adjoints match double reference");
        }
    }
    const std::vector<float> original(output.begin(), output.end());
    BeginGpu(gpu);
    EncodeResponse(gpu, state);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    Require(std::equal(output.begin(), output.end(), original.begin()), "Reusable response produces bit-exact repeated output");
    parameters[10] = gpu_parameters[10] = -30;
    std::ranges::fill(BufferSpan<float>(gpu_adjoint), 0.f);
    BeginGpu(gpu);
    EncodeResponse(gpu, state);
    EncodeResponseGradient(gpu, state, gpu_adjoint);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    EvaluateResponse(parameters, frames, sample_rate, noise, reference);
    for (uint32_t frame = 0; frame < frames; ++frame) Require(std::abs(output[frame] - reference[frame]) < 2e-5, "Parameter upload updates reusable GPU response");
    Require(std::ranges::all_of(gradient, [](float value) { return value == 0; }), "New adjoint overwrites every gradient partial");
}

void TestLongPhase(Gpu &gpu) {
    constexpr uint32_t frames = 95001;
    constexpr float sample_rate = 44100;
    auto parameters = Parameters();
    for (uint32_t mode = 0; mode < ResponseModeCount; ++mode) {
        parameters[mode] = float(11000.17 + 431.13 * mode);
        parameters[20 + mode] = 10 + mode;
        parameters[30 + mode] = -60;
    }
    const auto noise = Noise(frames);
    const auto state = CreateResponseGpu(gpu, frames, sample_rate, noise);
    for (uint32_t parameter = 0; parameter < ResponseParameterCount; ++parameter) BufferSpan<float>(state.Parameters)[parameter] = float(parameters[parameter]);
    std::vector<double> reference(frames);
    EvaluateResponse(parameters, frames, sample_rate, noise, reference);
    BeginGpu(gpu);
    EncodeResponse(gpu, state);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto output = BufferSpan<float>(state.Output);
    double peak_error = 0, error_energy = 0, energy = 0;
    for (uint32_t frame = 0; frame < frames; ++frame) {
        const double error = output[frame] - reference[frame];
        peak_error = std::max(peak_error, std::abs(error));
        error_energy += error * error;
        energy += reference[frame] * reference[frame];
    }
    const double relative_rms = std::sqrt(error_energy / energy);
    std::cout << "95,001-frame sustained 11-15 kHz phase: peak error " << peak_error << ", relative RMS " << relative_rms << '\n';
    Require(peak_error < 1e-6 && relative_rms < 2e-6, "Compensated phase retains long high-frequency IR tails");
}

void TestSpectralChain(Gpu &gpu) {
    constexpr uint32_t frames = 2049;
    constexpr float sample_rate = 44100;
    auto parameters = Parameters(), target_parameters = parameters;
    for (uint32_t mode = 0; mode < ResponseModeCount; ++mode) {
        target_parameters[mode] += 12;
        target_parameters[10 + mode] += 3;
        target_parameters[20 + mode] *= 1.1;
        target_parameters[30 + mode] += 2;
    }
    const auto noise = CreateResponseNoise(gpu, frames, sample_rate);
    std::vector<double> target_double(frames);
    EvaluateResponse(target_parameters, frames, sample_rate, noise, target_double);
    const std::vector<float> target(target_double.begin(), target_double.end());
    const auto response = CreateResponseGpu(gpu, frames, sample_rate, noise);
    const auto spectral = CreateSpectralLossGpu(gpu, target, uint32_t(sample_rate));
    const auto values = BufferSpan<float>(response.Parameters);
    auto evaluate = [&](bool gradient) {
        BeginGpu(gpu);
        EncodeResponse(gpu, response);
        EncodeSpectralLoss(gpu, spectral, response.Output);
        if (gradient) EncodeResponseGradient(gpu, response, spectral.Gradient);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        return double(BufferSpan<float>(spectral.Loss)[0]);
    };
    for (uint32_t parameter = 0; parameter < ResponseParameterCount; ++parameter) values[parameter] = float(parameters[parameter]);
    const double initial_loss = evaluate(true);
    const auto gpu_gradient = BufferSpan<float>(response.Gradient);
    const std::vector<float> gradient(gpu_gradient.begin(), gpu_gradient.end());
    double maximum_error = 0;
    for (uint32_t trial = 0; trial < 6; ++trial) {
        std::array<double, ResponseParameterCount> direction{};
        if (trial == 0) direction[2] = 3;
        if (trial == 1) direction[12] = 1;
        if (trial == 2) direction[22] = .01;
        if (trial == 3) direction[33] = 1;
        if (trial == 4) direction[43] = .01;
        if (trial == 5) {
            for (uint32_t parameter = 0; parameter < ResponseParameterCount; ++parameter) {
                const double scale = parameter < 10 ? 3 : ((parameter >= 20 && parameter < 30) || parameter >= 40) ? .01 :
                                                                                                                     1;
                direction[parameter] = scale * std::sin(1.73 * parameter + .5);
            }
        }
        for (double step : {.02, .01}) {
            std::array<float, ResponseParameterCount> plus{}, minus{};
            for (uint32_t parameter = 0; parameter < ResponseParameterCount; ++parameter) {
                plus[parameter] = float(parameters[parameter] + step * direction[parameter]);
                minus[parameter] = float(parameters[parameter] - step * direction[parameter]);
                values[parameter] = plus[parameter];
            }
            const double plus_loss = evaluate(false);
            std::ranges::copy(minus, values.begin());
            const double minus_loss = evaluate(false);
            double analytic = 0;
            for (uint32_t parameter = 0; parameter < ResponseParameterCount; ++parameter) analytic += gradient[parameter] * (double(plus[parameter]) - minus[parameter]) / (2 * step);
            const double numeric = (plus_loss - minus_loss) / (2 * step);
            const double error = std::abs(analytic - numeric) / std::max(1., std::abs(numeric));
            maximum_error = std::max(maximum_error, error);
            std::cout << "Spectral response direction " << trial << ", step " << step << ": analytic " << analytic << ", finite difference " << numeric << ", error " << error << '\n';
            Require(error < .001, "Composed spectral response adjoint matches quantization-aware finite differences");
        }
    }
    bool descended = false;
    for (double step : {.01, .003, .001}) {
        for (uint32_t parameter = 0; parameter < ResponseParameterCount; ++parameter) {
            const double scale = parameter < 10 ? 3 : ((parameter >= 20 && parameter < 30) || parameter >= 40) ? .01 :
                                                                                                                 1;
            values[parameter] = float(parameters[parameter] - step * scale * scale * gradient[parameter]);
        }
        if (evaluate(false) < initial_loss) descended = true;
    }
    Require(descended, "Composed gradient decreases off-target synthetic spectral loss");
    std::cout << "Composed spectral derivative maximum error " << maximum_error << '\n';
}
} // namespace

int main() {
    try {
        TestEquations();
        TestGradient();
        auto gpu = CreateGpu();
        TestNoise(gpu);
        TestGpu(gpu, 1);
        TestGpu(gpu, 257);
        TestGpu(gpu, 4099);
        TestLongPhase(gpu);
        TestSpectralChain(gpu);
        std::cout << "Response equations, 50 derivatives, fixed FIR noise, and GPU parity passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
