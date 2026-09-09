#include "core/GpuDecimate.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>

using namespace surface_audio;
namespace {
void Check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
void Near(double actual, double expected, double tolerance, const char *message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) throw std::runtime_error(message);
}
std::vector<float> Run(Gpu &gpu, std::span<const float> input, uint32_t frames, uint32_t channels, uint32_t factor) {
    const auto plan = CreateDecimatePlan(gpu, frames, channels, factor);
    const auto source = Upload<float>(gpu, input), output = CreateBuffer(gpu, size_t(plan.OutputFrames) * channels * sizeof(float));
    BeginGpu(gpu);
    DispatchDecimateGpu(gpu, plan, source, output);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto values = BufferSpan<float>(output);
    return {values.begin(), values.end()};
}
// Independent I0 integral quadrature, rather than the production power series.
double IntegralI0(double x) {
    constexpr uint32_t intervals{1024};
    double sum{0};
    for (uint32_t i = 0; i <= intervals; ++i) {
        const double weight{i == 0 || i == intervals ? 1. : i % 2 ? 4. :
                                                                    2.};
        sum += weight * std::exp(x * std::cos(std::numbers::pi * i / intervals));
    }
    return sum / (3 * intervals);
}
std::vector<double> IndependentCoefficients(uint32_t factor) {
    const int radius{int(64 * factor)};
    std::vector<double> result(2 * radius + 1);
    double sum{0};
    for (int offset = -radius; offset <= radius; ++offset) {
        const double cutoff{.475 / factor}, position{double(offset) / radius};
        const double sinc{offset ? std::sin(2 * std::numbers::pi * cutoff * offset) / (std::numbers::pi * offset) : 2 * cutoff};
        const double value{sinc * IntegralI0(10 * std::sqrt(std::max(0., 1 - position * position)))};
        result[offset + radius] = value;
        sum += value;
    }
    for (double &value : result) value /= sum;
    return result;
}
void TestIdentity(Gpu &gpu) {
    const std::vector<float> input{0.f, -0.f, 1.f, -.7f, 1e-20f, -1e20f};
    const auto output = Run(gpu, input, 3, 2, 1);
    Check(output.size() == input.size(), "Identity dimensions");
    for (size_t i = 0; i < input.size(); ++i) Check(std::bit_cast<uint32_t>(output[i]) == std::bit_cast<uint32_t>(input[i]), "Factor one bit identity");
}
void TestResponse(Gpu &gpu, uint32_t factor) {
    constexpr uint32_t output_frames{513};
    const uint32_t input_frames{(output_frames - 1) * factor + 1};
    const std::vector<double> frequencies{0, .1, .4, .45, .5, .6, 1.015625};
    const uint32_t channels{uint32_t(frequencies.size())};
    std::vector<float> input(size_t(input_frames) * channels);
    for (uint32_t channel = 0; channel < channels; ++channel)
        for (uint32_t sample = 0; sample < input_frames; ++sample) input[size_t(channel) * input_frames + sample] = float(std::cos(2 * std::numbers::pi * frequencies[channel] * sample / factor));
    const auto output = Run(gpu, input, input_frames, channels, factor);
    for (uint32_t n = 0; n < output_frames; ++n) Near(output[n], 1, 2e-6, "Unit DC gain including constant-extension edges");
    for (uint32_t channel = 1; channel < channels; ++channel) {
        double worst{0};
        for (uint32_t n = 65; n + 65 < output_frames; ++n) {
            const double value{output[size_t(channel) * output_frames + n]};
            const double expected{channel < 4 ? std::cos(2 * std::numbers::pi * frequencies[channel] * n) : 0};
            worst = std::max(worst, std::abs(value - expected));
        }
        Check(worst < (channel < 4 ? 3e-5 : 1e-5), "Gain-matched passband or output-Nyquist stopband suppression");
        std::cout << "Decimate factor=" << factor << " frequency/output_rate=" << frequencies[channel] << " max_interior_error=" << worst << '\n';
    }
}
void TestImpulseAndEdges(Gpu &gpu) {
    constexpr uint32_t factor{4}, input_frames{640}, impulse{321}, radius{64 * factor};
    const auto coefficients = IndependentCoefficients(factor), production = KaiserDecimateCoefficients(factor);
    for (size_t tap = 0; tap < coefficients.size(); ++tap) Near(production[tap], coefficients[tap], 3e-15, "Kaiser coefficients against independent Bessel integral");
    std::vector<float> input(input_frames);
    input[impulse] = 1;
    const auto output = Run(gpu, input, input_frames, 1, factor);
    Check(output.size() == (input_frames - 1) / factor + 1, "Truncated final input interval does not add an output frame");
    for (uint32_t n = 0; n < output.size(); ++n) {
        const int offset{int(impulse) - int(n * factor)};
        const double expected{std::abs(offset) <= int(radius) ? coefficients[offset + radius] : 0};
        Near(output[n], expected, 3e-8, "Off-grid impulse has exact centered filter phase");
    }
    constexpr uint32_t edge_frames{13};
    std::vector<float> edges(2 * edge_frames);
    edges[0] = 1;
    edges[2 * edge_frames - 1] = -2;
    const auto filtered = Run(gpu, edges, edge_frames, 2, factor);
    const uint32_t outputs{(edge_frames - 1) / factor + 1};
    for (uint32_t channel = 0; channel < 2; ++channel)
        for (uint32_t n = 0; n < outputs; ++n) {
            double expected{0};
            for (int offset = -int(radius); offset <= int(radius); ++offset) {
                const int sample{std::clamp(int(n * factor) + offset, 0, int(edge_frames - 1))};
                expected += coefficients[offset + radius] * edges[channel * edge_frames + sample];
            }
            Near(filtered[channel * outputs + n], expected, 3e-7, "Constant edge extension and channel isolation");
        }
    Near(filtered[0], (1 + coefficients[radius]) / 2, 3e-7, "Leading impulse extends constantly rather than zero padding");
    const std::vector<float> singleton{.75f, -2.f};
    const auto single_output = Run(gpu, singleton, 1, 2, 128);
    Near(single_output[0], .75, 2e-6, "One-frame constant extension");
    Near(single_output[1], -2, 2e-6, "One-frame second channel");
}
void TestInvalid(Gpu &gpu) {
    for (const auto dimensions : {std::array<uint32_t, 3>{0, 1, 4}, {4, 0, 4}, {4, 1, 0}, {4, 1, 129}, {UINT32_MAX, 2, 4}}) {
        bool rejected{false};
        try {
            (void)CreateDecimatePlan(gpu, dimensions[0], dimensions[1], dimensions[2]);
        } catch (const std::invalid_argument &) { rejected = true; }
        Check(rejected, "Invalid decimation dimensions rejected");
    }
    const auto plan = CreateDecimatePlan(gpu, 4, 1, 2);
    const auto input = CreateBuffer(gpu, 4 * sizeof(float)), output = CreateBuffer(gpu, sizeof(float));
    bool rejected{false};
    try {
        DispatchDecimateGpu(gpu, plan, input, output);
    } catch (const std::invalid_argument &) { rejected = true; }
    Check(rejected, "Mismatched output extent rejected before dispatch");
}
void TestBufferViews(Gpu &gpu) {
    const auto plan = CreateDecimatePlan(gpu, 4, 1, 2);
    const auto storage = CreateBuffer(gpu, 32);
    const auto view = [&](size_t offset, size_t bytes) {
        return GpuBuffer{static_cast<std::byte *>(storage.Data) + offset, bytes, storage.Address + offset};
    };
    const auto reject = [&](GpuBuffer input, GpuBuffer output) {
        bool rejected{false};
        try {
            DispatchDecimateGpu(gpu, plan, input, output);
        } catch (const std::invalid_argument &) { rejected = true; }
        Check(rejected, "Overlapping or null decimation views rejected before dispatch");
    };
    reject(view(0, 16), view(12, 8));
    reject(view(4, 16), view(0, 8));
    reject({nullptr, 16, storage.Address}, view(16, 8));
    reject({storage.Data, 16, 0}, view(16, 8));
    reject(view(0, 16), {nullptr, 8, storage.Address + 16});
    reject(view(0, 16), {static_cast<std::byte *>(storage.Data) + 16, 8, 0});
    reject({storage.Data, 16, UINT64_MAX - 7}, {static_cast<std::byte *>(storage.Data) + 12, 8, UINT64_MAX - 3});
    reject({static_cast<std::byte *>(storage.Data) + 4, 16, UINT64_MAX - 3}, {storage.Data, 8, UINT64_MAX - 7});
    for (bool reverse : {false, true}) {
        const auto input = view(reverse ? 8 : 0, 16), output = view(reverse ? 0 : 16, 8);
        std::ranges::fill(BufferSpan<float>(input), .375f);
        BeginGpu(gpu);
        DispatchDecimateGpu(gpu, plan, input, output);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        for (float value : BufferSpan<float>(output)) Near(value, .375, 3e-7, "Adjacent nonoverlapping GPU views are valid in both orders");
    }
}
} // namespace
int main() {
    try {
        auto gpu = CreateGpu();
        TestIdentity(gpu);
        for (uint32_t factor : {2u, 8u, 128u}) TestResponse(gpu, factor);
        TestImpulseAndEdges(gpu);
        TestInvalid(gpu);
        TestBufferViews(gpu);
        std::cout << "Decimator DC, passband, stopband, phase, constant edges and dimensions passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
