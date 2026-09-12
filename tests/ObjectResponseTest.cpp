#include "agarwal/ObjectResponse.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <numeric>
#include <stdexcept>
using namespace surface_audio;
using namespace surface_audio::agarwal;
namespace {
void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
ObjectResponseParameters Parameters() {
    ObjectResponseParameters p{};
    for (uint32_t i = 0; i < 10; ++i) {
        p[i] = 173 + 331 * i;
        p[10 + i] = -18 - float(i);
        p[20 + i] = .03f + .01f * i;
    }
    for (uint32_t i = 0; i < 20; ++i) {
        p[30 + i] = -30 - float(i);
        p[50 + i] = .01f + .003f * i;
    }
    return p;
}
void ResponseChecks(Gpu &gpu) {
    constexpr uint32_t frames = 817;
    constexpr double rate = 44100;
    const auto p = Parameters();
    const std::vector<double> parameters(p.begin(), p.end());
    std::vector<float> noise(size_t(frames) * 20), adjoint(frames);
    for (size_t i = 0; i < noise.size(); ++i) noise[i] = float(std::sin(.17 * i));
    for (uint32_t i = 0; i < frames; ++i) adjoint[i] = float(std::cos(.071 * i));
    const std::vector<double> adjoint64(adjoint.begin(), adjoint.end());
    std::vector<double> expected(frames), gradient(70);
    EvaluateResponse(parameters, frames, rate, noise, expected);
    EvaluateResponseGradient(parameters, frames, rate, noise, adjoint64, gradient);
    for (uint32_t frame = 0; frame < frames; ++frame) {
        const double t = frame / rate;
        double reference = 0;
        for (uint32_t mode = 0; mode < 10; ++mode) reference += std::pow(10., p[10 + mode] / 20. - 3 * t / p[20 + mode]) * std::sin(2 * std::numbers::pi * p[mode] * t);
        for (uint32_t band = 0; band < 20; ++band) reference += std::pow(10., p[30 + band] / 20. - 3 * t / p[50 + band]) * noise[band * frames + frame];
        Require(std::abs(reference - expected[frame]) < 1e-14, "All twenty noise bands enter the published response equation");
    }
    for (uint32_t i = 0; i < 70; ++i) {
        const double epsilon = i < 10 ? 1e-3 : (i >= 20 && i < 30) || i >= 50 ? 1e-7 :
                                                                                1e-4;
        auto plus = parameters, minus = parameters;
        plus[i] += epsilon;
        minus[i] -= epsilon;
        std::vector<double> a(frames), b(frames);
        EvaluateResponse(plus, frames, rate, noise, a);
        EvaluateResponse(minus, frames, rate, noise, b);
        double derivative = 0;
        for (uint32_t j = 0; j < frames; ++j) derivative += (a[j] - b[j]) * adjoint[j] / (2 * epsilon);
        Require(std::abs(derivative - gradient[i]) < 1e-6 * std::max(1., std::abs(gradient[i])), "Seventy-parameter finite-difference gradient");
    }
    const auto response = CreateResponseGpu(gpu, frames, rate, noise, 20);
    std::ranges::copy(p, BufferSpan<float>(response.Parameters).begin());
    const auto input_adjoint = Upload<float>(gpu, adjoint);
    BeginGpu(gpu);
    EncodeResponse(gpu, response);
    EncodeResponseGradient(gpu, response, input_adjoint);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto actual = BufferSpan<float>(response.Output), actual_gradient = BufferSpan<float>(response.Gradient);
    for (uint32_t i = 0; i < frames; ++i) Require(std::abs(actual[i] - expected[i]) < 1e-6, "Metal response agrees with independent FP64 equation");
    for (uint32_t i = 0; i < 70; ++i)
        Require(std::abs(actual_gradient[i] - gradient[i]) < 1e-4 * (1 + std::abs(gradient[i])), "Metal seventy-parameter adjoint with FP32 cancellation tolerance");
}
void Distributions() {
    std::array<std::array<double, 3>, 5> observations{};
    for (uint32_t i = 0; i < 5; ++i) observations[i] = {100 + 3. * i, -20 + .5 * i, .1};
    const auto gaussian = FitGaussian<3>(observations);
    Require(gaussian.Mean[0] == 106 && gaussian.Factor[2][2] == 0, "Gaussian means and constant dimensions");
    double covariance = 0;
    for (uint32_t i = 0; i < 3; ++i) covariance += gaussian.Factor[0][i] * gaussian.Factor[1][i];
    Require(std::abs(covariance - 3.75) < 1e-12, "Unbiased correlated covariance with deficient rank");
    std::array<ObjectResponseParameters, 5> records;
    for (uint32_t i = 0; i < records.size(); ++i) {
        records[i] = Parameters();
        for (uint32_t band = 0; band < 20; ++band) {
            records[i][30 + band] += float(i) * .5f;
            records[i][50 + band] += float(i) * .001f;
        }
    }
    records[0][0] = 10;
    records[1][20] = .001f;
    records[0][50] = .001f;
    const auto distribution = FitObjectResponseDistribution(records);
    Require(distribution.Modes.Count == 48 && distribution.Noise[0].Count == 4 && distribution.Noise[1].Count == 5, "Paper-specific exclusions before fitting");
    auto random = MakeRandom(2026), repeated = MakeRandom(2026);
    for (uint32_t sample = 0; sample < 1000; ++sample) {
        const auto p = SampleObjectResponse(distribution, random);
        Require(p == SampleObjectResponse(distribution, repeated), "Deterministic categorical sampling");
        for (uint32_t i = 0; i < 10; ++i) Require(p[i] >= 20 && p[i] <= 20000 && p[10 + i] <= 0 && p[20 + i] >= .01f && p[20 + i] <= 1, "Truncated modal support");
        for (uint32_t i = 0; i < 20; ++i) Require(p[30 + i] <= 0 && p[50 + i] >= .005f && p[50 + i] <= 1, "Truncated noise support");
    }
    for (uint32_t i = 1; i < records.size(); ++i) records[i][69] = .001f;
    bool rejected = false;
    try {
        FitObjectResponseDistribution(records);
    } catch (const std::invalid_argument &) { rejected = true; }
    Require(rejected, "A single eligible noise observation cannot determine sample covariance");
}
}
int main() {
    try {
        Distributions();
        auto gpu = CreateGpu();
        ResponseChecks(gpu);
        std::cout << "Object response equations, gradients and distributions passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
