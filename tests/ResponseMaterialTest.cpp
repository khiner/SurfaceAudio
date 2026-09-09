#include "agarwal/ResponseMaterial.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace surface_audio;
using namespace surface_audio::agarwal;

namespace {
void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
template<typename Function> void Invalid(Function function) {
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument &) { rejected = true; }
    Require(rejected, "Invalid response material input must be rejected");
}
constexpr std::array<double, 3> ModeMean{2048, -20, .25}, ModeSlope{128, 2, .015625};
std::array<double, 20> NoiseMean() {
    std::array<double, 20> result{};
    for (uint32_t band = 0; band < 10; ++band) {
        result[band] = -30. + band;
        result[10 + band] = .5 + double(band) / 32;
    }
    return result;
}
std::array<double, 20> NoiseSlope() {
    std::array<double, 20> result{};
    for (uint32_t band = 0; band < 10; ++band) {
        result[band] = double(band + 1) / 4;
        result[10 + band] = double(band + 1) / 1024 * (band % 2 ? -1 : 1);
    }
    return result;
}
std::array<ResponseParameters, 5> Records() {
    std::array<ResponseParameters, 5> result{};
    const auto noise_mean = NoiseMean(), noise_slope = NoiseSlope();
    for (size_t record = 0; record < result.size(); ++record) {
        const double z = double(record) - 2;
        for (uint32_t mode = 0; mode < 10; ++mode) {
            for (size_t dimension = 0; dimension < 3; ++dimension) result[record][dimension * 10 + mode] = float(ModeMean[dimension] + z * ModeSlope[dimension]);
            result[record][30 + mode] = float(noise_mean[mode] + z * noise_slope[mode]);
            result[record][40 + mode] = float(noise_mean[10 + mode] + z * noise_slope[10 + mode]);
        }
    }
    return result;
}

template<size_t Dimensions> void CheckFactor(const std::array<double, Dimensions> &mean, const std::vector<std::array<double, Dimensions>> &factor, const std::array<double, Dimensions> &expected_mean, const std::array<double, Dimensions> &slope) {
    for (size_t row = 0; row < Dimensions; ++row) {
        Require(std::abs(mean[row] - expected_mean[row]) < 1e-10, "Exact Gaussian mean from analytic rank-one fixture");
        double centered_sum = 0;
        for (const auto &observation : factor) centered_sum += observation[row];
        Require(std::abs(centered_sum) < 1e-10, "Centered-data factor preserves deficient rank");
        for (size_t column = 0; column < Dimensions; ++column) {
            double covariance = 0;
            for (const auto &observation : factor) covariance += observation[row] * observation[column];
            const double expected = 2 * slope[row] * slope[column]; // E[z^2] for {-2,-1,0,1,2}, denominator n.
            Require(std::abs(covariance - expected) < 1e-12 * std::max(1., std::abs(expected)), "Full analytic rank-one MLE covariance, including off-diagonal entries");
        }
    }
}

template<size_t Dimensions> struct Moments {
    uint32_t Count{};
    std::array<double, Dimensions> Sum{};
    std::array<std::array<double, Dimensions>, Dimensions> Product{};
};
template<size_t Dimensions> void Accumulate(Moments<Dimensions> &moments, const std::array<double, Dimensions> &standardized) {
    ++moments.Count;
    for (size_t row = 0; row < Dimensions; ++row) {
        moments.Sum[row] += standardized[row];
        for (size_t column = 0; column < Dimensions; ++column) moments.Product[row][column] += standardized[row] * standardized[column];
    }
}
template<size_t Dimensions> void CheckMoments(const Moments<Dimensions> &moments) {
    for (size_t row = 0; row < Dimensions; ++row) {
        const double mean = moments.Sum[row] / moments.Count;
        Require(std::abs(mean) < .08, "Sample means agree with analytic Gaussian");
        for (size_t column = 0; column < Dimensions; ++column) {
            const double covariance = moments.Product[row][column] / moments.Count - mean * moments.Sum[column] / moments.Count;
            Require(std::abs(covariance / 2 - 1) < .08, "Empirical full covariance retains joint amplitude/decay correlations");
        }
    }
}

void TestFitAndSampling() {
    const auto material = FitResponseMaterial(Records());
    Require(material.Records == 5 && material.ModeFactor.size() == 50 && material.NoiseFactor.size() == 5, "Modes pooled while noise remains whole-record vectors");
    CheckFactor(material.ModeMean, material.ModeFactor, ModeMean, ModeSlope);
    CheckFactor(material.NoiseMean, material.NoiseFactor, NoiseMean(), NoiseSlope());
    auto random = MakeRandom(783, 9), repeated = random;
    ResponseMaterialSampleStats stats{}, repeated_stats{};
    const auto first = SampleResponseMaterial(material, random, &stats), second = SampleResponseMaterial(material, repeated, &repeated_stats);
    Require(first == second && random.State == repeated.State && random.Sequence == repeated.Sequence, "Identical PCG state gives identical parameters and updated state");
    Require(stats.ModeRejections == repeated_stats.ModeRejections && stats.NoiseRejections == repeated_stats.NoiseRejections, "Rejection counts are deterministic");
    Require(SampleResponseMaterial(material, random) != first, "Successive runtime samples advance the caller's random stream");
    Moments<3> mode_moments;
    Moments<20> noise_moments;
    const auto noise_mean = NoiseMean(), noise_slope = NoiseSlope();
    for (uint32_t draw = 0; draw < 4000; ++draw) {
        const auto sample = SampleResponseMaterial(material, random, &stats);
        for (uint32_t mode = 0; mode < 10; ++mode) {
            std::array<double, 3> value{};
            for (size_t dimension = 0; dimension < 3; ++dimension) value[dimension] = (sample[dimension * 10 + mode] - ModeMean[dimension]) / ModeSlope[dimension];
            Require(std::abs(value[0] - value[1]) < 5e-6 && std::abs(value[0] - value[2]) < 5e-6, "Each sampled mode stays on the rank-one joint Gaussian line");
            Accumulate(mode_moments, value);
        }
        std::array<double, 20> value{};
        for (size_t dimension = 0; dimension < 20; ++dimension) {
            value[dimension] = (sample[30 + dimension] - noise_mean[dimension]) / noise_slope[dimension];
            Require(std::abs(value[dimension] - value[0]) < 5e-5, "Twenty-dimensional noise retains joint rank-one support after float conversion");
        }
        Accumulate(noise_moments, value);
    }
    Require(!stats.ModeRejections && !stats.NoiseRejections, "Analytic fixture stays sufficiently far from truncation boundaries");
    CheckMoments(mode_moments);
    CheckMoments(noise_moments);
}

void TestRejections() {
    auto material = FitResponseMaterial(Records());
    material.ModeMean[0] = 21;
    for (auto &factor : material.ModeFactor) factor[0] *= 5. / 128;
    material.NoiseMean[10] = .0005;
    auto random = MakeRandom(314), repeated = random;
    ResponseMaterialSampleStats stats{}, repeated_stats{};
    for (uint32_t draw = 0; draw < 100; ++draw) {
        const auto sample = SampleResponseMaterial(material, random, &stats);
        Require(sample == SampleResponseMaterial(material, repeated, &repeated_stats), "Whole-vector rejection advances PCG deterministically");
        for (uint32_t mode = 0; mode < 10; ++mode) {
            Require(sample[mode] >= 20 && sample[mode] <= 20000 && sample[20 + mode] > 0 && sample[40 + mode] > 0, "Returned parameters satisfy physical rejection domain");
            Require(std::abs((sample[mode] - 21.) / 5 - (sample[10 + mode] + 20.) / 2) < 5e-6, "Frequency rejection retains the complete correlated mode triple");
        }
        const double noise_z = (sample[40] - .0005) * 1024;
        Require(std::abs(noise_z - (sample[30] + 30.) * 4) < 1e-5, "Noise RT rejection retains all correlated amplitudes");
    }
    Require(stats.ModeRejections > 100 && stats.NoiseRejections > 10, "Both rejection paths exercised");
    Require(stats.ModeRejections == repeated_stats.ModeRejections && stats.NoiseRejections == repeated_stats.NoiseRejections, "Cumulative rejection statistics match repeated stream");
}

void TestInvalid() {
    const auto records = Records();
    const std::array identical{records[0], records[0]};
    const auto constant = FitResponseMaterial(identical);
    auto constant_random = MakeRandom(881);
    Require(SampleResponseMaterial(constant, constant_random) == records[0], "Two identical records retain zero covariance without artificial jitter");
    Invalid([] { FitResponseMaterial({}); });
    Invalid([&] { FitResponseMaterial(std::span(records).first(1)); });
    for (uint32_t parameter : {0u, 20u, 40u}) {
        auto invalid = records;
        invalid[0][parameter] = 0;
        Invalid([&] { FitResponseMaterial(invalid); });
    }
    auto nonfinite = records;
    nonfinite[2][13] = std::numeric_limits<float>::quiet_NaN();
    Invalid([&] { FitResponseMaterial(nonfinite); });
    auto random = MakeRandom(12);
    Invalid([&] { SampleResponseMaterial({}, random); });
    auto malformed = FitResponseMaterial(records);
    malformed.NoiseFactor.pop_back();
    Invalid([&] { SampleResponseMaterial(malformed, random); });
    auto impossible = FitResponseMaterial(std::span(records).first(2));
    impossible.ModeMean[0] = 1;
    for (auto &factor : impossible.ModeFactor) factor[0] = 0;
    ResponseMaterialSampleStats stats{};
    bool exhausted = false;
    try {
        SampleResponseMaterial(impossible, random, &stats);
    } catch (const std::runtime_error &) { exhausted = true; }
    Require(exhausted && stats.ModeRejections == 1000000, "Impossible distribution stops at the explicit rejection limit");
}
} // namespace

int main() {
    try {
        TestFitAndSampling();
        TestRejections();
        TestInvalid();
        std::cout << "Response material analytic covariance, 4000 draws, PCG determinism, and rejection gates passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
