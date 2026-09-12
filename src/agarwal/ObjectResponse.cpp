#include "ObjectResponse.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace surface_audio::agarwal {
namespace {
template<size_t D, typename Valid> std::array<float, D> Truncated(const Gaussian<D> &distribution, RandomState &random, Valid valid, uint64_t *rejections) {
    if (distribution.Count < 2 || !std::ranges::all_of(distribution.Mean, [](double x) { return std::isfinite(x); })) throw std::invalid_argument("Invalid object response distribution");
    for (const auto &row : distribution.Factor)
        if (!std::ranges::all_of(row, [](double x) { return std::isfinite(x); })) throw std::invalid_argument("Nonfinite Gaussian factor");
    for (uint32_t attempt = 0; attempt < 1000000; ++attempt) {
        const auto sample = SampleGaussian(distribution, random);
        std::array<float, D> result{};
        std::ranges::transform(sample, result.begin(), [](double x) { return float(x); });
        if (std::ranges::all_of(result, [](float x) { return std::isfinite(x); }) && valid(result)) return result;
        if (rejections) ++*rejections;
    }
    throw std::runtime_error("Truncated object response sampling exceeded one million attempts");
}
}

ObjectResponseDistribution FitObjectResponseDistribution(std::span<const ObjectResponseParameters> records) {
    if (records.size() < 2 || records.size() > UINT32_MAX / ResponseModeCount) throw std::invalid_argument("Object response fitting requires at least two records");
    std::vector<std::array<double, 3>> modes;
    std::array<std::vector<std::array<double, 2>>, ObjectResponseNoiseCount> noise;
    for (const auto &record : records) {
        if (!std::ranges::all_of(record, [](float x) { return std::isfinite(x); })) throw std::invalid_argument("Nonfinite object response parameters");
        for (uint32_t mode = 0; mode < 10; ++mode)
            if (record[mode] >= 20 && record[20 + mode] >= .01f) modes.push_back({record[mode], record[10 + mode], record[20 + mode]});
        for (uint32_t band = 0; band < 20; ++band)
            if (record[50 + band] >= .005f) noise[band].push_back({record[30 + band], record[50 + band]});
    }
    const auto noise_distributions = [&] {
        std::array<Gaussian<2>, 20> result{};
        for (uint32_t band = 0; band < 20; ++band) result[band] = FitGaussian<2>(noise[band]);
        return result;
    }();
    return {.Modes = FitGaussian<3>(modes), .Noise = noise_distributions};
}

ObjectResponseParameters SampleObjectResponse(const ObjectResponseDistribution &distribution, RandomState &random, ObjectResponseSampleStats *stats) {
    ObjectResponseParameters result{};
    for (uint32_t mode = 0; mode < 10; ++mode) {
        const auto x = Truncated(distribution.Modes, random, [](const auto &x) { return x[0] >= 20 && x[0] <= 20000 && x[1] <= 0 && x[2] >= .01f && x[2] <= 1; }, stats ? &stats->ModeRejections : nullptr);
        result[mode] = x[0];
        result[10 + mode] = x[1];
        result[20 + mode] = x[2];
    }
    for (uint32_t band = 0; band < 20; ++band) {
        const auto x = Truncated(distribution.Noise[band], random, [](const auto &x) { return x[0] <= 0 && x[1] >= .005f && x[1] <= 1; }, stats ? &stats->NoiseRejections : nullptr);
        result[30 + band] = x[0];
        result[50 + band] = x[1];
    }
    return result;
}
}
