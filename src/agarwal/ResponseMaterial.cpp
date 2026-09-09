#include "ResponseMaterial.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace surface_audio::agarwal {
namespace {
template<size_t Dimensions> std::array<double, Dimensions> Center(std::vector<std::array<double, Dimensions>> &factor) {
    std::array<double, Dimensions> mean{};
    for (const auto &observation : factor)
        for (size_t dimension = 0; dimension < Dimensions; ++dimension) mean[dimension] += observation[dimension] / double(factor.size());
    const double scale = 1 / std::sqrt(double(factor.size()));
    for (auto &observation : factor)
        for (size_t dimension = 0; dimension < Dimensions; ++dimension) observation[dimension] = (observation[dimension] - mean[dimension]) * scale;
    return mean;
}
template<size_t Dimensions> bool Finite(const std::array<double, Dimensions> &values) {
    return std::ranges::all_of(values, [](double value) { return std::isfinite(value); });
}
template<size_t Dimensions, typename Valid> std::array<float, Dimensions> Draw(const std::array<double, Dimensions> &mean, const std::vector<std::array<double, Dimensions>> &factor, RandomState &random, Valid valid, uint64_t *rejections) {
    for (uint32_t attempt = 0; attempt < 1000000; ++attempt) {
        auto sample = mean;
        for (const auto &observation : factor) {
            const double normal = Normal(random);
            for (size_t dimension = 0; dimension < Dimensions; ++dimension) sample[dimension] += observation[dimension] * normal;
        }
        std::array<float, Dimensions> result{};
        std::ranges::transform(sample, result.begin(), [](double value) { return float(value); });
        if (std::ranges::all_of(result, [](float value) { return std::isfinite(value); }) && valid(result)) return result;
        if (rejections) ++*rejections;
    }
    throw std::runtime_error("Response material sampling exceeded one million attempts");
}
} // namespace

ResponseMaterial FitResponseMaterial(std::span<const ResponseParameters> records) {
    if (records.size() < 2 || records.size() > std::numeric_limits<uint32_t>::max() / ResponseModeCount) throw std::invalid_argument("Response material requires at least two records");
    std::vector<std::array<double, 3>> mode_factor;
    std::vector<std::array<double, 20>> noise_factor;
    mode_factor.reserve(records.size() * ResponseModeCount);
    noise_factor.reserve(records.size());
    for (const auto &record : records) {
        if (!std::ranges::all_of(record, [](float value) { return std::isfinite(value); })) throw std::invalid_argument("Nonfinite response material record");
        std::array<double, 20> noise{};
        for (uint32_t mode = 0; mode < ResponseModeCount; ++mode) {
            if (record[mode] <= 0 || record[20 + mode] <= 0 || record[40 + mode] <= 0) throw std::invalid_argument("Response material frequencies and RT60 must be positive");
            mode_factor.push_back({record[mode], record[10 + mode], record[20 + mode]});
            noise[mode] = record[30 + mode];
            noise[10 + mode] = record[40 + mode];
        }
        noise_factor.push_back(noise);
    }
    const auto mode_mean = Center(mode_factor);
    const auto noise_mean = Center(noise_factor);
    return {.Records = uint32_t(records.size()), .ModeMean = mode_mean, .NoiseMean = noise_mean, .ModeFactor = std::move(mode_factor), .NoiseFactor = std::move(noise_factor)};
}

ResponseParameters SampleResponseMaterial(const ResponseMaterial &material, RandomState &random, ResponseMaterialSampleStats *stats) {
    if (material.Records < 2 || material.ModeFactor.size() != size_t(material.Records) * ResponseModeCount || material.NoiseFactor.size() != material.Records || !Finite(material.ModeMean) || !Finite(material.NoiseMean) || !std::ranges::all_of(material.ModeFactor, Finite<3>) || !std::ranges::all_of(material.NoiseFactor, Finite<20>)) throw std::invalid_argument("Invalid response material distribution");
    ResponseParameters result{};
    for (uint32_t mode = 0; mode < ResponseModeCount; ++mode) {
        const auto sample = Draw(material.ModeMean, material.ModeFactor, random, [](const auto &x) { return x[0] >= 20 && x[0] <= 20000 && x[2] > 0; }, stats ? &stats->ModeRejections : nullptr);
        result[mode] = sample[0];
        result[10 + mode] = sample[1];
        result[20 + mode] = sample[2];
    }
    const auto noise = Draw(material.NoiseMean, material.NoiseFactor, random, [](const auto &x) { return std::all_of(x.begin() + 10, x.end(), [](float rt60) { return rt60 > 0; }); }, stats ? &stats->NoiseRejections : nullptr);
    std::ranges::copy(noise, result.begin() + 30);
    return result;
}
} // namespace surface_audio::agarwal
