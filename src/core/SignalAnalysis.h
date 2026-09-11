#pragma once
#include <complex>
#include <cstdint>
#include <span>
#include <vector>

namespace surface_audio {
struct LinearPrediction {
    std::vector<double> Denominator;
    double Error{};
};
struct DampedSinusoid {
    std::complex<double> Pole, Gain;
};

// Requires a power-of-two record and normalizes the inverse by its length.
void FourierTransform(std::span<std::complex<double>>, bool inverse = false);
std::vector<std::complex<double>> FourierTransform(std::span<const double>, uint32_t size);
// Returns [1,a1,...] and unnormalized prediction energy from biased autocorrelation, with order less than the sample count.
LinearPrediction FitLinearPrediction(std::span<const double>, uint32_t order);
// Returns sampled complex poles and fitted gains.
// Rows defaults to half the record and must exceed poles.
std::vector<DampedSinusoid> EstimateEsprit(std::span<const std::complex<double>>, uint32_t poles, uint32_t rows = 0);
void FitDampedGains(std::span<const std::complex<double>>, std::span<DampedSinusoid>);
}
