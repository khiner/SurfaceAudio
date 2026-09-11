#pragma once
#include "Gpu.h"
#include <vector>

namespace surface_audio {
struct SparseConvolutionFit {
    std::vector<float> Coefficients;
    double RelativeKkt{};
};
struct ConvolutionObservation {
    std::span<const float> Signal, Impulse;
    double Weight{1};
};
// Minimizes 0.5 * ||impulse * coefficients - signal||² + lambda * sum(coefficients), with nonnegative coefficients.
// The target is zero beyond signal.size(); lambda is relative_penalty times the maximum positive adjoint correlation.
SparseConvolutionFit FitSparseConvolutionGpu(Gpu &, std::span<const float> signal, std::span<const float> impulse, double relative_penalty = .001, uint32_t iterations = 1200);
// Fits shared nonnegative coefficients to equally long observations with positive weights and complete convolution tails.
SparseConvolutionFit FitSparseConvolutionGpu(Gpu &, std::span<const ConvolutionObservation>, double relative_penalty = .001, uint32_t iterations = 1200);
}
