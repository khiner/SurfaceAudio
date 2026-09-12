#pragma once
#include "Random.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <stdexcept>

namespace surface_audio {
template<size_t D> struct Gaussian {
    uint32_t Count{};
    std::array<double, D> Mean{};
    std::array<std::array<double, D>, D> Factor{};
};

// Sample covariance uses 1/(n-1); zero-variance dimensions retain their exact means.
template<size_t D> Gaussian<D> FitGaussian(std::span<const std::array<double, D>> observations) {
    if (observations.size() < 2 || observations.size() > UINT32_MAX) throw std::invalid_argument("Gaussian fitting requires at least two observations");
    Gaussian<D> result{};
    std::array<std::array<double, D>, D> sum{};
    for (const auto &x : observations) {
        ++result.Count;
        std::array<double, D> delta{};
        for (size_t i = 0; i < D; ++i) {
            if (!std::isfinite(x[i])) throw std::invalid_argument("Nonfinite Gaussian observation");
            delta[i] = x[i] - result.Mean[i];
            result.Mean[i] += delta[i] / result.Count;
        }
        for (size_t i = 0; i < D; ++i)
            for (size_t j = 0; j <= i; ++j) sum[i][j] += delta[i] * (x[j] - result.Mean[j]);
    }
    std::array<double, D> scale{};
    std::array<std::array<double, D>, D> lower{};
    for (size_t i = 0; i < D; ++i) scale[i] = std::sqrt(std::max(0., sum[i][i] / (result.Count - 1)));
    for (size_t i = 0; i < D; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double residual = scale[i] > 0 && scale[j] > 0 ? sum[i][j] / (result.Count - 1) / scale[i] / scale[j] : 0;
            for (size_t k = 0; k < j; ++k) residual -= lower[i][k] * lower[j][k];
            if (i == j) {
                if (residual < -1e-10) throw std::invalid_argument("Indefinite Gaussian covariance");
                lower[i][j] = std::sqrt(std::max(0., residual));
            } else if (lower[j][j] > 1e-10) lower[i][j] = residual / lower[j][j];
            else if (std::abs(residual) > 1e-10) throw std::invalid_argument("Inconsistent singular Gaussian covariance");
            result.Factor[i][j] = scale[i] * lower[i][j];
        }
    }
    return result;
}

template<size_t D> std::array<double, D> SampleGaussian(const Gaussian<D> &distribution, RandomState &random) {
    std::array<double, D> noise{};
    for (double &x : noise) x = Normal(random);
    auto result = distribution.Mean;
    for (size_t i = 0; i < D; ++i)
        for (size_t j = 0; j <= i; ++j) result[i] += distribution.Factor[i][j] * noise[j];
    return result;
}
}
