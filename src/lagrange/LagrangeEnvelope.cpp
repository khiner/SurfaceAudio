#include "Lagrange.h"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace surface_audio::lagrange {
std::vector<float> DeconvolveEnvelopeSections(std::span<const float> envelope, std::span<const float> shape) {
    if (envelope.empty() || shape.empty() || envelope.size() > (1u << 23) || shape.size() > (1u << 23))
        throw std::invalid_argument("Invalid split-envelope dimensions");
    for (float value : envelope)
        if (!(value >= 0) || !std::isfinite(value)) throw std::invalid_argument("Invalid split-envelope input");
    for (float value : shape)
        if (!(value >= 0) || !std::isfinite(value)) throw std::invalid_argument("Invalid split-envelope shape");
    const size_t peak = size_t(std::max_element(shape.begin(), shape.end()) - shape.begin());
    if (!peak || !(shape[peak - 1] > 0)) throw std::invalid_argument("Split-envelope inversion requires a nonzero attack section");
    for (size_t n = 1; n < shape.size(); ++n)
        if (n <= peak ? shape[n] < shape[n - 1] : shape[n] > shape[n - 1]) throw std::invalid_argument("Split-envelope shape must be unimodal");
    std::vector<double> result(envelope.begin(), envelope.end());
    const auto invert = [&](std::span<const float> denominator) {
        const double leading = denominator.front();
        std::vector<double> reversed(denominator.size() - 1);
        for (size_t n = 1; n < denominator.size(); ++n) reversed[denominator.size() - n - 1] = denominator[n];
        for (size_t n = 0; n < result.size(); ++n) {
            const size_t count = std::min(n, reversed.size());
            double feedback = 0;
            if (count) vDSP_dotprD(reversed.data() + reversed.size() - count, 1, result.data() + n - count, 1, &feedback, count);
            result[n] = (result[n] - feedback) / leading;
            if (!std::isfinite(result[n])) throw std::overflow_error("Nonfinite split-envelope inverse");
        }
    };
    // Section V-C applies the decay inverse forward, followed by the reversed attack inverse backward, with zero boundary states.
    invert(shape.subspan(peak));
    std::ranges::reverse(result);
    const std::vector<float> attack(shape.rend() - peak, shape.rend());
    invert(attack);
    std::ranges::reverse(result);
    std::vector<float> output(result.size());
    for (size_t n = 0; n < result.size(); ++n) {
        if (std::abs(result[n]) > std::numeric_limits<float>::max()) throw std::overflow_error("Split-envelope inverse exceeds float range");
        output[n] = float(result[n]);
    }
    return output;
}
}
