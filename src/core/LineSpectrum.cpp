#include "LineSpectrum.h"
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace surface_audio {
std::vector<double> LineSpectrumPolynomial(std::span<const double> frequencies) {
    std::vector<double> sum(frequencies.size() + 2), difference(sum.size());
    sum[0] = difference[0] = 1;
    double previous = 0;
    for (size_t i = 0; i < frequencies.size(); ++i) {
        const double angle = frequencies[i];
        if (!(angle > previous && angle < std::numbers::pi)) throw std::invalid_argument("Line frequencies must increase within (0, pi)");
        previous = angle;
        auto &polynomial = i % 2 ? difference : sum;
        const double middle = -2 * std::cos(angle);
        for (size_t j = i - i % 2 + 2; j > 0; --j)
            polynomial[j] += middle * polynomial[j - 1] + (j > 1 ? polynomial[j - 2] : 0);
    }
    for (size_t j = frequencies.size() + 1; j > 0; --j) {
        if (frequencies.size() % 2) difference[j] -= j > 1 ? difference[j - 2] : 0;
        else {
            sum[j] += sum[j - 1];
            difference[j] -= difference[j - 1];
        }
    }
    sum.resize(frequencies.size() + 1);
    for (size_t i = 0; i < sum.size(); ++i) sum[i] = .5 * (sum[i] + difference[i]);
    return sum;
}
}
