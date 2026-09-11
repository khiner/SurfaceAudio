#include "Lagrange.h"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace surface_audio::lagrange {
namespace {
constexpr double Pi = std::numbers::pi;
struct Pole {
    double Angle, Radius, Loss, Gain;
};
std::complex<double> Denominator(Pole pole, double frequency) {
    const double delta = frequency - pole.Angle, sine = std::sin(delta / 2);
    return {pole.Loss + 2 * pole.Radius * sine * sine, pole.Radius * std::sin(delta)};
}
double Valley(Pole left, Pole right) {
    const auto balance = [&](double frequency) {
        return left.Gain / std::abs(Denominator(left, frequency)) - right.Gain / std::abs(Denominator(right, frequency));
    };
    double low = left.Angle, high = right.Angle;
    if (balance(low) >= 0 && balance(high) <= 0) {
        // Bisection avoids cancellation in the closed-form coefficients of DAFx 2008 Eq13.
        for (uint32_t n = 0; n < 64; ++n) {
            const double middle = (low + high) / 2;
            if (balance(middle) > 0) low = middle;
            else high = middle;
        }
        return (low + high) / 2;
    }
    // Eq10 remains defined when unequal gains leave Eq13 without a crossing between the poles.
    const auto power = [&](double frequency) {
        return std::norm(left.Gain / Denominator(left, frequency) + right.Gain / Denominator(right, frequency));
    };
    constexpr uint32_t intervals = 128;
    uint32_t minimum = 0;
    double best = power(low);
    for (uint32_t n = 1; n <= intervals; ++n) {
        const double value = power(std::lerp(left.Angle, right.Angle, double(n) / intervals));
        if (value < best) { best = value; minimum = n; }
    }
    low = std::lerp(left.Angle, right.Angle, double(minimum ? minimum - 1 : 0) / intervals);
    high = std::lerp(left.Angle, right.Angle, double(std::min(minimum + 1, intervals)) / intervals);
    for (uint32_t n = 0; n < 64; ++n) {
        const double first = std::lerp(low, high, .3819660112501051), second = std::lerp(low, high, .6180339887498949);
        if (power(first) < power(second)) high = second;
        else low = first;
    }
    const double refined = (low + high) / 2, sampled = std::lerp(left.Angle, right.Angle, double(minimum) / intervals);
    return power(refined) < best ? refined : sampled;
}
}

std::vector<Resonance> AdaptModalPhases(std::span<const Resonance> modes, double rate) {
    if (!(rate > 0) || !std::isfinite(rate)) throw std::invalid_argument("Invalid phase-adaptation sample rate");
    std::vector<Resonance> result(modes.begin(), modes.end());
    for (auto mode : result)
        if (!(mode.Frequency > 0 && mode.Frequency < rate / 2 && mode.Damping > 0) || !std::isfinite(mode.Damping) ||
            !std::isfinite(std::hypot(mode.RealGain, mode.ImaginaryGain))) throw std::invalid_argument("Invalid phase-adaptation mode");
    std::ranges::sort(result, {}, &Resonance::Frequency);
    Pole previous{};
    double phase = 0;
    bool first = true;
    for (auto &mode : result) {
        const double gain = std::hypot(mode.RealGain, mode.ImaginaryGain);
        if (!gain) continue;
        const Pole current{2 * Pi * mode.Frequency / rate, std::exp(-mode.Damping / rate), -std::expm1(-mode.Damping / rate), gain};
        if (!first) {
            if (!(current.Angle > previous.Angle)) throw std::invalid_argument("Phase adaptation requires distinct mode frequencies");
            const double scale = std::max(previous.Gain, current.Gain);
            const double frequency = Valley({previous.Angle, previous.Radius, previous.Loss, previous.Gain / scale},
                                           {current.Angle, current.Radius, current.Loss, current.Gain / scale});
            // Eq11–12 require adjacent one-pole responses to differ by pi/2 at the selected valley.
            phase = std::remainder(phase - std::arg(Denominator(previous, frequency)) + Pi / 2 + std::arg(Denominator(current, frequency)), 2 * Pi);
        }
        mode = {mode.Frequency, mode.Damping, gain * std::cos(phase), gain * std::sin(phase)};
        previous = current;
        first = false;
    }
    return result;
}
}
