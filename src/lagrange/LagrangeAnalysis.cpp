#include "Lagrange.h"
#include "core/SignalAnalysis.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace surface_audio::lagrange {
std::vector<Resonance> AnalyzeModes(std::span<const float> signal, uint32_t rate, Interval interval, const AnalysisOptions &options) {
    if (signal.empty() || signal.size() > (1u << 23) || !rate || !options.Bands || !std::has_single_bit(options.Bands) ||
        !options.ModesPerBand || options.ModesPerBand > 1024 || !options.MaximumModes || options.MaximumModes > 1024 ||
        interval.End <= interval.Begin || interval.End > signal.size()) throw std::invalid_argument("Invalid modal analysis parameters");
    const uint32_t size = std::bit_ceil(uint32_t(signal.size()) + 1), decimation = options.Bands;
    if (options.Bands > size / 2) throw std::invalid_argument("Too many subbands");
    std::vector<std::complex<double>> spectrum(size);
    for (size_t n = 0; n < signal.size(); ++n) {
        if (!std::isfinite(signal[n])) throw std::invalid_argument("Nonfinite modal input");
        spectrum[n] = signal[n];
    }
    FourierTransform(spectrum);
    std::vector<DampedSinusoid> candidates;
    for (uint32_t band = 0; band < options.Bands; ++band) {
        std::vector<std::complex<double>> filtered(size);
        const uint32_t low = uint32_t(uint64_t(band) * size / (2 * options.Bands));
        const uint32_t high = uint32_t(uint64_t(band + 1) * size / (2 * options.Bands));
        const uint32_t center = (low + high) / 2;
        for (uint32_t k = low; k < high; ++k) filtered[(k + size - center) % size] = 2. * spectrum[k];
        FourierTransform(filtered, true);
        std::vector<std::complex<double>> observed;
        for (uint32_t n = interval.Begin; n < interval.End; n += decimation) observed.push_back(filtered[n]);
        if (observed.size() < 6) continue;
        const uint32_t order = std::min(options.ModesPerBand, uint32_t(observed.size() / 2 - 1));
        const uint32_t whitening = std::min(options.WhiteningOrder, uint32_t(observed.size() / 8));
        std::vector<std::complex<double>> whitened(observed);
        if (whitening) {
            std::vector<double> real(observed.size());
            for (size_t n = 0; n < real.size(); ++n) real[n] = observed[n].real();
            const auto prediction = FitLinearPrediction(real, whitening);
            for (size_t n = 0; n < observed.size(); ++n) {
                whitened[n] = 0;
                for (size_t k = 0; k < prediction.Denominator.size(); ++k) {
                    const int64_t sample = int64_t(interval.Begin) + int64_t(n * decimation) - int64_t(k * decimation);
                    if (sample >= 0) whitened[n] += prediction.Denominator[k] * filtered[size_t(sample)];
                }
            }
        }
        auto poles = EstimateEsprit(whitened, order);
        FitDampedGains(observed, poles);
        for (auto mode : poles) {
            if (!(std::abs(mode.Pole) < 1 && std::abs(mode.Pole) > 0)) continue;
            const double frequency = std::arg(mode.Pole) / decimation + 2 * std::numbers::pi * center / size;
            if (frequency <= 2 * std::numbers::pi * low / size || frequency >= 2 * std::numbers::pi * high / size) continue;
            mode.Pole = std::polar(std::pow(std::abs(mode.Pole), 1. / decimation), frequency);
            candidates.push_back(mode);
        }
    }
    std::ranges::sort(candidates, [](const auto &a, const auto &b) { return std::abs(a.Gain) > std::abs(b.Gain); });
    if (candidates.size() > options.MaximumModes) candidates.resize(options.MaximumModes);
    std::vector<DampedSinusoid> real_modes;
    for (auto mode : candidates) {
        real_modes.push_back(mode);
        real_modes.push_back({std::conj(mode.Pole), std::conj(mode.Gain)});
    }
    std::vector<std::complex<double>> observed(interval.End - interval.Begin);
    for (size_t n = 0; n < observed.size(); ++n) observed[n] = signal[n + interval.Begin];
    FitDampedGains(observed, real_modes);
    std::vector<Resonance> result;
    for (size_t k = 0; k < candidates.size(); ++k) {
        const auto pole = candidates[k].Pole, gain = real_modes[2 * k].Gain + std::conj(real_modes[2 * k + 1].Gain);
        result.push_back({std::arg(pole) * rate / (2 * std::numbers::pi), -std::log(std::abs(pole)) * rate, gain.real(), gain.imag()});
    }
    std::ranges::sort(result, {}, &Resonance::Frequency);
    return result;
}
}
