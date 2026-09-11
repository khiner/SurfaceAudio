#include "Lagrange.h"
#include "core/GpuConvolution.h"
#include "core/GpuSparseConvolution.h"
#include "core/SignalAnalysis.h"
#include <bit>
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace surface_audio::lagrange {
namespace {
constexpr double Pi = std::numbers::pi;
}

std::vector<double> Meixner(uint32_t frames, double beta, double gamma) {
    if (!frames || !(beta > 0) || !(gamma > 0 && gamma < 1)) throw std::invalid_argument("Invalid Meixner parameters");
    std::vector<double> result(frames);
    result[0] = std::pow(1 - gamma * gamma, beta / 2);
    for (uint32_t n = 1; n < frames; ++n) result[n] = result[n - 1] * gamma * std::sqrt((beta + n - 1) / n);
    return result;
}

std::vector<float> PeakEnvelope(std::span<const float> signal) {
    if (signal.empty()) return {};
    std::vector<double> knots{0}, values{std::abs(signal.front())};
    for (size_t n = 1; n + 1 < signal.size(); ++n)
        if (std::abs(signal[n]) >= std::abs(signal[n - 1]) && std::abs(signal[n]) > std::abs(signal[n + 1])) {
            knots.push_back(double(n));
            values.push_back(std::abs(signal[n]));
        }
    if (signal.size() > 1) {
        knots.push_back(double(signal.size() - 1));
        values.push_back(std::abs(signal.back()));
    }
    const size_t count = knots.size();
    std::vector<double> second(count), diagonal(count), rhs(count);
    for (size_t k = 1; k + 1 < count; ++k) {
        const double left = knots[k] - knots[k - 1], right = knots[k + 1] - knots[k];
        const double factor = k > 1 ? left / diagonal[k - 1] : 0;
        diagonal[k] = 2 * (left + right) - factor * left;
        rhs[k] = 6 * ((values[k + 1] - values[k]) / right - (values[k] - values[k - 1]) / left) - factor * rhs[k - 1];
    }
    for (size_t k = count > 1 ? count - 2 : 0; k > 0; --k)
        second[k] = (rhs[k] - (knots[k + 1] - knots[k]) * second[k + 1]) / diagonal[k];
    std::vector<float> result(signal.size());
    size_t k = 0;
    for (size_t n = 0; n < signal.size(); ++n) {
        while (k + 2 < count && n > knots[k + 1]) ++k;
        if (count == 1) {
            result[n] = float(values[0]);
            continue;
        }
        const double width = knots[k + 1] - knots[k], a = (knots[k + 1] - n) / width, b = 1 - a;
        result[n] = float(std::max(0., a * values[k] + b * values[k + 1] + ((a * a * a - a) * second[k] + (b * b * b - b) * second[k + 1]) * width * width / 6));
    }
    return result;
}

Interval SelectInterval(std::span<const float> signal, uint32_t sample_rate, uint32_t requested_frames) {
    if (signal.empty() || signal.size() > (1u << 23) || !sample_rate) throw std::invalid_argument("Invalid analysis signal dimensions");
    for (float value : signal)
        if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite analysis signal");
    const uint32_t delta = std::max(1u, uint32_t(std::round(.005 * sample_rate))), frames = uint32_t(signal.size());
    std::vector<double> level((frames + delta - 1) / delta);
    for (size_t k = 0; k < level.size(); ++k) {
        const size_t begin = k * delta, end = std::min(signal.size(), begin + delta);
        for (size_t n = begin; n < end; ++n) level[k] += 20 * std::log10(std::abs(signal[n]) + 1e-12);
        level[k] /= end - begin;
    }
    const uint32_t peak = uint32_t(std::max_element(signal.begin(), signal.end(), [](float a, float b) { return std::abs(a) < std::abs(b); }) - signal.begin());
    const uint32_t center = peak / delta;
    const auto acceptable = [&](uint32_t b, uint32_t e) {
        if (e <= b + 1) return true;
        const double count = e - b + 1, mean_x = .5 * (b + e);
        double mean_y = 0, numerator = 0, denominator = 0, maximum = -1e300;
        for (uint32_t k = b; k <= e; ++k) {
            mean_y += level[k] / count;
            maximum = std::max(maximum, level[k]);
        }
        for (uint32_t k = b; k <= e; ++k) {
            numerator += (k - mean_x) * (level[k] - mean_y);
            denominator += (k - mean_x) * (k - mean_x);
        }
        const double slope = numerator / denominator;
        double error = 0;
        for (uint32_t k = b; k <= e; ++k) error += std::abs(level[k] - mean_y - slope * (k - mean_x));
        return error / ((e - b) * std::max(std::abs(maximum), 1e-12)) < 1.8;
    };
    uint32_t left = center, right = center;
    while (left > 0 && acceptable(left - 1, center)) --left;
    while (right + 1 < level.size() && acceptable(center, right + 1)) ++right;
    uint32_t begin = requested_frames ? peak : left * delta, end = std::min(frames, (right + 1) * delta);
    if (requested_frames && end - begin > requested_frames) {
        begin += (end - begin - requested_frames) / 2;
        end = begin + requested_frames;
    }
    while (begin + 1 < end && std::signbit(signal[begin]) == std::signbit(signal[begin + 1])) ++begin;
    while (end > begin + 1 && std::signbit(signal[end - 1]) == std::signbit(signal[end - 2])) --end;
    return {begin, peak, end};
}

std::vector<float> FitImpactEnvelope(std::span<const float> impact) {
    if (impact.empty()) throw std::invalid_argument("Empty impact");
    const auto envelope = PeakEnvelope(impact);
    const auto base = Meixner();
    const double peak = double(std::max_element(envelope.begin(), envelope.end()) - envelope.begin());
    const double base_peak = double(std::max_element(base.begin(), base.end()) - base.begin());
    double best = 1e300;
    std::vector<float> result(impact.size()), candidate(impact.size());
    for (uint32_t step = 0; step < 129; ++step) {
        const double stretch = impact.size() / 200. * std::exp2((double(step) - 64) / 32);
        const double delay = peak - base_peak * stretch;
        double cross = 0, energy = 0;
        for (size_t n = 0; n < impact.size(); ++n) {
            const double position = (n - delay) / stretch;
            const size_t index = size_t(std::clamp(position, 0., 198.));
            candidate[n] = position >= 0 && position < 199 ? float(std::lerp(base[index], base[index + 1], position - index)) : 0;
            cross += candidate[n] * envelope[n];
            energy += candidate[n] * candidate[n];
        }
        const double gain = cross / std::max(energy, 1e-30);
        double error = 0;
        for (size_t n = 0; n < impact.size(); ++n) {
            candidate[n] *= gain;
            error += std::pow(candidate[n] - envelope[n], 2);
        }
        if (error < best) {
            best = error;
            result = candidate;
        }
    }
    return result;
}

std::vector<float> DeconvolveEnvelope(std::span<const float> envelope, std::span<const float> shape, double relative_floor) {
    if (envelope.empty() || shape.empty() || envelope.size() + shape.size() > (1u << 24) || !(relative_floor > 0 && relative_floor <= 1))
        throw std::invalid_argument("Invalid envelope inverse dimensions");
    const size_t size = std::bit_ceil(envelope.size() + shape.size());
    std::vector<std::complex<double>> input(size), response(size);
    double sum = 0;
    for (size_t n = 0; n < envelope.size(); ++n) {
        if (!std::isfinite(envelope[n])) throw std::invalid_argument("Nonfinite envelope");
        input[n] = envelope[n];
    }
    for (size_t n = 0; n < shape.size(); ++n) {
        if (!(shape[n] >= 0) || !std::isfinite(shape[n])) throw std::invalid_argument("Invalid impact envelope");
        response[n] = shape[n];
        sum += shape[n];
    }
    if (!(sum > 0)) throw std::invalid_argument("Zero impact envelope");
    FourierTransform(input);
    FourierTransform(response);
    const double floor = sum * relative_floor;
    for (size_t n = 0; n < size; ++n) input[n] *= std::conj(response[n]) / (std::norm(response[n]) + floor * floor);
    FourierTransform(input, true);
    std::vector<float> result(envelope.size());
    for (size_t n = 0; n < result.size(); ++n) result[n] = float(input[n].real());
    return result;
}

std::vector<Trigger> FitTriggerAmplitudes(std::span<const float> envelope, std::span<const float> shape,
                                        std::span<const Trigger> candidates, uint32_t iterations) {
    if (envelope.empty() || shape.empty() || envelope.size() > (1u << 23) || shape.size() > (1u << 23) || !iterations)
        throw std::invalid_argument("Invalid amplitude fit dimensions");
    if (candidates.empty()) return {};
    const double peak = *std::max_element(shape.begin(), shape.end());
    if (!(peak > 0)) throw std::invalid_argument("Zero amplitude fit envelope");
    std::vector<double> window(shape.size()), target(envelope.begin(), envelope.end());
    for (size_t n = 0; n < shape.size(); ++n) {
        if (!(shape[n] >= 0) || !std::isfinite(shape[n])) throw std::invalid_argument("Invalid amplitude fit envelope");
        window[n] = shape[n] / peak;
    }
    for (double value : target)
        if (!(value >= 0) || !std::isfinite(value)) throw std::invalid_argument("Invalid amplitude fit target");
    std::vector<double> correlation(window.size());
    for (size_t lag = 0; lag < window.size(); ++lag)
        vDSP_dotprD(window.data(), 1, window.data() + lag, 1, &correlation[lag], window.size() - lag);
    const size_t count = candidates.size();
    for (size_t j = 0; j < count; ++j)
        if (candidates[j].Sample >= envelope.size() || (j && candidates[j].Sample < candidates[j - 1].Sample))
            throw std::invalid_argument("Unsorted amplitude fit locations");
    std::vector<double> rhs(count), amplitude(count), extrapolated(count), next(count);
    std::vector<size_t> offsets(count + 1), first(count);
    std::vector<double> values;
    double lipschitz = 0;
    for (size_t j = 0; j < count; ++j) {
        const size_t position = candidates[j].Sample;
        const size_t available = std::min(window.size(), envelope.size() - position);
        vDSP_dotprD(window.data(), 1, target.data() + position, 1, &rhs[j], available);
        const size_t low = size_t(std::lower_bound(candidates.begin(), candidates.end(), position >= window.size() ? position - window.size() + 1 : 0,
            [](auto event, size_t sample) { return event.Sample < sample; }) - candidates.begin());
        first[j] = low;
        double row_sum = 0;
        for (size_t k = low; k < count && candidates[k].Sample < position + window.size(); ++k) {
            const size_t other = candidates[k].Sample, lag = position > other ? position - other : other - position;
            const size_t overlap = std::min(window.size() - lag, envelope.size() - std::max(position, other));
            double value = correlation[lag];
            if (overlap < window.size() - lag) vDSP_dotprD(window.data(), 1, window.data() + lag, 1, &value, overlap);
            values.push_back(value);
            row_sum += value;
        }
        offsets[j + 1] = values.size();
        lipschitz = std::max(lipschitz, row_sum);
    }
    if (!(lipschitz > 0)) return {};
    const double tolerance = 1e-7 * *std::max_element(rhs.begin(), rhs.end());
    double acceleration = 1;
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
        for (size_t j = 0; j < count; ++j) {
            double product = 0;
            vDSP_dotprD(values.data() + offsets[j], 1, extrapolated.data() + first[j], 1, &product, offsets[j + 1] - offsets[j]);
            next[j] = std::max(0., extrapolated[j] + (rhs[j] - product) / lipschitz);
        }
        const double updated = .5 * (1 + std::sqrt(1 + 4 * acceleration * acceleration));
        for (size_t j = 0; j < count; ++j) extrapolated[j] = next[j] + (acceleration - 1) / updated * (next[j] - amplitude[j]);
        amplitude.swap(next);
        acceleration = updated;
        if (iteration % 25 == 24) {
            double violation = 0;
            for (size_t j = 0; j < count; ++j) {
                double product = 0;
                vDSP_dotprD(values.data() + offsets[j], 1, amplitude.data() + first[j], 1, &product, offsets[j + 1] - offsets[j]);
                const double gradient = product - rhs[j];
                violation = std::max(violation, amplitude[j] > 0 ? std::abs(gradient) : std::max(0., -gradient));
            }
            if (violation <= tolerance) break;
        }
    }
    std::vector<Trigger> result;
    for (size_t j = 0; j < count; ++j)
        if (amplitude[j] > 0) result.push_back({candidates[j].Sample, amplitude[j] / peak});
    return result;
}

std::vector<Trigger> DetectTriggers(std::span<const float> signal, double minimum_peak_fraction) {
    if (!(minimum_peak_fraction >= 0 && minimum_peak_fraction < 1)) throw std::invalid_argument("Invalid trigger pruning threshold");
    for (float value : signal)
        if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite trigger input");
    std::vector<Trigger> result;
    if (signal.empty()) return result;
    std::vector<float> residual(signal.begin(), signal.end());
    std::vector<bool> selected(signal.size());
    const double threshold = *std::max_element(signal.begin(), signal.end()) * minimum_peak_fraction;
    for (;;) {
        double indicator = 0;
        for (size_t n = 1; n + 1 < residual.size(); ++n) {
            const double coefficient = residual[n] > indicator ? .999 : .3;
            indicator = coefficient * indicator + (1 - coefficient) * residual[n];
            if (residual[n] > 0 && residual[n] > indicator && residual[n] >= residual[n - 1] && residual[n] > residual[n + 1] &&
                signal[n] >= signal[n - 1] && signal[n] > signal[n + 1] && !selected[n]) {
                result.push_back({uint32_t(n), signal[n]});
                selected[n] = true;
            }
        }
        const double cutoff = .5 * *std::max_element(residual.begin(), residual.end());
        if (!minimum_peak_fraction || !(cutoff > 0) || cutoff < threshold) break;
        bool removed = false;
        for (size_t n = 1; n + 1 < residual.size(); ++n)
            if (residual[n] > cutoff && residual[n] >= residual[n - 1] && residual[n] > residual[n + 1]) {
                size_t begin = n, end = n;
                // Section V-C omits peak boundaries; remove each dominant peak between its adjacent local minima.
                while (begin && residual[begin - 1] > 0 && residual[begin - 1] <= residual[begin]) --begin;
                while (end + 1 < residual.size() && residual[end + 1] > 0 && residual[end + 1] <= residual[end]) ++end;
                std::fill(residual.begin() + begin, residual.begin() + end + 1, 0);
                n = end;
                removed = true;
            }
        if (!removed) break;
    }
    std::ranges::sort(result, {}, &Trigger::Sample);
    return result;
}

std::vector<float> TriggerSignal(std::span<const Trigger> events, uint32_t frames) {
    std::vector<float> result(frames);
    for (auto event : events) {
        if (event.Sample >= frames) throw std::invalid_argument("Trigger outside signal");
        result[event.Sample] += event.Amplitude;
    }
    return result;
}

std::complex<double> Transfer(std::span<const Resonance> modes, double radians, double sample_rate) {
    std::complex<double> result{}, inverse_z = std::polar(1., -radians);
    for (auto mode : modes) {
        const auto pole = std::polar(std::exp(-mode.Damping / sample_rate), 2 * Pi * mode.Frequency / sample_rate);
        const std::complex<double> gain{mode.RealGain, mode.ImaginaryGain};
        result += .5 * (gain / (1. - pole * inverse_z) + std::conj(gain) / (1. - std::conj(pole) * inverse_z));
    }
    return result;
}

std::vector<float> ModalResponse(std::span<const Resonance> modes, uint32_t frames, double sample_rate) {
    std::vector<float> result(frames);
    for (uint32_t n = 0; n < frames; ++n) {
        double sum = 0;
        const double t = n / sample_rate;
        for (auto mode : modes) sum += std::exp(-mode.Damping * t) * (mode.RealGain * std::cos(2 * Pi * mode.Frequency * t) - mode.ImaginaryGain * std::sin(2 * Pi * mode.Frequency * t));
        result[n] = float(sum);
    }
    return result;
}

Analysis Analyze(Gpu &gpu, std::span<const float> signal, uint32_t rate, const AnalysisOptions &options) {
    if (signal.empty() || signal.size() > (1u << 23) || !rate || !std::isfinite(options.WindowSeconds) ||
        !(options.WindowSeconds > 0 && options.WindowSeconds * rate <= (1u << 23)))
        throw std::invalid_argument("Invalid sustained contact recording dimensions");
    const uint32_t frames = uint32_t(signal.size());
    const auto modal_interval = options.ModalInterval.End ? options.ModalInterval : SelectInterval(signal, rate, uint32_t(rate * options.WindowSeconds));
    auto modes = AnalyzeModes(signal, rate, modal_interval, options);
    if (options.Gains != ModalGain::Fitted && options.Gains != ModalGain::Unit && options.Gains != ModalGain::Magnitude &&
        options.Gains != ModalGain::CausalMagnitude && options.Gains != ModalGain::IncrementalPhase)
        throw std::invalid_argument("Invalid modal gain convention");
    if (options.Gains == ModalGain::IncrementalPhase) modes = AdaptModalPhases(modes, rate);
    else if (options.Gains != ModalGain::Fitted)
        for (auto &mode : modes) {
            const double gain = options.Gains == ModalGain::Unit ? 1 : std::hypot(mode.RealGain, mode.ImaginaryGain);
            const double phase = options.Gains == ModalGain::CausalMagnitude ? 2 * Pi * mode.Frequency / rate - Pi / 2 : 0;
            // JASS ModalObject uses a causal damped sine with a one-sample phase offset.
            mode = {mode.Frequency, mode.Damping, gain * std::cos(phase), gain * std::sin(phase)};
        }
    if (modes.empty()) throw std::runtime_error("No damped modes identified");
    auto excitation = InverseModalGpu(gpu, signal, modes, rate, options.InverseFloor);
    const auto impact_interval = options.ImpactInterval.End ? options.ImpactInterval : SelectInterval(excitation, rate);
    if (impact_interval.End > signal.size() || impact_interval.End <= impact_interval.Begin) throw std::invalid_argument("Invalid impact interval");
    auto impact = [&] {
        std::vector<float> result(excitation.begin() + impact_interval.Begin, excitation.begin() + impact_interval.End);
        if (options.CenterImpact) {
            const double mean = std::accumulate(result.begin(), result.end(), 0.) / result.size();
            for (auto &sample : result) sample = float(sample - mean);
        }
        return result;
    }();
    auto envelope = PeakEnvelope(excitation), impact_envelope = FitImpactEnvelope(impact);
    auto deconvolved = options.Fit == TriggerFit::SplitEnvelope ? DeconvolveEnvelopeSections(envelope, impact_envelope) :
                            DeconvolveEnvelopeGpu(gpu, envelope, impact_envelope, options.EnvelopeFloor);
    if (options.Fit != TriggerFit::Envelope && options.Fit != TriggerFit::Waveform &&
        options.Fit != TriggerFit::SplitEnvelope && options.Fit != TriggerFit::JointWaveform)
        throw std::invalid_argument("Invalid trigger fit");
    const auto fit = [&] {
        if (options.Fit == TriggerFit::Waveform) return FitSparseConvolutionGpu(gpu, excitation, impact);
        if (options.Fit != TriggerFit::JointWaveform) return SparseConvolutionFit{};
        if (!(options.AudioWeight > 0) || !std::isfinite(options.AudioWeight)) throw std::invalid_argument("Invalid audio fit weight");
        const auto energy = [](std::span<const float> values) {
            return std::max(1e-30, std::transform_reduce(values.begin(), values.end(), 0., std::plus{}, [](float x) { return double(x) * x; }));
        };
        const auto response = ModalResponseGpu(gpu, modes, frames, rate), audible = ConvolveFftGpu(gpu, impact, response);
        const std::array observations{ConvolutionObservation{excitation, impact, 1 / energy(excitation)},
                                     ConvolutionObservation{signal, audible, options.AudioWeight / energy(signal)}};
        return FitSparseConvolutionGpu(gpu, observations, .001, 4800);
    }();
    auto triggers = [&] {
        if (options.Fit == TriggerFit::Envelope) return FitTriggerAmplitudes(envelope, impact_envelope, DetectTriggers(deconvolved));
        if (options.Fit == TriggerFit::SplitEnvelope) {
            const uint32_t origin = uint32_t(std::max_element(impact_envelope.begin(), impact_envelope.end()) - impact_envelope.begin()) - 1;
            auto events = DetectTriggers(deconvolved, .2);
            std::erase_if(events, [&](auto event) { return event.Sample < origin; });
            for (auto &event : events) event.Sample -= origin;
            return events;
        }
        std::vector<Trigger> result;
        for (uint32_t n = 0; n < frames; ++n)
            if (fit.Coefficients[n] > 0) result.push_back({n, fit.Coefficients[n]});
        return result;
    }();
    return {rate, frames, modal_interval, impact_interval, std::move(modes), std::move(excitation), std::move(impact),
            std::move(envelope), std::move(impact_envelope), std::move(deconvolved), std::move(triggers), fit.RelativeKkt};
}

std::vector<float> Synthesize(Gpu &gpu, const Analysis &analysis, std::span<const Trigger> triggers) {
    const auto excitation = ConvolveFftGpu(gpu, TriggerSignal(triggers, analysis.Frames), analysis.Impact);
    return FilterModalGpu(gpu, excitation, analysis.Modes, uint32_t(excitation.size() + analysis.Frames - 1), analysis.SampleRate);
}
}
