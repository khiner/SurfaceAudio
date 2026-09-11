#include "Lee.h"
#include "LeeInternal.h"
#include "Qmf.h"
#include "core/SignalAnalysis.h"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace surface_audio::lee {
namespace {
constexpr double Pi = std::numbers::pi;
std::vector<double> Convolve(std::span<const double> a, std::span<const double> b) {
    if (a.empty() || b.empty()) return {};
    std::vector<double> result(a.size() + b.size() - 1);
    for (size_t j = 0; j < b.size(); ++j) {
        const double scale = b[j];
        vDSP_vsmaD(a.data(), 1, &scale, result.data() + j, 1, result.data() + j, 1, a.size());
    }
    return result;
}
std::vector<double> Resample(std::span<const double> input, std::span<const double> taps, size_t frames, uint32_t up, uint32_t down, size_t delay) {
    std::vector<double> result(frames);
    if (input.empty()) return result;
    for (size_t n = 0; n < frames; ++n) {
        const size_t position = n * down + delay;
        const size_t first = position >= taps.size() ? (position - taps.size()) / up + 1 : 0;
        const size_t last = std::min(input.size() - 1, position / up);
        if (first <= last)
            vDSP_dotprD(input.data() + first, 1, taps.data() + position - first * up, -vDSP_Stride(up), &result[n], last - first + 1);
    }
    return result;
}
std::vector<double> Dilate(std::span<const double> values, uint32_t factor) {
    std::vector<double> result((values.size() - 1) * factor + 1);
    for (size_t i = 0; i < values.size(); ++i) result[i * factor] = values[i];
    return result;
}
void Validate(const Settings &s, uint32_t rate) {
    if (rate == 0 || !std::isfinite(s.HighpassHz) || s.HighpassHz <= 0 || s.HighpassHz >= rate * .5 ||
        s.EnvelopeFrames == 0 || s.HighpassTaps < 3 || s.HighpassTaps % 2 == 0 ||
        !std::isfinite(s.EnvelopeThreshold) || s.EnvelopeThreshold <= 0 || s.EnvelopeThreshold >= 1 ||
        !std::isfinite(s.NotchProminenceDb) || s.NotchProminenceDb < 0 || !std::isfinite(s.TailSeconds) ||
        s.TailSeconds < 0 || s.TailSeconds > 10 || s.MaximumNotches > 64)
        throw std::invalid_argument("Invalid Lee analysis settings");
    for (auto order : s.LpcOrder)
        if (order == 0 || order > 128) throw std::invalid_argument("Invalid Lee LPC order");
}
}
uint32_t CheckedFrameCount(double count) {
    if (!std::isfinite(count) || count < 0 || count > UINT32_MAX) throw std::invalid_argument("Lee frame count exceeds uint32 range");
    return uint32_t(count);
}
FirBank MakeFilterBank() {
    FirBank bank;
    const std::vector<double> low(QmfLowpass.begin(), QmfLowpass.end());
    auto high = low;
    for (size_t i = 1; i < high.size(); i += 2) high[i] = -high[i];
    bank.Analysis[3] = high;
    bank.Analysis[2] = Convolve(low, Dilate(high, 2));
    const auto low2 = Convolve(low, Dilate(low, 2));
    bank.Analysis[1] = Convolve(low2, Dilate(high, 4));
    bank.Analysis[0] = Convolve(low2, Dilate(low, 4));
    bank.Synthesis = bank.Analysis;
    for (size_t b = 1; b < 4; ++b)
        for (auto &x : bank.Synthesis[b]) x = -x;
    return bank;
}
std::array<std::vector<double>, 4> SplitBands(std::span<const double> input, const FirBank &bank) {
    std::array<std::vector<double>, 4> bands;
    if (input.empty()) return bands;
    for (size_t b = 0; b < 4; ++b)
        bands[b] = Resample(input, bank.Analysis[b], (input.size() + bank.Analysis[b].size() - 2) / Decimation[b] + 1, 1, Decimation[b], 0);
    return bands;
}
std::vector<double> MergeBands(const std::array<std::vector<double>, 4> &bands, const FirBank &bank, uint32_t frames, bool analysis_delay) {
    std::vector<double> output(frames);
    for (size_t b = 0; b < 4; ++b) {
        if (bands[b].empty()) continue;
        const size_t delay = (bank.Synthesis[b].size() - 1) / (analysis_delay ? 1 : 2);
        const auto filtered = Resample(bands[b], bank.Synthesis[b], frames, Decimation[b], 1, delay);
        vDSP_vaddD(output.data(), 1, filtered.data(), 1, output.data(), 1, frames);
    }
    return output;
}
std::vector<double> HighpassCoefficients(uint32_t rate, const Settings &s) {
    Validate(s, rate);
    std::vector<double> high(s.HighpassTaps);
    const int middle = s.HighpassTaps / 2;
    const double cutoff = s.HighpassHz / rate;
    double low_sum = 0;
    for (uint32_t i = 0; i < s.HighpassTaps; ++i) {
        const int n = int(i) - middle;
        high[i] = (n ? std::sin(2 * Pi * cutoff * n) / (Pi * n) : 2 * cutoff) * (.54 - .46 * std::cos(2 * Pi * i / (s.HighpassTaps - 1)));
        low_sum += high[i];
    }
    for (auto &x : high) x /= -low_sum;
    high[middle] += 1;
    return high;
}
std::vector<uint32_t> ContactEnvelope(std::span<const double> filtered, const Settings &s, std::vector<double> &envelope) {
    std::vector<double> prefix(filtered.size() + 1);
    for (size_t n = 0; n < filtered.size(); ++n) prefix[n + 1] = prefix[n] + filtered[n] * filtered[n];
    envelope.resize(filtered.size());
    for (size_t n = 0; n < filtered.size(); ++n) {
        const size_t first = n > s.EnvelopeFrames / 2 ? n - s.EnvelopeFrames / 2 : 0;
        const size_t last = std::min(filtered.size(), n + (s.EnvelopeFrames + 1) / 2);
        envelope[n] = (prefix[last] - prefix[first]) / s.EnvelopeFrames;
    }
    const size_t first_valid = s.HighpassTaps / 2 + s.EnvelopeFrames / 2;
    const size_t invalid_end = s.HighpassTaps / 2 + (s.EnvelopeFrames + 1) / 2;
    if (first_valid >= filtered.size() || invalid_end > filtered.size() - first_valid) {
        std::ranges::fill(envelope, 0.);
        return {};
    }
    const size_t end_valid = filtered.size() - invalid_end + 1;
    std::fill(envelope.begin(), envelope.begin() + first_valid, 0.);
    std::fill(envelope.begin() + end_valid, envelope.end(), 0.);
    const double peak = *std::max_element(envelope.begin() + first_valid, envelope.begin() + end_valid);
    std::vector<uint32_t> onsets;
    bool previous = envelope[first_valid] > s.EnvelopeThreshold * peak;
    for (size_t n = first_valid + 1; n < end_valid; ++n) {
        const bool active = envelope[n] > s.EnvelopeThreshold * peak;
        if (active && !previous) onsets.push_back(uint32_t(n));
        previous = active;
    }
    return onsets;
}
std::vector<uint32_t> DetectContacts(std::span<const float> input, uint32_t rate, const Settings &s, std::vector<double> &envelope) {
    CheckedFrameCount(input.size());
    const auto high = HighpassCoefficients(rate, s);
    const std::vector<double> samples(input.begin(), input.end());
    for (double x : samples)
        if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite Lee recording");
    if (samples.empty()) {
        envelope.clear();
        return {};
    }
    return ContactEnvelope(Resample(samples, high, samples.size(), 1, 1, s.HighpassTaps / 2), s, envelope);
}
std::vector<Notch> EstimateNotches(std::span<const double> input, double prominence, uint32_t maximum) {
    if (input.size() > (1u << 25)) throw std::invalid_argument("Lee notch spectrum exceeds FFT bounds");
    if (input.size() < 4 || maximum == 0) return {};
    const uint32_t size = std::bit_ceil(uint32_t(std::max<size_t>(256, input.size() * 4)));
    const auto spectrum = FourierTransform(input, size);
    std::vector<double> reciprocal(size / 2 + 1);
    double maximum_magnitude = 0;
    for (const auto &x : spectrum) maximum_magnitude = std::max(maximum_magnitude, std::abs(x));
    if (maximum_magnitude == 0) return {};
    for (size_t n = 0; n < reciprocal.size(); ++n) reciprocal[n] = -20 * std::log10(std::max(std::abs(spectrum[n]), maximum_magnitude * 1e-8));
    struct Peak {
        Notch Value;
        double Prominence;
    };
    std::vector<Peak> peaks;
    const double spacing = 2 * Pi / size;
    for (size_t i = 2; i + 2 < reciprocal.size(); ++i) {
        if (reciprocal[i] <= reciprocal[i - 1] || reciprocal[i] <= reciprocal[i + 1]) continue;
        double left = reciprocal[i - 1], right = reciprocal[i + 1];
        for (size_t j = i - 1; j > 0 && reciprocal[j] < reciprocal[i]; --j) left = std::min(left, reciprocal[j]);
        for (size_t j = i + 1; j < reciprocal.size() && reciprocal[j] < reciprocal[i]; ++j) right = std::min(right, reciprocal[j]);
        const double height = reciprocal[i] - std::max(left, right);
        if (height < prominence) continue;
        const double curvature = .5 * (reciprocal[i - 1] + reciprocal[i + 1]) - reciprocal[i];
        const double offset = .25 * (reciprocal[i - 1] - reciprocal[i + 1]) / curvature;
        const double bandwidth = 2 * std::sqrt(-3 / curvature) * spacing;
        peaks.push_back({{(i + offset) * spacing, std::clamp(bandwidth, spacing * .25, Pi / 2)}, height});
    }
    std::ranges::sort(peaks, [](const Peak &a, const Peak &b) { return a.Prominence > b.Prominence; });
    const double isolation = -2 * std::log(.95);
    std::vector<Notch> result;
    for (const auto &peak : peaks) {
        const double half_width = (peak.Value.Bandwidth + isolation) / 2;
        if (peak.Value.Frequency <= half_width || peak.Value.Frequency >= Pi - half_width) continue;
        if (std::ranges::any_of(result, [&](const Notch &other) {
                return std::abs(peak.Value.Frequency - other.Frequency) < isolation + (peak.Value.Bandwidth + other.Bandwidth) / 2;
            })) continue;
        result.push_back(peak.Value);
        if (result.size() == maximum) break;
    }
    std::ranges::sort(result, {}, &Notch::Frequency);
    return result;
}
std::vector<double> FilterNotches(std::span<const double> input, std::span<const Notch> notches, bool inverse) {
    std::vector<double> output(input.begin(), input.end());
    for (const auto &notch : notches) {
        if (!std::isfinite(notch.Frequency) || !std::isfinite(notch.Bandwidth) || notch.Frequency <= 0 ||
            notch.Frequency >= Pi || notch.Bandwidth <= 0) throw std::invalid_argument("Invalid Lee notch");
        const double radius = std::exp(-notch.Bandwidth * .5), cosine = std::cos(notch.Frequency);
        const double zero = inverse ? .95 * radius : radius, pole = inverse ? radius : .95 * radius;
        double x1 = 0, x2 = 0, y1 = 0, y2 = 0;
        for (auto &sample : output) {
            const double x = sample;
            sample = x - 2 * zero * cosine * x1 + zero * zero * x2 + 2 * pole * cosine * y1 - pole * pole * y2;
            x2 = x1;
            x1 = x;
            y2 = y1;
            y1 = sample;
        }
    }
    return output;
}
std::array<Band, 4> FitContact(const std::array<std::vector<double>, 4> &signals, const Settings &options) {
    std::array<Band, 4> bands;
    for (size_t b = 0; b < 4; ++b) {
        auto notches = EstimateNotches(signals[b], options.NotchProminenceDb, options.MaximumNotches);
        const auto whitened = FilterNotches(signals[b], notches, true);
        auto prediction = FitLinearPrediction(whitened, std::min<uint32_t>(options.LpcOrder[b], whitened.size() - 1));
        bands[b] = {std::move(prediction.Denominator), std::move(notches), std::sqrt(std::max(0., prediction.Error))};
    }
    return bands;
}
Analysis Analyze(std::span<const float> input, uint32_t rate, Settings options) {
    const auto frames = CheckedFrameCount(input.size());
    std::vector<double> envelope;
    auto onsets = DetectContacts(input, rate, options, envelope);
    const auto bank = MakeFilterBank();
    std::vector<Contact> contacts;
    contacts.reserve(onsets.size());
    for (size_t i = 0; i < onsets.size(); ++i) {
        const uint32_t first = onsets[i], last = i + 1 < onsets.size() ? onsets[i + 1] : frames;
        const std::vector<double> segment(input.begin() + first, input.begin() + last);
        contacts.push_back({first, last - first, FitContact(SplitBands(segment, bank), options)});
    }
    return {rate, frames, options, std::move(onsets), std::move(envelope), std::move(contacts)};
}
RenderPlan PrepareSynthesis(const Analysis &analysis, double rate, double gain, bool reverse) {
    if (!std::isfinite(rate) || rate < .01 || rate > 100 || !std::isfinite(gain))
        throw std::invalid_argument("Invalid Lee synthesis control");
    Validate(analysis.Options, analysis.SampleRate);
    const uint32_t tail = CheckedFrameCount(std::ceil(analysis.SampleRate * analysis.Options.TailSeconds));
    RenderPlan plan{CheckedFrameCount(std::ceil(analysis.Frames / rate) + tail), MakeFilterBank(), {}};
    for (size_t i = 0; i < analysis.Contacts.size(); ++i) {
        const auto &contact = analysis.Contacts[reverse ? analysis.Contacts.size() - 1 - i : i];
        const uint32_t onset = CheckedFrameCount(std::round(analysis.Contacts[i].Frame / rate));
        RenderContact render{onset, CheckedFrameCount(double(contact.Frames) + tail), {}};
        plan.Frames = std::max(plan.Frames, CheckedFrameCount(double(onset) + render.Frames));
        for (size_t b = 0; b < 4; ++b) {
            const auto &band = contact.Bands[b];
            if (band.Denominator.empty() || band.Denominator[0] != 1 || !std::isfinite(band.Gain))
                throw std::invalid_argument("Invalid Lee contact filter");
            std::vector<double> impulse((render.Frames + plan.Filters.Synthesis[b].size()) / Decimation[b] + 1);
            for (size_t n = 0; n < impulse.size(); ++n) {
                double sample = n == 0 ? gain * band.Gain : 0;
                for (size_t k = 1; k < band.Denominator.size() && k <= n; ++k) sample -= band.Denominator[k] * impulse[n - k];
                if (!std::isfinite(sample)) throw std::runtime_error("Unstable Lee contact filter");
                impulse[n] = sample;
            }
            render.Signals[b] = FilterNotches(impulse, band.Notches, false);
        }
        plan.Contacts.push_back(std::move(render));
    }
    return plan;
}
std::vector<float> Synthesize(const Analysis &analysis, double rate, double gain, bool reverse) {
    const auto plan = PrepareSynthesis(analysis, rate, gain, reverse);
    std::vector<double> sum(plan.Frames);
    for (const auto &contact : plan.Contacts) {
        const auto rendered = MergeBands(contact.Signals, plan.Filters, contact.Frames, false);
        for (size_t n = 0; n < rendered.size(); ++n) sum[contact.Frame + n] += rendered[n];
    }
    return {sum.begin(), sum.end()};
}
}
