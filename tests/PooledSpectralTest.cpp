#include "core/GpuPooledSpectral.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>
#include <numbers>
#include <numeric>
#include <vector>

using namespace surface_audio;
namespace {
void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
struct ReferenceCells {
    std::vector<double> Power;
    std::vector<uint32_t> Count;
    uint32_t Times;
};

// Independent scalar double DFT and center/bin membership; no GPU transforms,
// regions, membership maps, target scales or reduction buffers are consulted.
ReferenceCells Cells(std::span<const double> samples, uint32_t rate, uint32_t size, uint32_t hop, std::span<const SpectralPoolBand> bands, SpectralPoolOptions options) {
    const uint32_t times = 1 + (options.EventSamples - 1) / options.TimePoolSamples;
    ReferenceCells result{std::vector<double>(bands.size() * times), std::vector<uint32_t>(bands.size() * times), times};
    for (uint32_t center = 0; center < options.EventSamples; center += hop) {
        const uint32_t time = center / options.TimePoolSamples;
        for (uint32_t bin = 0; bin <= size / 2; ++bin) {
            const double frequency = double(bin) * rate / size;
            for (uint32_t band = 0; band < bands.size(); ++band) {
                if (frequency < bands[band].LowerHz || frequency >= bands[band].UpperHz) continue;
                std::complex<double> sum = 0;
                for (uint32_t n = 0; n < size; ++n) {
                    const int64_t sample = int64_t(center) + n - size / 2;
                    if (sample >= 0 && sample < int64_t(samples.size())) {
                        const double window = .5 - .5 * std::cos(2 * std::numbers::pi * n / size);
                        sum += samples[size_t(sample)] * window * std::polar(1., -2 * std::numbers::pi * bin * n / size);
                    }
                }
                result.Power[band * times + time] += std::norm(sum);
                ++result.Count[band * times + time];
            }
        }
    }
    return result;
}

double Loss(const ReferenceCells &current, const ReferenceCells &target, float epsilon, SpectralPoolOptions options) {
    const uint32_t bands = uint32_t(target.Power.size()) / target.Times;
    std::vector<double> rms(bands);
    for (uint32_t band = 0; band < bands; ++band) {
        double energy = 0, count = 0;
        for (uint32_t time = 0; time < target.Times; ++time) {
            energy += target.Power[band * target.Times + time];
            count += target.Count[band * target.Times + time];
        }
        rms[band] = std::sqrt(energy / count + double(epsilon) * epsilon);
    }
    const double floor = std::max(double(epsilon), options.RelativeFloor * *std::max_element(rms.begin(), rms.end()));
    double loss = 0;
    uint32_t cells = 0;
    for (uint32_t index = 0; index < target.Power.size(); ++index) {
        if (!target.Count[index]) continue;
        const double a = std::sqrt(current.Power[index] / current.Count[index] + double(epsilon) * epsilon);
        const double b = std::sqrt(target.Power[index] / target.Count[index] + double(epsilon) * epsilon);
        const double difference = (a - b) / std::max(rms[index / target.Times], floor);
        loss += .5 * difference * difference;
        ++cells;
    }
    return options.Weight * loss / cells;
}
void Evaluate(Gpu &gpu, const SpectralLossGpu &spectral, const GpuPooledSpectral &pool, GpuBuffer input, bool pooled = true) {
    BeginGpu(gpu);
    EncodeSpectralLoss(gpu, spectral, input);
    if (pooled) EncodePooledSpectral(gpu, spectral, pool);
    SubmitGpu(gpu);
    WaitGpu(gpu);
}

void ReferenceTest(Gpu &gpu, uint32_t resolution, bool silent_target, float epsilon) {
    const uint32_t size = std::array{4096u, 1024u, 256u, 64u}[resolution];
    const uint32_t samples = size == 64 ? 257 : 319;
    const uint32_t event = size == 64 ? 139 : 217;
    constexpr uint32_t rate = 16000;
    constexpr std::array bands{SpectralPoolBand{0, 3000}, SpectralPoolBand{3000, 6000}, SpectralPoolBand{6000, 8000}};
    const SpectralLossOptions base_options{.MagnitudeFloor = epsilon, .Scale = SpectralMagnitudeScale::Linear};
    const SpectralPoolOptions options{.Weight = .4f, .Resolution = resolution, .EventSamples = event, .TimePoolSamples = size == 64 ? 40u : 1600u};
    std::vector<float> target(samples), waveform(samples);
    for (uint32_t sample = 0; sample < samples; ++sample) {
        target[sample] = !silent_target && sample < event ? float(.07 * std::sin(.21 * sample) + .02 * std::cos(.71 * sample)) : 0;
        waveform[sample] = float(.81 * target[sample] + .009 * std::sin(.43 * sample));
    }
    const auto spectral = CreateSpectralLossGpu(gpu, target, rate, base_options);
    const auto pool = CreatePooledSpectral(gpu, spectral, bands, options);
    const auto input = Upload<float>(gpu, waveform);
    Evaluate(gpu, spectral, pool, input, false);
    const std::vector<float> base_gradient(BufferSpan<float>(spectral.Gradient).begin(), BufferSpan<float>(spectral.Gradient).end());
    const std::vector<float> base_loss(BufferSpan<float>(spectral.Loss).begin(), BufferSpan<float>(spectral.Loss).end());
    const auto disabled = CreatePooledSpectral(gpu, spectral, {}, {});
    Evaluate(gpu, spectral, disabled, input);
    Require(std::memcmp(base_gradient.data(), spectral.Gradient.Data, spectral.Gradient.Size) == 0 && std::memcmp(base_loss.data(), spectral.Loss.Data, spectral.Loss.Size) == 0, "Disabled pooled term preserves all base loss and gradient bytes");
    Evaluate(gpu, spectral, pool, input);
    const std::vector<double> target_double(target.begin(), target.end()), waveform_double(waveform.begin(), waveform.end());
    const auto reference_target = Cells(target_double, rate, size, size / 4, bands, options);
    const double expected = Loss(Cells(waveform_double, rate, size, size / 4, bands, options), reference_target, epsilon, options);
    const double actual = BufferSpan<float>(pool.Loss)[0];
    Require(std::abs(actual - expected) < 3e-6 * std::max(1., std::abs(expected)), "Pooled loss agrees with independent double DFT");
    for (uint32_t index = 1; index < 5; ++index) Require(BufferSpan<float>(spectral.Loss)[index] == base_loss[index], "Pooled term retains every original resolution loss");
    Require(std::abs(BufferSpan<float>(spectral.Loss)[0] - (base_loss[0] + actual)) < 1e-6 * std::max(1., std::abs(actual)), "Total is base plus weighted pooled term");
    const auto combined = BufferSpan<float>(spectral.Gradient);
    std::vector<double> gradient(samples);
    for (uint32_t sample = 0; sample < samples; ++sample) gradient[sample] = double(combined[sample]) - base_gradient[sample];
    double worst = 0;
    for (uint32_t trial = 0; trial < (size == 64 ? 3u : 1u); ++trial) {
        std::vector<double> direction(samples), plus(waveform_double), minus(waveform_double);
        if (trial == 1) direction[event - 1] = 1;
        else if (trial == 2) direction.back() = 1;
        else
            for (uint32_t sample = 0; sample < samples; ++sample) direction[sample] = std::sin(.17 * sample) / std::sqrt(double(samples));
        const double analytic = std::inner_product(gradient.begin(), gradient.end(), direction.begin(), 0.);
        for (double step : {1e-5, 3e-6}) {
            for (uint32_t sample = 0; sample < samples; ++sample) {
                plus[sample] = waveform_double[sample] + step * direction[sample];
                minus[sample] = waveform_double[sample] - step * direction[sample];
            }
            const double numeric = (Loss(Cells(plus, rate, size, size / 4, bands, options), reference_target, epsilon, options) - Loss(Cells(minus, rate, size, size / 4, bands, options), reference_target, epsilon, options)) / (2 * step);
            const double error = std::abs(analytic - numeric) / std::max(1., std::abs(numeric));
            worst = std::max(worst, error);
            Require(error < 3e-4, "Pooled sample adjoint agrees with independent finite difference");
        }
    }
    if (size == 64) {
        Require(pool.Pools == 12, "Final partial event pool retained with unequal bin and frame counts");
        Require(gradient.back() == 0 && std::abs(base_gradient.back()) > 0, "Pure tail outside pooled support retains original linear gradient");
    }
    std::copy(target.begin(), target.end(), BufferSpan<float>(input).begin());
    Evaluate(gpu, spectral, pool, input);
    Require(BufferSpan<float>(pool.Loss)[0] == 0 && BufferSpan<float>(spectral.Loss)[0] == 0 && std::ranges::all_of(BufferSpan<float>(spectral.Gradient), [](float value) { return value == 0; }), "Identical input has exact zero base and pooled loss/gradient");
    std::ranges::fill(BufferSpan<float>(input), 0);
    Evaluate(gpu, spectral, pool, input);
    Require(std::isfinite(BufferSpan<float>(pool.Loss)[0]) && std::ranges::all_of(BufferSpan<float>(spectral.Gradient), [](float value) { return value == 0; }), "Exact synthesis silence gives finite loss and zero adjoint");
    Require(silent_target ? BufferSpan<float>(pool.Loss)[0] == 0 : BufferSpan<float>(pool.Loss)[0] > 0, "Silent synthesis distinguishes silent and nonsilent targets");
    std::cout << "Pooled FFT " << size << " silent target " << silent_target << " epsilon " << epsilon << " value " << actual << " reference " << expected << " directional error " << worst << '\n';
}
void MaskBoundaryTest(Gpu &gpu) {
    const std::vector<float> target(65);
    const auto spectral = CreateSpectralLossGpu(gpu, target, 16000, {.Scale = SpectralMagnitudeScale::Linear});
    constexpr std::array bands{SpectralPoolBand{0, 8000}};
    const auto pool = CreatePooledSpectral(gpu, spectral, bands, {.Weight = 1, .Resolution = 3, .EventSamples = 64, .TimePoolSamples = 7});
    Require(pool.Pools == 4, "Pool spans shorter than hop omit empty spans");
    const auto regions = BufferSpan<SpectralPoolRegion>(pool.Regions);
    for (uint32_t index = 0; index < pool.Pools; ++index) Require(regions[index].FirstFrame == index && regions[index].EndFrame == index + 1, "Exact event boundary excludes frame center at event end");
    const auto membership = BufferSpan<uint32_t>(pool.Membership);
    Require(membership[32] == UINT32_MAX && membership[4 * 33] == UINT32_MAX, "Nyquist upper bound and event-end frame are excluded");
}
void InvalidTest(Gpu &gpu) {
    const std::vector<float> target(65);
    const auto spectral = CreateSpectralLossGpu(gpu, target, 16000, {.Scale = SpectralMagnitudeScale::Linear});
    constexpr std::array valid{SpectralPoolBand{0, 3000}};
    for (const SpectralPoolOptions options : {SpectralPoolOptions{.Weight = -1}, SpectralPoolOptions{.Weight = 1}, SpectralPoolOptions{.Weight = 1, .RelativeFloor = 0, .EventSamples = 65, .TimePoolSamples = 40}, SpectralPoolOptions{.Weight = 1, .EventSamples = 66, .TimePoolSamples = 40}}) {
        bool rejected = false;
        try {
            CreatePooledSpectral(gpu, spectral, valid, options);
        } catch (const std::invalid_argument &) { rejected = true; }
        Require(rejected, "Invalid pooled options rejected");
    }
    constexpr std::array overlap{SpectralPoolBand{0, 3000}, SpectralPoolBand{2000, 4000}};
    bool rejected = false;
    try {
        CreatePooledSpectral(gpu, spectral, overlap, {.Weight = 1, .EventSamples = 65, .TimePoolSamples = 40});
    } catch (const std::invalid_argument &) { rejected = true; }
    Require(rejected, "Overlapping pooled bins rejected");
}
} // namespace

int main() {
    try {
        auto gpu = CreateGpu();
        ReferenceTest(gpu, 3, false, 1e-6f);
        ReferenceTest(gpu, 3, true, .02f);
        ReferenceTest(gpu, 0, false, 1e-6f);
        MaskBoundaryTest(gpu);
        InvalidTest(gpu);
        std::cout << "Pooled spectral tests passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
