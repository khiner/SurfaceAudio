#include "conan/Conan.h"
#include "conan/Reference.h"
#include "core/Gpu.h"

#include <array>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace surface_audio;
using namespace surface_audio::conan;

static void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
static bool Near(double actual, double expected, double tolerance) { return std::abs(actual - expected) <= tolerance; }

static void TestControls() {
    const auto smooth = MakeParameters({.Roughness = 0});
    const auto rough = MakeParameters({.Roughness = 1});
    const std::array actual_endpoints{smooth.Amplitude.Sigma, smooth.Amplitude.A1, smooth.Amplitude.B1, smooth.Amplitude.Mean, smooth.Interval.Sigma, smooth.Interval.A1, smooth.Interval.B1, smooth.Interval.Mean, rough.Amplitude.Sigma, rough.Amplitude.A1, rough.Amplitude.B1, rough.Amplitude.Mean, rough.Interval.Sigma, rough.Interval.A1, rough.Interval.B1, rough.Interval.Mean};
    const std::array expected_endpoints{.04, -.97, .07, .43, .00019, -.97, -.34, .0031, .04, -.93, .32, .27, .00085, -.93, .35, .0064};
    for (size_t index = 0; index < actual_endpoints.size(); ++index) Require(Near(actual_endpoints[index], expected_endpoints[index], 1e-7), "Every Table I endpoint and millisecond-to-second conversion");
    Require(Near(smooth.Amplitude.Mean, 0.43, 1e-7) && Near(rough.Amplitude.Mean, 0.27, 1e-7), "Table I amplitude mean");
    Require(Near(smooth.Interval.Sigma, 0.00019, 1e-9) && Near(rough.Interval.Sigma, 0.00085, 1e-9), "Table I interval innovation deviation");
    const auto small = MakeParameters({.Size = 0.25f}), large = MakeParameters({.Size = 0.5f});
    Require(Near(Duration(large, 0.5f) / Duration(small, 0.5f), 2, 1e-6), "Equation 16 size scaling");
    Require(Near(Duration(large, 1) / Duration(large, 0.5f), std::pow(2., -0.29), 1e-6), "Equation 15 amplitude-duration exponent");
    Require(Near(small.ModulationHz / large.ModulationHz, 2, 1e-6), "Equation 17 inverse size modulation");
    Require(Near(MakeParameters({.Velocity = 1}).ModulationHz / MakeParameters({.Velocity = 0.5f}).ModulationHz, 2, 1e-6), "Equation 17 speed modulation");
    Require(Pulse(0, 0.001f) == 1 && Pulse(0.001f, 0.001f) == 0 && Near(Pulse(0.00025f, 0.001f), .5, 1e-6), "Equation 9 raised cosine");
    double integral = 0;
    for (int index = -10000; index <= 10000; ++index) integral += Pulse(index * 1e-7f, 0.001f) * 1e-7;
    Require(Near(integral, 0.0005, 1e-9), "Raised cosine area = duration/2");
    for (float x = -6; x < 6; x += 0.01f) Require(Near(NormalCdf(x), 0.5 * std::erfc(-x / std::sqrt(2.)), 3e-7), "Portable normal CDF accuracy");
}

static void TestStatistics() {
    auto parameters = MakeParameters({.Roughness = 0.6f});
    auto state = MakeState(879);
    FilterState whitened_amplitude{}, whitened_interval{};
    std::vector<double> amplitudes, intervals;
    constexpr unsigned count = 250000;
    double mean_a = 0, mean_t = 0, covariance = 0, variance_a = 0, variance_t = 0, lag_a = 0, lag_t = 0, previous_a = 0, previous_t = 0;
    for (unsigned index = 0; index < count + 2000; ++index) {
        const auto event = NextEvent(parameters, state);
        const double a = event.Amplitude - parameters.Amplitude.Mean, t = event.Interval - parameters.Interval.Mean;
        const double residual_a = Whiten(parameters.Amplitude, whitened_amplitude, static_cast<float>(a)) / parameters.Amplitude.Sigma;
        const double residual_t = Whiten(parameters.Interval, whitened_interval, static_cast<float>(t)) / parameters.Interval.Sigma;
        if (index < 2000) continue;
        Require(Near(residual_a, residual_t, 2e-5), "Shared whitened amplitude/timing innovations");
        mean_a += a;
        mean_t += t;
        variance_a += a * a;
        variance_t += t * t;
        covariance += a * t;
        lag_a += a * previous_a;
        lag_t += t * previous_t;
        previous_a = a;
        previous_t = t;
        if (amplitudes.size() < 50000) {
            amplitudes.push_back(event.Amplitude);
            intervals.push_back(event.Interval);
        }
    }
    auto variance = [](const Process &p) { return p.Sigma * p.Sigma * (1 + p.B1 * p.B1 - 2 * p.A1 * p.B1) / (1 - p.A1 * p.A1); };
    auto lag = [&](const Process &p) { return -p.A1 * variance(p) + p.B1 * p.Sigma * p.Sigma; };
    const double expected_covariance = parameters.Amplitude.Sigma * parameters.Interval.Sigma * (1 + (parameters.Amplitude.B1 - parameters.Amplitude.A1) * (parameters.Interval.B1 - parameters.Interval.A1) / (1 - parameters.Amplitude.A1 * parameters.Interval.A1));
    Require(std::abs(mean_a / count) < 0.005 && std::abs(mean_t / count) < 0.00008, "Stationary means");
    Require(Near(variance_a / count / variance(parameters.Amplitude), 1, 0.05) && Near(variance_t / count / variance(parameters.Interval), 1, 0.05), "Analytic ARMA stationary variances");
    Require(Near(lag_a / count / lag(parameters.Amplitude), 1, 0.05) && Near(lag_t / count / lag(parameters.Interval), 1, 0.05), "Analytic lag-one covariance");
    Require(Near(covariance / count / expected_covariance, 1, 0.05), "Analytic correlated-process covariance");
    const auto fit = FitProcess(amplitudes, true);
    Require(Near(fit.Model.A1, parameters.Amplitude.A1, 0.015) && Near(fit.Model.B1, parameters.Amplitude.B1, 0.03), "Gauss-Newton ARMA coefficient recovery");
    Require(Near(fit.Model.Sigma, parameters.Amplitude.Sigma, 0.002) && std::abs(fit.ResidualLagOne) < 0.02, "Whitening residual variance and lag-one correlation");
    Require(Near(Quantile(fit.Model, .5f), 0, .002), "Empirical inverse CDF median");
    double quantile_mean = 0, quantile_variance = 0;
    for (unsigned index = 1; index < QuantileCount; ++index) {
        const double left = fit.Model.Quantiles[index - 1], right = fit.Model.Quantiles[index];
        const double probability = QuantileProbability(index) - QuantileProbability(index - 1);
        quantile_mean += .5 * (left + right) * probability;
        quantile_variance += (left * left + left * right + right * right) * probability / 3;
    }
    Require(std::abs(quantile_mean) < 1e-8, "Empirical inverse CDF preserves the separately restored process mean");
    Require(Near(quantile_variance / fit.ResidualVariance, 1, .03), "Tail-resolved inverse CDF preserves whitening residual variance");
    Process uniform{};
    for (unsigned index = 0; index < QuantileCount; ++index) uniform.Quantiles[index] = 2 * QuantileProbability(index) - 1;
    for (unsigned index = 0; index <= 10000; ++index) Require(Near(Quantile(uniform, index / 10000.f), 2 * index / 10000. - 1, 3e-7), "Nonuniform quantile knots interpolate in probability space");
    Process curved{};
    std::array<double, QuantileCount> probabilities{};
    for (unsigned index = 0; index < QuantileCount; ++index) {
        const double offset = index <= (QuantileCount - 1) / 2 ? index : QuantileCount - 1 - index;
        const double tail = 2 * offset * offset / ((QuantileCount - 1) * (QuantileCount - 1));
        probabilities[index] = index <= (QuantileCount - 1) / 2 ? tail : 1 - tail;
        curved.Quantiles[index] = static_cast<float>(std::expm1(3. * index / (QuantileCount - 1)) / std::expm1(3.));
    }
    auto check_curved = [&](float probability) {
        unsigned upper = 1;
        // Linear search is independent of Quantile's inverse-grid index calculation.
        while (upper + 1 < QuantileCount && probabilities[upper] < probability) ++upper;
        const double fraction = (probability - probabilities[upper - 1]) / (probabilities[upper] - probabilities[upper - 1]);
        const double expected = std::lerp(double(curved.Quantiles[upper - 1]), double(curved.Quantiles[upper]), fraction);
        Require(Near(Quantile(curved, probability), expected, 2e-7), "Curved inverse CDF selects the independent probability interval");
    };
    for (unsigned index = 0; index <= 10000; ++index) check_curved(index / 10000.f);
    for (unsigned index = 1; index + 1 < QuantileCount; ++index) {
        const float knot = static_cast<float>(probabilities[index]);
        check_curved(knot);
        check_curved(std::nextafter(knot, 0.f));
        check_curved(std::nextafter(knot, 1.f));
    }
}

static void TestStreamingAndCalibration() {
    const auto parameters = MakeParameters({.Roughness = 1});
    auto state = MakeState(77), blocked_state = state;
    std::vector<float> whole(144000), blocked(whole.size());
    Render(parameters, state, whole);
    for (size_t begin = 0; begin < blocked.size(); begin += 127) Render(parameters, blocked_state, std::span(blocked).subspan(begin, std::min(size_t(127), blocked.size() - begin)));
    Require(whole == blocked, "Exact streaming block continuity");
    Require(state.PulseOverflows == 0 && state.Events > 300, "Bounded scheduler event capacity");
    Require(std::ranges::all_of(whole, [](float value) { return std::isfinite(value) && value >= 0; }), "Finite nonnegative force");
    const auto impacts = ExtractImpacts(whole, parameters.SampleRate, 0.01);
    Require(impacts.size() > 300, "Impact extraction");
    const auto calibrated = Calibrate(impacts, parameters.SampleRate);
    Require(calibrated.Interval.Mean > 0 && calibrated.DurationScale > 0, "End-to-end force calibration");
    std::vector<Impact> analytic;
    for (unsigned index = 0; index < 200; ++index) {
        const double amplitude = 0.5 + 0.2 * std::sin(index * 0.73);
        analytic.push_back({index * 0.004, amplitude, 0.0006 * std::pow(amplitude, -0.29)});
    }
    Require(Near(Calibrate(analytic, 48000).DurationScale, 0.0006, 1e-9), "Calibrated duration least-squares scale");
    std::vector<float> pulse(2048);
    for (size_t index = 0; index < pulse.size(); ++index) pulse[index] = 0.7f * Pulse((static_cast<float>(index) - 1000.3f) / 48000, 0.002f);
    const auto extracted = ExtractImpacts(pulse, 48000);
    Require(extracted.size() == 1 && Near(extracted[0].Time * 48000, 1000.3, 0.002) && Near(extracted[0].Duration, 0.002, 1e-6), "Subsample peak and raised-cosine contact duration");
}

static void TestCalibrationPeakSeparation() {
    std::vector<float> force(12000);
    constexpr double sample_rate = 48000;
    for (unsigned index = 0; index < force.size(); ++index) force[index] = .3f + .1f * std::cos(2 * std::numbers::pi * index / 120);
    auto impacts = ExtractImpacts(force, sample_rate);
    Require(impacts.size() == 99 && std::ranges::all_of(impacts, [](const Impact &impact) { return impact.Duration == 0; }), "Continuous-contact peaks remain events without fabricated isolated durations");
    for (unsigned index = 0; index < 8; ++index) impacts[index].Duration = .0006 * std::pow(impacts[index].Amplitude, -.29);
    const auto model = Calibrate(impacts, sample_rate);
    Require(Near(model.Interval.Mean, .0025, 1e-9) && Near(model.DurationScale, .0006, 1e-9), "Timing uses all peaks while duration uses measured contacts");
    bool rejected = false;
    for (auto &impact : impacts) impact.Duration = 0;
    try {
        Calibrate(impacts, sample_rate);
    } catch (const std::invalid_argument &) { rejected = true; }
    Require(rejected, "Calibration must not fabricate a duration law without measured durations");
}

static void TestEquationTraceCase(const Parameters &parameters, unsigned frames, uint64_t seed) {
    auto event_state = MakeState(seed), render_state = event_state;
    std::vector<float> actual(frames);
    Render(parameters, render_state, actual);
    std::vector<double> expected(frames);
    const double delay = parameters.MaximumDuration * .5;
    unsigned amplitude_clamps = 0, interval_clamps = 0, duration_clamps = 0, event_count = 0;
    for (double event_time = 0; event_time < double(frames) / parameters.SampleRate;) {
        ++event_count;
        const auto event = NextEvent(parameters, event_state);
        const double amplitude = std::max(event.Amplitude, parameters.MinimumAmplitude);
        const double interval = std::max(event.Interval, parameters.MinimumInterval);
        amplitude_clamps += event.Amplitude < parameters.MinimumAmplitude;
        interval_clamps += event.Interval < parameters.MinimumInterval;
        const double raw_duration = parameters.DurationScale * std::pow(amplitude, -parameters.DurationExponent);
        duration_clamps += raw_duration > parameters.MaximumDuration;
        const double duration = std::min(double(parameters.MaximumDuration), raw_duration);
        const double center = event_time + delay;
        const auto begin = static_cast<unsigned>(std::max(0., std::ceil((center - duration / 2) * parameters.SampleRate)));
        const auto end = static_cast<unsigned>(std::min(double(frames), std::floor((center + duration / 2) * parameters.SampleRate) + 1));
        for (unsigned frame = begin; frame < end; ++frame) {
            const double time = frame / double(parameters.SampleRate);
            expected[frame] += amplitude * .5 * (1 + std::cos(2 * std::numbers::pi * (time - center) / duration));
        }
        event_time += interval;
    }
    double maximum_error = 0, difference_energy = 0, expected_energy = 0;
    for (unsigned frame = 0; frame < frames; ++frame) {
        expected[frame] *= 1 + parameters.ModulationDepth * std::sin(2 * std::numbers::pi * parameters.ModulationHz * (frame / double(parameters.SampleRate) - delay));
        const double difference = actual[frame] - expected[frame];
        maximum_error = std::max(maximum_error, std::abs(difference));
        difference_energy += difference * difference;
        expected_energy += expected[frame] * expected[frame];
    }
    std::cout << "Conan event counts " << render_state.Events << "/" << event_count << " clamps " << render_state.AmplitudeClamps << "/" << amplitude_clamps << " interval " << render_state.IntervalClamps << "/" << interval_clamps << " phase " << render_state.Phase << '\n';
    std::cout << "Conan Eq5/7/9 direct sum: max_error=" << maximum_error << ", relative_RMS=" << std::sqrt(difference_energy / expected_energy) << '\n';
    Require(maximum_error < .0002 && std::sqrt(difference_energy / expected_energy) < .0001, "Delayed streaming force must match independent paper-equation sum");
    Require(render_state.AmplitudeClamps == amplitude_clamps && render_state.IntervalClamps == interval_clamps && render_state.DurationClamps == duration_clamps && !render_state.PulseOverflows, "Gaussian tail clamps and overlapping pulse capacity are accounted for exactly");
}

static void TestEquationTrace() {
    TestEquationTraceCase(MakeParameters({.Size = .7f, .Velocity = .8f, .Roughness = .5f, .Asymmetry = .8f}, 48000), 144000, 832);
    auto clamped = MakeParameters({.Size = 1, .Velocity = .2f, .Roughness = 1, .Asymmetry = .8f}, 48000);
    clamped.Amplitude.Mean = .01f;
    clamped.Interval.Mean = .0005f;
    TestEquationTraceCase(clamped, 12000, 92);
}

static void TestPhysicalReference() {
    ReferenceParameters parameters{};
    parameters.Dissipation = 0;
    parameters.SampleRate = 96000;
    const auto slow = MeasureContact(parameters, 0.01), fast = MeasureContact(parameters, 0.02);
    Require(Near(slow.ExitVelocity, -0.01, 1e-8), "Elastic contact energy/restitution");
    Require(Near(fast.Duration / slow.Duration, std::pow(2., -0.2), 1e-4), "Hertz analytic contact-duration exponent -1/5");
    const double peak_compression = std::pow((parameters.Exponent + 1) * parameters.Mass * .01 * .01 / (2 * parameters.Stiffness), 1 / (parameters.Exponent + 1));
    Require(Near(slow.PeakForce / ContactForce(parameters, peak_compression, 0), 1, 1e-4), "Hertz peak force from conserved energy");
    std::vector<Impact> isolated;
    for (unsigned index = 1; index <= 10; ++index) {
        const auto contact = MeasureContact(parameters, index * .005);
        isolated.push_back({0, contact.PeakForce, contact.Duration});
    }
    const auto law = FitDurationLaw(isolated);
    Require(Near(law.Exponent, 1. / 6, 1e-4) && law.RSquared > .99999, "Isolated-contact calibration recovers Hertz amplitude exponent 1/6");
    const double exponent = parameters.Exponent;
    const double beta_integral = std::tgamma(1 / (exponent + 1)) * std::tgamma(.5) / std::tgamma(1 / (exponent + 1) + .5) / (exponent + 1);
    const double exact_duration = 2 * peak_compression / .01 * beta_integral;
    Require(Near(slow.Duration / exact_duration, 1, 1e-6), "Absolute Hertz contact duration from independent beta integral");
    parameters.Dissipation = 1;
    const auto damped = MeasureContact(parameters, 0.01);
    Require(std::abs(damped.ExitVelocity) < .01 && damped.ExitVelocity < 0, "Dissipative contact loses energy");
    const double residual = std::log((1 + parameters.Dissipation * damped.ExitVelocity) / (1 + parameters.Dissipation * .01)) - parameters.Dissipation * (damped.ExitVelocity - .01);
    Require(std::abs(residual) < 1e-9, "Hunt-Crossley analytic outgoing-velocity relation");
    Require(Near(ContactStiffness(.02, 7e10, .2, 7e10, .2) / ContactStiffness(.01, 7e10, .2, 7e10, .2), std::sqrt(2.), 1e-12), "Equation 4 stiffness-radius scaling");
    const std::array flat_surface{0., 0.};
    ReferenceState equilibrium{.Position = std::pow(parameters.Gravity * parameters.Mass / parameters.Stiffness, 1 / parameters.Exponent)};
    std::vector<float> constant_force(512);
    RenderReference(parameters, equilibrium, flat_surface, constant_force);
    for (float force : constant_force) Require(Near(force, parameters.Mass * parameters.Gravity, 1e-8), "Rigid flat-surface equilibrium force balances gravity");
    const auto surface = FractalSurface(4096, -1, 1e-7, 99);
    const auto curve = RollingCurve(surface, parameters.Spacing, .01);
    for (size_t index = 0; index < curve.size(); ++index) Require(curve[index] >= surface[index], "Sphere envelope does not penetrate asperities");
    ReferenceState state{}, blocked_state{};
    std::vector<float> whole(8000), blocked(whole.size());
    RenderReference(parameters, state, curve, whole);
    RenderReference(parameters, blocked_state, curve, std::span(blocked).first(311));
    RenderReference(parameters, blocked_state, curve, std::span(blocked).subspan(311));
    Require(whole == blocked && state.Position == blocked_state.Position, "RK4 reference block continuity");
    auto refined_parameters = parameters;
    refined_parameters.Substeps *= 2;
    ReferenceState refined_state{};
    std::vector<float> refined(whole.size());
    RenderReference(refined_parameters, refined_state, curve, refined);
    double error_energy = 0, reference_energy = 0;
    for (size_t index = 0; index < whole.size(); ++index) {
        const double difference = whole[index] - refined[index];
        error_energy += difference * difference;
        reference_energy += refined[index] * refined[index];
    }
    Require(std::sqrt(error_energy / reference_energy) < .005, "Full reference force trace converges under RK4 step refinement");
    std::cout << "Conan physical contact: duration=" << slow.Duration << "s, peak=" << slow.PeakForce << "N, elastic exit=" << slow.ExitVelocity << "m/s\n";
}

static void TestGpu() {
    constexpr unsigned voices = 8, frames = 16384;
    auto gpu = CreateGpu();
    const auto kernel = CreateKernel(gpu, "ConanSynthesize");
    std::array<Parameters, voices> parameters{};
    std::array<State, voices> states{};
    for (unsigned voice = 0; voice < voices; ++voice) {
        parameters[voice] = MakeParameters({.Size = 0.1f + 0.12f * voice, .Velocity = 0.2f + 0.1f * voice, .Roughness = voice / 7.f});
        if (voice == voices - 1) {
            parameters[voice].Amplitude.Empirical = parameters[voice].Interval.Empirical = 1;
            for (unsigned index = 0; index < QuantileCount; ++index) {
                const float quantile = std::sqrt(3.f) * (2.f * QuantileProbability(index) - 1);
                parameters[voice].Amplitude.Quantiles[index] = parameters[voice].Amplitude.Sigma * quantile;
                parameters[voice].Interval.Quantiles[index] = parameters[voice].Interval.Sigma * quantile;
            }
        }
        states[voice] = MakeState(431, voice);
    }
    const auto parameters_buffer = Upload<Parameters>(gpu, std::span<const Parameters>(parameters));
    const auto states_buffer = Upload<State>(gpu, std::span<const State>(states));
    const auto output_buffer = CreateBuffer(gpu, voices * frames * sizeof(float));
    const auto frames_buffer = Upload(gpu, frames);
    const std::array<GpuBinding, 4> bindings{{{parameters_buffer, 0}, {states_buffer, 1}, {output_buffer, 2}, {frames_buffer, 3}}};
    BeginGpu(gpu);
    DispatchGpu(gpu, kernel, bindings, {voices, 1, 1}, {8, 1, 1});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto output = BufferSpan<float>(output_buffer);
    const std::vector<float> whole(output.begin(), output.end());
    std::vector<float> cpu(frames);
    double maximum_error = 0, squared_error = 0, energy = 0;
    for (unsigned voice = 0; voice < voices; ++voice) {
        auto state = states[voice];
        Render(parameters[voice], state, cpu);
        for (unsigned frame = 0; frame < frames; ++frame) {
            const double error = output[voice * frames + frame] - cpu[frame];
            maximum_error = std::max(maximum_error, std::abs(error));
            squared_error += error * error;
            energy += cpu[frame] * cpu[frame];
        }
        const auto &gpu_state = BufferSpan<State>(states_buffer)[voice];
        Require(gpu_state.Events == state.Events && gpu_state.Random.State == state.Random.State && gpu_state.PulseOverflows == 0, "GPU event counts and RNG state");
    }
    Require(maximum_error < 0.0005 && std::sqrt(squared_error / energy) < 0.0002, "Complete GPU/CPU force trace equivalence");
    std::copy(states.begin(), states.end(), BufferSpan<State>(states_buffer).begin());
    constexpr unsigned block_frames = 256;
    BufferSpan<unsigned>(frames_buffer)[0] = block_frames;
    for (unsigned begin = 0; begin < frames; begin += block_frames) {
        BeginGpu(gpu);
        DispatchGpu(gpu, kernel, bindings, {voices, 1, 1}, {8, 1, 1});
        SubmitGpu(gpu);
        WaitGpu(gpu);
        for (unsigned voice = 0; voice < voices; ++voice)
            for (unsigned frame = 0; frame < block_frames; ++frame) Require(output[voice * block_frames + frame] == whole[voice * frames + begin + frame], "GPU exact block continuity");
    }
    std::cout << "Conan Metal 4 full trace: max_error=" << maximum_error << ", relative_RMS=" << std::sqrt(squared_error / energy) << " on " << DeviceName(gpu) << '\n';
}

int main() {
    try {
        TestControls();
        TestStatistics();
        TestStreamingAndCalibration();
        TestCalibrationPeakSeparation();
        TestEquationTrace();
        TestPhysicalReference();
        TestGpu();
        std::cout << "Conan controls, correlations, whitening, calibration, streaming, and physical reference passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
