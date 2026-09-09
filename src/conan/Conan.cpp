#include "Conan.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace surface_audio::conan {

Parameters MakeParameters(Controls controls, float sample_rate) {
    if (!std::isfinite(controls.Size) || !std::isfinite(controls.Velocity) || !std::isfinite(controls.Roughness) || !std::isfinite(controls.Asymmetry)) throw std::invalid_argument("Conan controls must be finite");
    controls.Size = std::clamp(controls.Size, 0.1f, 1.f);
    controls.Velocity = std::clamp(controls.Velocity, 0.1f, 1.f);
    controls.Roughness = std::clamp(controls.Roughness, 0.f, 1.f);
    controls.Asymmetry = std::clamp(controls.Asymmetry, 0.f, 1.f);
    const float roughness = controls.Roughness;
    Parameters parameters{};
    parameters.SampleRate = sample_rate;
    parameters.Amplitude = {.Mean = std::lerp(0.43f, 0.27f, roughness), .Sigma = 0.04f, .A1 = std::lerp(-0.97f, -0.93f, roughness), .B1 = std::lerp(0.07f, 0.32f, roughness)};
    parameters.Interval = {.Mean = std::lerp(0.0031f, 0.0064f, roughness), .Sigma = std::lerp(0.00019f, 0.00085f, roughness), .A1 = parameters.Amplitude.A1, .B1 = std::lerp(-0.34f, 0.35f, roughness)};
    parameters.DurationScale = 0.000788f * controls.Size;
    parameters.ModulationHz = 3 * controls.Velocity / controls.Size;
    parameters.ModulationDepth = controls.Asymmetry;
    Validate(parameters);
    return parameters;
}

State MakeState(uint64_t seed, uint64_t stream) { return {.Random = MakeRandom(seed, stream)}; }

void Validate(const Parameters &parameters) {
    auto validate_process = [](const Process &process) {
        if (!std::isfinite(process.Mean) || !std::isfinite(process.Sigma) || !std::isfinite(process.A1) || !std::isfinite(process.B1) || process.Sigma < 0 || std::abs(process.A1) >= 1 || std::abs(process.B1) >= 1 || process.Empirical > 1) throw std::invalid_argument("Invalid Conan ARMA process");
        if (process.Empirical) {
            for (unsigned index = 0; index < QuantileCount; ++index)
                if (!std::isfinite(process.Quantiles[index]) || (index && process.Quantiles[index] < process.Quantiles[index - 1])) throw std::invalid_argument("Invalid Conan inverse CDF");
        }
    };
    validate_process(parameters.Amplitude);
    validate_process(parameters.Interval);
    if (!std::isfinite(parameters.SampleRate) || parameters.SampleRate < 8000 || parameters.SampleRate > 384000 || !std::isfinite(parameters.DurationScale) || parameters.DurationScale <= 0 || !std::isfinite(parameters.DurationExponent) || parameters.DurationExponent < 0 || !std::isfinite(parameters.ModulationHz) || parameters.ModulationHz < 0 || parameters.ModulationHz >= parameters.SampleRate / 2 || !std::isfinite(parameters.ModulationDepth) || parameters.ModulationDepth < 0 || parameters.ModulationDepth > 1 || !std::isfinite(parameters.MinimumAmplitude) || parameters.MinimumAmplitude <= 0 || !std::isfinite(parameters.MinimumInterval) || parameters.MinimumInterval < 1 / parameters.SampleRate || !std::isfinite(parameters.MaximumDuration) || parameters.MaximumDuration <= 0 || parameters.MaximumDuration / parameters.MinimumInterval > PulseCapacity - 2) throw std::invalid_argument("Invalid Conan synthesis parameters");
}

void Render(const Parameters &parameters, State &state, std::span<float> output) {
    Validate(parameters);
    for (float &sample : output) sample = Step(parameters, state);
}

static void Residuals(std::span<const double> centered, double a, double b, std::span<double> residuals) {
    for (size_t index = 0; index < centered.size(); ++index) residuals[index] = centered[index] + (index ? a * centered[index - 1] - b * residuals[index - 1] : 0);
}

Fit FitProcess(std::span<const double> series, bool empirical) {
    if (series.size() < 32) throw std::invalid_argument("Conan ARMA fit requires at least 32 events");
    for (double value : series)
        if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite Conan event");
    const double mean = std::accumulate(series.begin(), series.end(), 0.) / series.size();
    std::vector<double> centered(series.size());
    std::ranges::transform(series, centered.begin(), [mean](double value) { return value - mean; });
    double a = -0.8, b = 0;
    std::vector<double> residuals(series.size()), candidate(series.size());
    Residuals(centered, a, b, residuals);
    auto energy = [](std::span<const double> values) { return std::inner_product(values.begin() + 8, values.end(), values.begin() + 8, 0.); };
    for (unsigned iteration = 0; iteration < 80; ++iteration) {
        double aa = 1e-20, ab = 0, bb = 1e-20, ae = 0, be = 0, derivative_a = 0, derivative_b = 0;
        for (size_t index = 1; index < centered.size(); ++index) {
            derivative_a = centered[index - 1] - b * derivative_a;
            derivative_b = -residuals[index - 1] - b * derivative_b;
            if (index < 8) continue;
            aa += derivative_a * derivative_a;
            ab += derivative_a * derivative_b;
            bb += derivative_b * derivative_b;
            ae += derivative_a * residuals[index];
            be += derivative_b * residuals[index];
        }
        const double determinant = aa * bb - ab * ab;
        if (determinant <= 1e-28) break;
        const double delta_a = (bb * ae - ab * be) / determinant;
        const double delta_b = (aa * be - ab * ae) / determinant;
        const double previous_energy = energy(residuals);
        bool accepted = false;
        for (double rate = 1; rate >= 1.0 / 1024; rate *= 0.5) {
            const double candidate_a = std::clamp(a - rate * delta_a, -0.995, 0.995);
            const double candidate_b = std::clamp(b - rate * delta_b, -0.995, 0.995);
            Residuals(centered, candidate_a, candidate_b, candidate);
            if (energy(candidate) >= previous_energy) continue;
            a = candidate_a;
            b = candidate_b;
            residuals.swap(candidate);
            accepted = true;
            break;
        }
        if (!accepted || std::max(std::abs(delta_a), std::abs(delta_b)) < 1e-8) break;
    }
    const double variance = energy(residuals) / (residuals.size() - 8);
    double lag_one = 0;
    for (size_t index = 9; index < residuals.size(); ++index) lag_one += residuals[index] * residuals[index - 1];
    Fit fit{.Model = {.Mean = static_cast<float>(mean), .Sigma = static_cast<float>(std::sqrt(variance)), .A1 = static_cast<float>(a), .B1 = static_cast<float>(b), .Empirical = empirical}, .ResidualVariance = variance, .ResidualLagOne = variance > 0 ? lag_one / ((residuals.size() - 9) * variance) : 0};
    std::sort(residuals.begin() + 8, residuals.end());
    for (unsigned index = 0; index < QuantileCount; ++index) {
        const double position = 8 + double(QuantileProbability(index)) * (residuals.size() - 9);
        const size_t lower = static_cast<size_t>(position);
        fit.Model.Quantiles[index] = static_cast<float>(std::lerp(residuals[lower], residuals[std::min(lower + 1, residuals.size() - 1)], position - lower));
    }
    // Integrate the interpolated inverse CDF and its tails to keep innovations zero-mean.
    double quantile_mean = 0;
    for (unsigned index = 1; index < QuantileCount; ++index)
        quantile_mean += .5 * (fit.Model.Quantiles[index - 1] + fit.Model.Quantiles[index]) * (QuantileProbability(index) - QuantileProbability(index - 1));
    for (float &quantile : fit.Model.Quantiles) quantile -= static_cast<float>(quantile_mean);
    return fit;
}

std::vector<Impact> ExtractImpacts(std::span<const float> force, double sample_rate, double threshold) {
    if (!std::isfinite(sample_rate) || sample_rate <= 0 || !std::isfinite(threshold) || threshold < 0) throw std::invalid_argument("Invalid Conan extraction parameters");
    for (float sample : force)
        if (!std::isfinite(sample)) throw std::invalid_argument("Nonfinite force sample");
    std::vector<Impact> impacts;
    for (size_t index = 1; index + 1 < force.size(); ++index) {
        if (force[index] <= threshold || force[index] <= force[index - 1] || force[index] < force[index + 1]) continue;
        const double denominator = force[index - 1] - 2. * force[index] + force[index + 1];
        const double offset = denominator != 0 ? 0.5 * (force[index - 1] - force[index + 1]) / denominator : 0;
        const double amplitude = force[index] - 0.25 * (force[index - 1] - force[index + 1]) * offset;
        // A neighboring minimum must not be crossed when measuring a single pulse's FWHM.
        const double half = amplitude * 0.5;
        size_t left = index, right = index;
        while (left && force[left] > half && force[left - 1] <= force[left]) --left;
        while (right + 1 < force.size() && force[right] > half && force[right + 1] <= force[right]) ++right;
        double duration = 0;
        if (left < index && right > index && force[left] <= half && force[right] <= half) {
            const double left_time = left + (half - force[left]) / (force[left + 1] - force[left]);
            const double right_time = right - (half - force[right]) / (force[right - 1] - force[right]);
            duration = 2 * (right_time - left_time) / sample_rate;
        }
        impacts.push_back({(index + offset) / sample_rate, amplitude, duration});
    }
    return impacts;
}

DurationLaw FitDurationLaw(std::span<const Impact> impacts, double fixed_exponent) {
    if (impacts.size() < 2 || !std::isfinite(fixed_exponent)) throw std::invalid_argument("Duration-law fit requires at least two contacts");
    double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
    for (const auto &impact : impacts) {
        if (!std::isfinite(impact.Amplitude) || impact.Amplitude <= 0 || !std::isfinite(impact.Duration) || impact.Duration <= 0) throw std::invalid_argument("Invalid contact duration or amplitude");
        const double x = std::log(impact.Amplitude), y = std::log(impact.Duration);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    double exponent = fixed_exponent;
    if (exponent < 0) {
        const double denominator = impacts.size() * sum_xx - sum_x * sum_x;
        if (denominator <= 1e-20) throw std::invalid_argument("Duration exponent requires varying contact amplitudes");
        exponent = -(impacts.size() * sum_xy - sum_x * sum_y) / denominator;
    }
    double numerator = 0, denominator = 0, mean = 0;
    for (const auto &impact : impacts) {
        const double predictor = std::pow(impact.Amplitude, -exponent);
        numerator += predictor * impact.Duration;
        denominator += predictor * predictor;
        mean += impact.Duration;
    }
    const double scale = numerator / denominator;
    mean /= impacts.size();
    double errors = 0, total = 0;
    for (const auto &impact : impacts) {
        const double error = impact.Duration - scale * std::pow(impact.Amplitude, -exponent);
        errors += error * error;
        total += (impact.Duration - mean) * (impact.Duration - mean);
    }
    return {scale, exponent, total > 0 ? 1 - errors / total : (errors < 1e-20 ? 1 : 0)};
}

Parameters Calibrate(std::span<const Impact> impacts, float sample_rate, bool empirical, float duration_exponent) {
    if (impacts.size() < 33 || !std::isfinite(duration_exponent) || duration_exponent < 0) throw std::invalid_argument("Conan calibration requires 33 valid impacts");
    std::vector<double> amplitudes, intervals;
    std::vector<Impact> measured_durations;
    for (size_t index = 0; index < impacts.size(); ++index) {
        const auto &impact = impacts[index];
        if (!std::isfinite(impact.Time) || !std::isfinite(impact.Amplitude) || impact.Amplitude <= 0 || !std::isfinite(impact.Duration) || impact.Duration < 0 || (index && impact.Time <= impacts[index - 1].Time)) throw std::invalid_argument("Invalid calibration impacts");
        if (impact.Duration > 0) measured_durations.push_back(impact);
        if (index + 1 < impacts.size()) {
            amplitudes.push_back(impact.Amplitude);
            intervals.push_back(impacts[index + 1].Time - impact.Time);
        }
    }
    Parameters parameters = MakeParameters({}, sample_rate);
    parameters.Amplitude = FitProcess(amplitudes, empirical).Model;
    parameters.Interval = FitProcess(intervals, empirical).Model;
    parameters.DurationExponent = duration_exponent;
    parameters.DurationScale = static_cast<float>(FitDurationLaw(measured_durations, duration_exponent).Scale);
    parameters.ModulationDepth = 0;
    Validate(parameters);
    return parameters;
}

} // namespace surface_audio::conan
