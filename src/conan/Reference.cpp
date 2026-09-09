#include "Reference.h"

#include "core/Random.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace surface_audio::conan {

double ContactStiffness(double radius, double young_exciter, double poisson_exciter, double young_resonator, double poisson_resonator) {
    if (!std::isfinite(radius) || radius <= 0 || !std::isfinite(young_exciter) || young_exciter <= 0 || !std::isfinite(young_resonator) || young_resonator <= 0 || !std::isfinite(poisson_exciter) || poisson_exciter <= -1 || poisson_exciter >= 0.5 || !std::isfinite(poisson_resonator) || poisson_resonator <= -1 || poisson_resonator >= 0.5) throw std::invalid_argument("Invalid Hertz material properties");
    return (4. / 3) * std::sqrt(radius) / ((1 - poisson_exciter * poisson_exciter) / young_exciter + (1 - poisson_resonator * poisson_resonator) / young_resonator);
}

double ContactForce(const ReferenceParameters &parameters, double compression, double velocity) {
    return compression > 0 ? parameters.Stiffness * std::pow(compression, parameters.Exponent) * (1 + parameters.Dissipation * velocity) : 0;
}

std::vector<double> FractalSurface(unsigned sample_count, double power_exponent, double maximum_height, uint64_t seed) {
    if (sample_count < 4 || (sample_count & (sample_count - 1)) || !std::isfinite(power_exponent) || power_exponent < -4 || power_exponent > 0 || !std::isfinite(maximum_height) || maximum_height < 0) throw std::invalid_argument("Fractal surface requires power-of-two sample count and exponent in [-4,0]");
    auto random = MakeRandom(seed);
    std::vector<std::complex<double>> spectrum(sample_count);
    for (unsigned index = 1; index < sample_count / 2; ++index) {
        const double amplitude = std::pow(static_cast<double>(index), power_exponent / 2);
        spectrum[index] = amplitude * std::complex<double>(Normal(random), Normal(random));
        spectrum[sample_count - index] = std::conj(spectrum[index]);
    }
    spectrum[sample_count / 2] = std::pow(sample_count / 2., power_exponent / 2) * Normal(random);
    for (unsigned index = 1, reverse = 0; index < sample_count; ++index) {
        unsigned bit = sample_count >> 1;
        for (; reverse & bit; bit >>= 1) reverse ^= bit;
        reverse ^= bit;
        if (index < reverse) std::swap(spectrum[index], spectrum[reverse]);
    }
    for (unsigned size = 2; size <= sample_count; size *= 2) {
        const auto rotation = std::polar(1., 2 * std::numbers::pi / size);
        for (unsigned begin = 0; begin < sample_count; begin += size) {
            std::complex<double> phase = 1;
            for (unsigned index = 0; index < size / 2; ++index) {
                const auto first = spectrum[begin + index], second = phase * spectrum[begin + index + size / 2];
                spectrum[begin + index] = first + second;
                spectrum[begin + index + size / 2] = first - second;
                phase *= rotation;
            }
        }
    }
    double maximum = 0;
    for (auto value : spectrum) maximum = std::max(maximum, std::abs(value.real()));
    std::vector<double> surface(sample_count);
    for (unsigned index = 0; index < sample_count; ++index) surface[index] = maximum ? maximum_height * spectrum[index].real() / maximum : 0;
    return surface;
}

std::vector<double> RollingCurve(std::span<const double> surface, double spacing, double radius) {
    if (surface.empty() || !std::isfinite(spacing) || spacing <= 0 || !std::isfinite(radius) || radius <= 0) throw std::invalid_argument("Invalid rolling surface");
    for (double height : surface)
        if (!std::isfinite(height)) throw std::invalid_argument("Nonfinite surface height");
    const auto extent = static_cast<int64_t>(std::min(std::floor(radius / spacing), static_cast<double>(surface.size() / 2)));
    std::vector<double> sag(size_t(2 * extent + 1));
    for (int64_t offset = -extent; offset <= extent; ++offset) {
        const double distance = offset * spacing;
        sag[size_t(offset + extent)] = distance * distance / (radius + std::sqrt(std::max(0., radius * radius - distance * distance)));
    }
    std::vector<double> curve(surface.size());
    const auto count = int64_t(surface.size());
    for (int64_t index = 0; index < count; ++index) {
        double height = surface[index];
        for (int64_t offset = -extent; offset <= extent; ++offset) {
            auto neighbor = index + offset;
            if (neighbor < 0) neighbor += count;
            else if (neighbor >= count) neighbor -= count;
            height = std::max(height, surface[neighbor] - sag[size_t(offset + extent)]);
        }
        curve[index] = height;
    }
    return curve;
}

static void ValidateReference(const ReferenceParameters &parameters) {
    if (!std::isfinite(parameters.SampleRate) || parameters.SampleRate <= 0 || !std::isfinite(parameters.Mass) || parameters.Mass <= 0 || !std::isfinite(parameters.Stiffness) || parameters.Stiffness <= 0 || !std::isfinite(parameters.Dissipation) || parameters.Dissipation < 0 || !std::isfinite(parameters.Exponent) || parameters.Exponent < 1 || !std::isfinite(parameters.Gravity) || parameters.Gravity < 0 || !std::isfinite(parameters.Speed) || !std::isfinite(parameters.Spacing) || parameters.Spacing <= 0 || !parameters.Substeps || parameters.Substeps > 4096) throw std::invalid_argument("Invalid Conan reference parameters");
}

struct SurfaceSample {
    double Height{}, Velocity{};
};

static SurfaceSample SampleSurface(std::span<const double> curve, const ReferenceParameters &parameters, double distance) {
    double position = std::fmod(distance / parameters.Spacing, static_cast<double>(curve.size()));
    if (position < 0) position += curve.size();
    const auto index = static_cast<size_t>(position);
    const double difference = curve[(index + 1) % curve.size()] - curve[index];
    return {curve[index] + (position - index) * difference, parameters.Speed * difference / parameters.Spacing};
}

static void ReferenceStep(const ReferenceParameters &parameters, ReferenceState &state, std::span<const double> curve) {
    const double step = 1 / (parameters.SampleRate * parameters.Substeps);
    auto acceleration = [&](double position, double velocity, double distance) {
        const auto sample = SampleSurface(curve, parameters, distance);
        return parameters.Gravity - ContactForce(parameters, position + sample.Height, velocity + sample.Velocity) / parameters.Mass;
    };
    const double v1 = state.Velocity, a1 = acceleration(state.Position, v1, state.Distance);
    const double v2 = v1 + step * a1 / 2, a2 = acceleration(state.Position + step * v1 / 2, v2, state.Distance + parameters.Speed * step / 2);
    const double v3 = v1 + step * a2 / 2, a3 = acceleration(state.Position + step * v2 / 2, v3, state.Distance + parameters.Speed * step / 2);
    const double v4 = v1 + step * a3, a4 = acceleration(state.Position + step * v3, v4, state.Distance + parameters.Speed * step);
    state.Position += step * (v1 + 2 * v2 + 2 * v3 + v4) / 6;
    state.Velocity += step * (a1 + 2 * a2 + 2 * a3 + a4) / 6;
    state.Distance += parameters.Speed * step;
}

void RenderReference(const ReferenceParameters &parameters, ReferenceState &state, std::span<const double> rolling_curve, std::span<float> output) {
    ValidateReference(parameters);
    if (rolling_curve.size() < 2) throw std::invalid_argument("Rolling curve requires two or more samples");
    for (double height : rolling_curve)
        if (!std::isfinite(height)) throw std::invalid_argument("Nonfinite rolling curve");
    for (float &sample : output) {
        const auto surface = SampleSurface(rolling_curve, parameters, state.Distance);
        sample = static_cast<float>(ContactForce(parameters, state.Position + surface.Height, state.Velocity + surface.Velocity));
        for (unsigned substep = 0; substep < parameters.Substeps; ++substep) ReferenceStep(parameters, state, rolling_curve);
        if (!std::isfinite(state.Position) || !std::isfinite(state.Velocity)) throw std::runtime_error("Conan reference integration diverged; increase Substeps");
    }
}

ContactMeasurement MeasureContact(const ReferenceParameters &parameters, double input_velocity) {
    ValidateReference(parameters);
    if (!std::isfinite(input_velocity) || input_velocity <= 0) throw std::invalid_argument("Positive contact velocity required");
    auto contact_parameters = parameters;
    contact_parameters.Gravity = contact_parameters.Speed = 0;
    const double step = 1 / (parameters.SampleRate * parameters.Substeps);
    ReferenceState state{.Velocity = input_velocity};
    const double curve[]{0, 0};
    ContactMeasurement result{};
    for (uint64_t index = 0; index < static_cast<uint64_t>(10 / step); ++index) {
        const double previous_position = state.Position;
        ReferenceStep(contact_parameters, state, curve);
        result.PeakForce = std::max(result.PeakForce, ContactForce(contact_parameters, state.Position, state.Velocity));
        if (state.Position <= 0 && state.Velocity < 0) {
            result.Duration = step * (index + previous_position / (previous_position - state.Position));
            result.ExitVelocity = state.Velocity;
            return result;
        }
    }
    throw std::runtime_error("Contact did not separate within ten seconds");
}

} // namespace surface_audio::conan
