#include "Agarwal.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace surface_audio::agarwal {
namespace {
void Require(bool condition, const char *message) {
    if (!condition) throw std::invalid_argument(message);
}

bool Positive(double value) { return std::isfinite(value) && value > 0; }
bool Nonnegative(double value) { return std::isfinite(value) && value >= 0; }

void ValidateRolling(const RollingSettings &settings) {
    Require(Positive(settings.Radius) && Nonnegative(settings.Eccentricity) && settings.Eccentricity < settings.Radius && Nonnegative(settings.Stiffness) && Nonnegative(settings.Dissipation), "Invalid Agarwal rolling parameters");
}

struct AxisSample {
    double Height{}, Slope{}, Curvature{};
};

AxisSample SampleAxis(const SurfaceGrid &surface, uint32_t fixed, bool along_x, double position, double alpha, const ConstraintSettings &settings) {
    const uint32_t count = along_x ? surface.Width : surface.Height;
    const double spacing = along_x ? surface.SpacingX : surface.SpacingY;
    const auto height = [&](uint32_t i) { return surface.Heights[along_x ? size_t(fixed) * surface.Width + i : size_t(i) * surface.Width + fixed]; };
    const auto curvature = [&](uint32_t i) {
        const uint32_t center = std::clamp(i, 1u, count - 2);
        return ConstrainCurvature((height(center + 1) - 2 * height(center) + height(center - 1)) / (spacing * spacing), alpha);
    };
    const double half_width = settings.GaussianHalfWidth * alpha / settings.ReferenceAlpha;
    const auto smoothed = [&](uint32_t i) {
        if (half_width <= 0) return curvature(i);
        const auto radius = uint32_t(std::ceil(half_width));
        const double sigma = std::max(half_width * settings.GaussianSigmaRatio, 1e-12);
        double sum = 0, weight_sum = 0;
        const uint32_t begin = i > radius ? i - radius : 0, end = uint32_t(std::min(uint64_t(i) + radius, uint64_t(count - 1)));
        for (uint32_t j = begin; j <= end; ++j) {
            const double distance = (double(j) - i) / sigma, weight = std::exp(-0.5 * distance * distance);
            sum += weight * curvature(j);
            weight_sum += weight;
        }
        return sum / weight_sum;
    };
    double z = height(0), slope = along_x ? surface.LeftSlopes[fixed] : surface.TopSlopes[fixed], c0 = smoothed(0);
    const double grid_position = position / spacing;
    const auto end = std::min(uint32_t(grid_position), count - 2);
    for (uint32_t i = 0; i <= end; ++i) {
        const double c1 = smoothed(i + 1), distance = i == end ? position - double(i) * spacing : spacing;
        const double derivative = (c1 - c0) / spacing;
        z += slope * distance + 0.5 * c0 * distance * distance + derivative * distance * distance * distance / 6;
        slope += c0 * distance + 0.5 * derivative * distance * distance;
        if (i == end) return {z, slope, c0 + derivative * distance};
        c0 = c1;
    }
    return {};
}

Trajectory SampleUnchecked(const SurfaceGrid &surface, const Motion &motion, const ConstraintSettings &settings) {
    Require(std::isfinite(motion.X) && std::isfinite(motion.Y) && motion.X >= 0 && motion.Y >= 0 && motion.X <= double(surface.Width - 1) * surface.SpacingX && motion.Y <= double(surface.Height - 1) * surface.SpacingY, "Agarwal trajectory leaves the finite surface grid");
    const double alpha = Alpha(motion.NormalForce, settings);
    const double gx = motion.X / surface.SpacingX, gy = motion.Y / surface.SpacingY;
    const auto x0 = std::min(uint32_t(gx), surface.Width - 2), y0 = std::min(uint32_t(gy), surface.Height - 2);
    const auto sx0 = SampleAxis(surface, y0, true, motion.X, alpha, settings), sx1 = SampleAxis(surface, y0 + 1, true, motion.X, alpha, settings);
    const auto sy0 = SampleAxis(surface, x0, false, motion.Y, alpha, settings), sy1 = SampleAxis(surface, x0 + 1, false, motion.Y, alpha, settings);
    return {std::lerp(sx0.Height, sx1.Height, gy - y0), std::lerp(sx0.Slope, sx1.Slope, gy - y0), std::lerp(sy0.Slope, sy1.Slope, gx - x0), std::lerp(sx0.Curvature, sx1.Curvature, gy - y0), std::lerp(sy0.Curvature, sy1.Curvature, gx - x0)};
}

double LogMix(double a, double b, double morph) {
    if (morph == 0) return a;
    if (morph == 1) return b;
    return std::exp(std::lerp(std::log(a), std::log(b), morph));
}

void BuildUnchecked(const ImpulseResponses &responses, double morph, std::span<float> output) {
    Require(std::isfinite(morph) && morph >= 0 && morph <= 1, "Agarwal IR morph must be in [0, 1]");
    std::fill(output.begin(), output.end(), 0.f);
    for (size_t mode = 0; mode < responses.Surface0.Frequencies.size(); ++mode) {
        const double frequency = LogMix(responses.Surface0.Frequencies[mode], responses.Surface1.Frequencies[mode], morph);
        const double omega = 2 * std::numbers::pi * frequency / responses.SampleRate;
        for (uint32_t lag = 0; lag < responses.TapCount; ++lag) {
            const size_t index = mode * responses.TapCount + lag;
            output[lag] += float(LogMix(responses.Surface0.Amplitudes[index], responses.Surface1.Amplitudes[index], morph) * std::sin(omega * lag));
        }
    }
    for (size_t mode = 0; mode < responses.Object.Frequencies.size(); ++mode) {
        const double omega = 2 * std::numbers::pi * responses.Object.Frequencies[mode] / responses.SampleRate;
        for (uint32_t lag = 0; lag < responses.TapCount; ++lag) output[lag] += float(responses.ObjectGain * responses.Object.Amplitudes[mode * responses.TapCount + lag] * std::sin(omega * lag));
    }
}
} // namespace

double Alpha(double normal_force, const ConstraintSettings &settings) {
    Require(Nonnegative(normal_force), "Agarwal normal force must be finite and nonnegative");
    if (settings.NormalMin == settings.NormalMax) return settings.ConstantAlpha;
    const double weight = std::pow(std::clamp((normal_force - settings.NormalMin) / (settings.NormalMax - settings.NormalMin), 0., 1.), settings.Exponent);
    return std::lerp(settings.AlphaMax, settings.AlphaMin, weight);
}

double ConstrainCurvature(double curvature, double alpha) {
    return alpha == 0 ? curvature : std::tanh(alpha * curvature) / alpha;
}

NormalRange ScrapingNormalRange(double mass, double angular_frequency, double half_length, double angle, double friction, double gravity) {
    Require(Positive(mass) && Nonnegative(angular_frequency) && Nonnegative(half_length) && Nonnegative(angle) && angle < std::numbers::pi / 2 && Nonnegative(friction) && Positive(gravity), "Invalid Agarwal scraping normal-force parameters");
    const double tangent = std::tan(angle), denominator = 1 - friction * tangent;
    Require(denominator > 0, "Agarwal scraping normal force is singular or tensile");
    const double mean = mass * gravity / denominator, amplitude = std::abs(mass * angular_frequency * angular_frequency * half_length * tangent / denominator);
    Require(mean >= amplitude, "Agarwal harmonic motion loses contact");
    return {mean - amplitude, mean + amplitude};
}

double ScrapingNormalForce(double position, double mass, double angular_frequency, double angle, double friction, double gravity) {
    const auto range = ScrapingNormalRange(mass, angular_frequency, std::abs(position), angle, friction, gravity);
    const double mean = 0.5 * (range.Min + range.Max);
    return mean - mass * angular_frequency * angular_frequency * position * std::tan(angle) / (1 - friction * std::tan(angle));
}

double RollingPosition(double angle, const RollingSettings &settings) {
    ValidateRolling(settings);
    Require(std::isfinite(angle), "Agarwal rolling angle must be finite");
    return settings.Radius * angle - settings.Eccentricity * std::sin(angle);
}
double RollingVelocity(double angle, double angular_velocity, const RollingSettings &settings) {
    ValidateRolling(settings);
    Require(std::isfinite(angle) && std::isfinite(angular_velocity), "Agarwal rolling kinematics must be finite");
    return (settings.Radius - settings.Eccentricity * std::cos(angle)) * angular_velocity;
}

double RollingNormalForce(double angle, double angular_velocity, double angular_acceleration, double mass, const RollingSettings &settings, double gravity) {
    ValidateRolling(settings);
    Require(std::isfinite(angle) && std::isfinite(angular_velocity) && std::isfinite(angular_acceleration) && Positive(mass) && Positive(gravity), "Invalid Agarwal rolling normal-force parameters");
    const double acceleration = settings.Eccentricity * (std::cos(angle) * angular_velocity * angular_velocity + std::sin(angle) * angular_acceleration);
    const double normal = mass * (gravity + acceleration);
    Require(Nonnegative(normal), "Agarwal eccentric rolling motion loses contact");
    return normal;
}

void Validate(const SurfaceGrid &surface, const ConstraintSettings &settings) {
    Require(surface.Width >= 3 && surface.Height >= 3 && surface.Heights.size() == size_t(surface.Width) * surface.Height && Positive(surface.SpacingX) && Positive(surface.SpacingY), "Invalid Agarwal surface grid");
    Require(std::ranges::all_of(surface.Heights, [](double x) { return std::isfinite(x); }), "Agarwal surface heights must be finite");
    Require(surface.LeftSlopes.size() == surface.Height && surface.TopSlopes.size() == surface.Width && std::ranges::all_of(surface.LeftSlopes, [](double x) { return std::isfinite(x); }) && std::ranges::all_of(surface.TopSlopes, [](double x) { return std::isfinite(x); }), "Agarwal surface requires finite explicit boundary slopes");
    Require(Nonnegative(settings.NormalMin) && std::isfinite(settings.NormalMax) && settings.NormalMax >= settings.NormalMin && Nonnegative(settings.AlphaMin) && std::isfinite(settings.AlphaMax) && settings.AlphaMax >= settings.AlphaMin && Nonnegative(settings.ConstantAlpha) && Positive(settings.Exponent) && Positive(settings.ReferenceAlpha) && Nonnegative(settings.GaussianHalfWidth) && Positive(settings.GaussianSigmaRatio), "Invalid Agarwal trajectory constraints");
    Require(settings.GaussianHalfWidth * std::max(settings.AlphaMax, settings.ConstantAlpha) / settings.ReferenceAlpha < double(UINT32_MAX / 2), "Agarwal Gaussian window is too large");
}

Trajectory SampleTrajectory(const SurfaceGrid &surface, const Motion &motion, const ConstraintSettings &settings) {
    Validate(surface, settings);
    return SampleUnchecked(surface, motion, settings);
}

void PrepareTrajectory(const SurfaceGrid &surface, std::span<const Motion> motion, const ConstraintSettings &settings, std::span<Trajectory> trajectory) {
    Validate(surface, settings);
    Require(trajectory.size() == motion.size(), "Agarwal trajectory and motion sizes differ");
    for (size_t i = 0; i < motion.size(); ++i) trajectory[i] = SampleUnchecked(surface, motion[i], settings);
}

Force ScrapingForce(const Trajectory &trajectory, const Motion &motion, const ScrapingSettings &settings) {
    Require(Positive(settings.Mass) && Nonnegative(settings.Beta1) && Positive(settings.Beta2), "Invalid Agarwal scraping force parameters");
    Require(Nonnegative(motion.NormalForce) && std::isfinite(motion.VelocityX) && std::isfinite(motion.VelocityY), "Invalid Agarwal motion");
    Require(std::isfinite(trajectory.Height) && std::isfinite(trajectory.SlopeX) && std::isfinite(trajectory.SlopeY) && std::isfinite(trajectory.CurvatureX) && std::isfinite(trajectory.CurvatureY), "Agarwal trajectory samples must be finite");
    if (motion.NormalForce == 0) return {};
    const double slope_speed = motion.VelocityX * trajectory.SlopeX + motion.VelocityY * trajectory.SlopeY;
    return {settings.Beta1 * std::pow(std::abs(slope_speed), settings.Beta2), settings.Mass * (trajectory.CurvatureX * motion.VelocityX * motion.VelocityX + trajectory.CurvatureY * motion.VelocityY * motion.VelocityY), 0};
}

Force RollingForce(const Trajectory &trajectory, const Motion &motion, const ScrapingSettings &scraping, const RollingSettings &rolling) {
    ValidateRolling(rolling);
    Require(motion.VelocityY == 0, "Agarwal rolling equations require an x-axis path");
    Require(std::isfinite(motion.X), "Agarwal rolling position must be finite");
    auto force = ScrapingForce(trajectory, motion, scraping);
    if (motion.NormalForce == 0) return force;
    const double phase = motion.X / rolling.Radius;
    const double penetration = rolling.Radius - rolling.Eccentricity * std::cos(phase) + trajectory.Height;
    Require(Nonnegative(penetration), "Agarwal Eq16 penetration must be nonnegative; check the height datum");
    const double velocity = rolling.Eccentricity / rolling.Radius * motion.VelocityX * std::sin(phase) + motion.VelocityX * trajectory.SlopeX;
    force.Rolling = penetration * std::sqrt(penetration) * (rolling.Stiffness + rolling.Dissipation * velocity);
    return force;
}

void GenerateForces(std::span<const Trajectory> trajectory, std::span<const Motion> motion, const ScrapingSettings &scraping, const RollingSettings *rolling, std::span<float> output) {
    Require(trajectory.size() == motion.size() && output.size() == motion.size(), "Agarwal force buffer sizes differ");
    for (size_t i = 0; i < motion.size(); ++i) {
        const auto force = rolling ? RollingForce(trajectory[i], motion[i], scraping, *rolling) : ScrapingForce(trajectory[i], motion[i], scraping);
        output[i] = float(force.Horizontal + force.Vertical + force.Rolling);
    }
}

void Validate(const ImpulseResponses &responses) {
    Require(Positive(responses.SampleRate) && responses.TapCount > 0 && std::isfinite(responses.ObjectGain), "Invalid Agarwal IR parameters");
    Require(responses.Surface0.Frequencies.size() == responses.Surface1.Frequencies.size(), "Agarwal surface modes must be matched between endpoints");
    for (const auto &endpoint : {responses.Surface0, responses.Surface1, responses.Object}) {
        Require(endpoint.Amplitudes.size() == endpoint.Frequencies.size() * responses.TapCount, "Agarwal modal envelope sizes differ");
        Require(std::ranges::all_of(endpoint.Frequencies, [&](double f) { return Positive(f) && f < responses.SampleRate / 2; }), "Agarwal mode frequencies must lie below Nyquist");
        Require(std::ranges::all_of(endpoint.Amplitudes, Positive), "Agarwal log-morphed modal envelopes must be strictly positive");
    }
}

void BuildImpulseResponse(const ImpulseResponses &responses, double morph, std::span<float> output) {
    Validate(responses);
    Require(output.size() == responses.TapCount, "Agarwal IR output size differs");
    BuildUnchecked(responses, morph, output);
}

void BuildImpulseResponseBlock(const ImpulseResponses &responses, std::span<const float> morph, std::span<float> coefficients) {
    Validate(responses);
    Require(coefficients.size() == morph.size() * responses.TapCount, "Agarwal IR coefficient buffer size differs");
    for (size_t frame = 0; frame < morph.size(); ++frame) BuildUnchecked(responses, morph[frame], coefficients.subspan(frame * responses.TapCount, responses.TapCount));
}

std::vector<double> ExponentialEnvelopes(std::span<const double> amplitude, std::span<const double> decay_seconds, uint32_t tap_count, double sample_rate) {
    Require(amplitude.size() == decay_seconds.size() && Positive(sample_rate) && tap_count > 0, "Invalid Agarwal exponential envelope dimensions");
    std::vector<double> result(amplitude.size() * tap_count);
    for (size_t mode = 0; mode < amplitude.size(); ++mode) {
        Require(Positive(amplitude[mode]) && Positive(decay_seconds[mode]), "Invalid Agarwal exponential envelope parameters");
        for (uint32_t lag = 0; lag < tap_count; ++lag) result[mode * tap_count + lag] = amplitude[mode] * std::exp(-double(lag) / (sample_rate * decay_seconds[mode]));
    }
    return result;
}

} // namespace surface_audio::agarwal
