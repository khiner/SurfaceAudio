#include "agarwal/Agarwal.h"
#include "agarwal/AgarwalGpu.h"
#include "core/Convolution.h"
#include "core/Gpu.h"
#include "core/GpuDecimate.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <numbers>
#include <stdexcept>

using namespace surface_audio;
using namespace surface_audio::agarwal;

namespace {
void Check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void Near(double actual, double expected, double tolerance, const char *message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::cerr << message << ": actual=" << actual << ", expected=" << expected << '\n';
        throw std::runtime_error(message);
    }
}

void TestTrajectory() {
    constexpr uint32_t width = 17, height = 9;
    constexpr double spacing = 0.02;
    std::vector<double> heights(width * height);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const double xx = x * spacing, yy = y * spacing;
            heights[y * width + x] = 0.3 + 2 * xx + 3 * yy + 7 * xx * xx + 11 * yy * yy;
        }
    }
    const std::vector<double> left_slopes(height, 2), top_slopes(width, 3);
    const SurfaceGrid surface{width, height, spacing, spacing, heights, left_slopes, top_slopes};
    ConstraintSettings constraints{.NormalMin = 1, .NormalMax = 1, .ConstantAlpha = 0, .GaussianHalfWidth = 0};
    const Motion motion{.X = 0.131, .Y = 0.08, .NormalForce = 1};
    auto trajectory = SampleTrajectory(surface, motion, constraints);
    Near(trajectory.Height, 0.3 + 2 * motion.X + 3 * motion.Y + 7 * motion.X * motion.X + 11 * motion.Y * motion.Y, 1e-12, "Unconstrained quadratic height");
    Near(trajectory.SlopeX, 2 + 14 * motion.X, 1e-11, "Unconstrained x derivative");
    Near(trajectory.SlopeY, 3 + 22 * motion.Y, 1e-11, "Unconstrained y derivative");
    Near(trajectory.CurvatureX, 14, 1e-10, "Unconstrained x curvature");
    Near(trajectory.CurvatureY, 22, 1e-10, "Unconstrained y curvature");
    constraints.ConstantAlpha = 0.1;
    constraints.GaussianHalfWidth = 5;
    trajectory = SampleTrajectory(surface, motion, constraints);
    const double cx = 10 * std::tanh(1.4), cy = 10 * std::tanh(2.2);
    Near(trajectory.CurvatureX, cx, 1e-10, "Gaussian preserves constant saturated curvature");
    Near(trajectory.CurvatureY, cy, 1e-10, "Y Gaussian preserves constant saturated curvature");
    Near(trajectory.SlopeX, 2 + cx * motion.X, 1e-11, "Integrated saturated curvature");
    Check(std::abs(trajectory.CurvatureX) < 1 / constraints.ConstantAlpha, "Curvature bound");

    std::array<Motion, 19> motions;
    for (size_t i = 0; i < motions.size(); ++i) motions[i] = {.X = 0.05 + 0.005 * i, .Y = 0.04, .VelocityX = 0.1, .NormalForce = 1};
    std::array<Trajectory, 19> whole{}, split{};
    PrepareTrajectory(surface, motions, constraints, whole);
    PrepareTrajectory(surface, std::span(motions).first(7), constraints, std::span(split).first(7));
    PrepareTrajectory(surface, std::span(motions).subspan(7), constraints, std::span(split).subspan(7));
    for (size_t i = 0; i < whole.size(); ++i) {
        Near(split[i].Height, whole[i].Height, 0, "Trajectory block continuity");
        Near(split[i].CurvatureX, whole[i].CurvatureX, 0, "Curvature block continuity");
    }
    bool rejected = false;
    try {
        SampleTrajectory(surface, {.X = -0.1}, constraints);
    } catch (const std::invalid_argument &) { rejected = true; }
    Check(rejected, "Out-of-grid path must be rejected");

    // A discrete curvature impulse has an independently known normalized Gaussian response.
    std::vector<double> impulse_heights(33 * 3, 0.);
    constexpr uint32_t center = 16;
    for (uint32_t row = 0; row < 3; ++row) {
        for (uint32_t x = 1; x + 1 < 33; ++x) impulse_heights[row * 33 + x + 1] = 2 * impulse_heights[row * 33 + x] - impulse_heights[row * 33 + x - 1] + (x == center ? 10. : 0.);
    }
    const std::vector<double> impulse_left(3, 0), impulse_top(33, 0);
    const SurfaceGrid impulse_surface{33, 3, 1, 1, impulse_heights, impulse_left, impulse_top};
    const ConstraintSettings impulse_settings{.NormalMin = 1, .NormalMax = 1, .ConstantAlpha = 0.1, .ReferenceAlpha = 0.1, .GaussianHalfWidth = 2, .GaussianSigmaRatio = 0.5};
    const double normalizer = 1 + 2 * std::exp(-0.5) + 2 * std::exp(-2.);
    for (int offset = -3; offset <= 3; ++offset) {
        const auto sample = SampleTrajectory(impulse_surface, {.X = double(center) + offset, .Y = 1, .NormalForce = 1}, impulse_settings);
        const double expected = std::abs(offset) <= 2 ? 10 * std::tanh(1.) * std::exp(-0.5 * offset * offset) / normalizer : 0;
        Near(sample.CurvatureX, expected, 1e-13, "Gaussian filtering follows curvature nonlinearity");
    }
}

void TestForces() {
    ConstraintSettings constraints;
    Near(Alpha(constraints.NormalMin, constraints), 0.05, 0, "Minimum normal maximum softening");
    Near(Alpha(constraints.NormalMax, constraints), 0.01, 1e-17, "Maximum normal minimum softening");
    const auto normal = ScrapingNormalRange(0.2, 3, 0.1, 0.3, 0.2);
    Near(ScrapingNormalForce(-0.1, 0.2, 3, 0.3, 0.2), normal.Max, 1e-14, "Near torso normal maximum");
    Near(ScrapingNormalForce(0.1, 0.2, 3, 0.3, 0.2), normal.Min, 1e-14, "Far torso normal minimum");
    const Trajectory trajectory{.Height = 0.002, .SlopeX = -2, .SlopeY = 3, .CurvatureX = 7, .CurvatureY = -11};
    const Motion motion{.X = 0.012, .VelocityX = -0.4, .VelocityY = 0.3, .NormalForce = 1};
    const ScrapingSettings scraping{.Mass = 0.2, .Beta1 = 0.05, .Beta2 = 1};
    auto force = ScrapingForce(trajectory, motion, scraping);
    Near(force.Horizontal, 0.085, 1e-15, "Paper Eq3 absolute slope speed");
    Near(force.Vertical, 0.026, 1e-15, "Paper Eq2 diagonal curvature acceleration");
    auto released = motion;
    released.NormalForce = 0;
    force = ScrapingForce(trajectory, released, scraping);
    Near(force.Horizontal + force.Vertical, 0, 0, "Released scrape has no force");

    const RollingSettings rolling{.Radius = 0.02, .Eccentricity = 0.001, .Stiffness = 400, .Dissipation = 3};
    constexpr double angle = 1.2, angular_velocity = 4, epsilon = 1e-6;
    const double numerical_velocity = (RollingPosition(angle + epsilon * angular_velocity, rolling) - RollingPosition(angle - epsilon * angular_velocity, rolling)) / (2 * epsilon);
    Near(RollingVelocity(angle, angular_velocity, rolling), numerical_velocity, 1e-11, "Eq18 rolling position finite difference");
    const auto vertical_position = [&](double theta) { return rolling.Radius - rolling.Eccentricity * std::cos(theta); };
    constexpr double acceleration_dt = 1e-4;
    const double acceleration = (vertical_position(angle + angular_velocity * acceleration_dt) - 2 * vertical_position(angle) + vertical_position(angle - angular_velocity * acceleration_dt)) / (acceleration_dt * acceleration_dt);
    Near(RollingNormalForce(angle, angular_velocity, 0, 0.2, rolling), 0.2 * (9.81 + acceleration), 2e-10, "Rolling normal follows eccentric center acceleration");
    auto rolling_motion = motion;
    rolling_motion.VelocityY = 0;
    force = RollingForce(trajectory, rolling_motion, scraping, rolling);
    const double rho = 0.02 - 0.001 * std::cos(0.6) + 0.002;
    const double rho_dot = 0.001 / 0.02 * -0.4 * std::sin(0.6) + -0.4 * -2;
    Near(force.Rolling, std::pow(rho, 1.5) * (400 + 3 * rho_dot), 1e-15, "Paper Eqs15-17 rolling impact");
}

void TestImpulseResponses() {
    constexpr uint32_t taps = 257, frames = 37;
    const std::array<double, 1> f0{100}, f1{400}, object_frequency{250};
    std::vector<double> a0(taps), a1(taps), object(taps, 0.2);
    for (uint32_t lag = 0; lag < taps; ++lag) {
        a0[lag] = 0.3 + 0.001 * lag;
        a1[lag] = 1.2 + 0.002 * lag;
    }
    const ImpulseResponses responses{.SampleRate = 4000, .ObjectGain = 0.7, .TapCount = taps, .Surface0 = {f0, a0}, .Surface1 = {f1, a1}, .Object = {object_frequency, object}};
    std::array<float, taps> ir{};
    BuildImpulseResponse(responses, 0.5, ir);
    for (uint32_t lag = 0; lag < taps; ++lag) {
        const double time = double(lag) / 4000;
        const double expected = std::sqrt(a0[lag] * a1[lag]) * std::sin(2 * std::numbers::pi * 200 * time) + 0.14 * std::sin(2 * std::numbers::pi * 250 * time);
        Near(ir[lag], expected, 2e-7, "Geometric frequency and measured amplitude morph");
    }
    std::array<float, frames> morph{};
    for (uint32_t i = 0; i < frames; ++i) morph[i] = float(i) / (frames - 1);
    std::vector<float> coefficients(frames * taps), split(coefficients.size());
    BuildImpulseResponseBlock(responses, morph, coefficients);
    BuildImpulseResponseBlock(responses, std::span(morph).first(3), std::span(split).first(3 * taps));
    BuildImpulseResponseBlock(responses, std::span(morph).subspan(3), std::span(split).subspan(3 * taps));
    Check(coefficients == split, "IR block continuity");
    std::vector<float> excitation(taps - 1 + frames, 0.f);
    excitation[taps - 1] = 1;
    std::array<float, frames> output{};
    Convolve({taps, frames}, coefficients, excitation, output);
    for (uint32_t frame = 0; frame < frames; ++frame) Near(output[frame], coefficients[frame * taps + frame], 1e-7, "Impulse tail uses current output location Eq13");
    Check(std::abs(output.back() - coefficients[frames - 1]) > 0.01, "Location morph must affect earlier excitations");
    std::array<float, frames> blocked{};
    Convolve({taps, 3}, std::span(coefficients).first(3 * taps), std::span(excitation).first(taps - 1 + 3), std::span(blocked).first(3));
    Convolve({taps, frames - 3}, std::span(coefficients).subspan(3 * taps), std::span(excitation).subspan(3), std::span(blocked).subspan(3));
    Check(output == blocked, "Convolution force history block continuity");
    const auto gpu = PrepareGpuImpulseData(responses, frames);
    Check(gpu.Parameters.SurfaceModeCount == 1 && gpu.Parameters.ObjectModeCount == 1 && gpu.Frequencies.size() == 3 && gpu.Amplitudes.size() == 3 * taps, "GPU SoA packing");

    auto device = CreateGpu();
    const auto build_kernel = CreateKernel(device, "AgarwalBuildImpulseResponses"), convolve_kernel = CreateKernel(device, "FirConvolve");
    const auto parameters = Upload(device, gpu.Parameters), frequencies = Upload<float>(device, gpu.Frequencies), amplitudes = Upload<float>(device, gpu.Amplitudes), morphs = Upload<float>(device, morph);
    const auto gpu_coefficients = CreateBuffer(device, coefficients.size() * sizeof(float));
    for (size_t i = 0; i < excitation.size(); ++i) excitation[i] = float(0.5 * std::sin(0.17 * i) + 0.3 * std::cos(0.37 * i));
    Convolve({taps, frames}, coefficients, excitation, output);
    const auto fir_parameters = Upload(device, FirBlock{taps, frames}), gpu_excitation = Upload<float>(device, excitation), gpu_output = CreateBuffer(device, sizeof(output));
    const std::array<GpuBinding, 5> build_bindings{{{parameters, 0}, {frequencies, 1}, {amplitudes, 2}, {morphs, 3}, {gpu_coefficients, 4}}};
    const std::array<GpuBinding, 4> fir_bindings{{{fir_parameters, 0}, {gpu_coefficients, 1}, {gpu_excitation, 2}, {gpu_output, 3}}};
    BeginGpu(device);
    DispatchGpu(device, build_kernel, build_bindings, {frames * taps});
    DispatchGroupsGpu(device, convolve_kernel, fir_bindings, {frames}, {64});
    SubmitGpu(device);
    WaitGpu(device);
    const auto actual_coefficients = BufferSpan<const float>(gpu_coefficients), actual_output = BufferSpan<const float>(gpu_output);
    double coefficient_error = 0, output_error = 0;
    for (size_t i = 0; i < coefficients.size(); ++i) coefficient_error = std::max(coefficient_error, std::abs(double(actual_coefficients[i]) - coefficients[i]));
    for (size_t i = 0; i < output.size(); ++i) output_error = std::max(output_error, std::abs(double(actual_output[i]) - output[i]));
    std::cout << DeviceName(device) << ": IR maximum error=" << coefficient_error << ", audio maximum error=" << output_error << '\n';
    Check(coefficient_error < 1e-4, "Metal output-location IR coefficient agreement");
    Check(output_error < 2e-4, "Metal IR generation and FIR convolution end-to-end agreement");
}

void TestGpuTrajectory() {
    constexpr uint32_t width = 33, height = 11, frames = 67;
    std::vector<double> heights(width * height);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) heights[y * width + x] = 3e-5 * std::sin(0.47 * x) + 2e-5 * std::cos(0.71 * y) + 1e-5 * std::sin(0.13 * x * y);
    }
    std::vector<double> left_slopes(height), top_slopes(width);
    for (uint32_t y = 0; y < height; ++y) left_slopes[y] = (3e-5 * 0.47 + 1e-5 * 0.13 * y) / 0.0003;
    for (uint32_t x = 0; x < width; ++x) top_slopes[x] = 1e-5 * 0.13 * x / 0.0005;
    const SurfaceGrid surface{width, height, 0.0003, 0.0005, heights, left_slopes, top_slopes};
    std::vector<Motion> motion(frames);
    for (uint32_t i = 0; i < frames; ++i) motion[i] = {.X = double(width - 1) * surface.SpacingX * i / (frames - 1), .Y = double(height - 1) * surface.SpacingY * i / (frames - 1), .VelocityX = 0.03 + 0.04 * std::sin(0.1 * i), .VelocityY = 0.04 * std::cos(0.2 * i), .NormalForce = i ? 1. + 9. * i / (frames - 1) : 0};
    const ScrapingSettings scraping{.Mass = 0.1, .Beta1 = 0.05, .Beta2 = 1.1};
    const RollingSettings rolling{.Radius = 0.01, .Eccentricity = 0.0002, .Stiffness = 500, .Dissipation = 5};
    auto gpu = CreateGpu();
    const auto kernel = CreateKernel(gpu, "AgarwalPrepareForces");
    struct Result {
        std::vector<float> Trajectory, Force;
        std::vector<uint32_t> Status;
    };
    const auto run = [&](const GpuTrajectoryData &data) {
        const auto parameters = Upload(gpu, data.Parameters), gpu_heights = Upload<float>(gpu, data.Heights), gpu_motion = Upload<float>(gpu, data.Motions);
        const auto trajectory = CreateBuffer(gpu, 5 * data.Parameters.FrameCount * sizeof(float)), force = CreateBuffer(gpu, data.Parameters.FrameCount * sizeof(float)), status = CreateBuffer(gpu, data.Parameters.FrameCount * sizeof(uint32_t));
        const auto boundary_slopes = Upload<float>(gpu, data.BoundarySlopes);
        const std::array<GpuBinding, 7> bindings{{{parameters, 0}, {gpu_heights, 1}, {gpu_motion, 2}, {trajectory, 3}, {force, 4}, {status, 5}, {boundary_slopes, 6}}};
        BeginGpu(gpu);
        DispatchGpu(gpu, kernel, bindings, {data.Parameters.FrameCount});
        SubmitGpu(gpu);
        WaitGpu(gpu);
        const auto trajectory_span = BufferSpan<const float>(trajectory), force_span = BufferSpan<const float>(force);
        const auto status_span = BufferSpan<const uint32_t>(status);
        return Result{{trajectory_span.begin(), trajectory_span.end()}, {force_span.begin(), force_span.end()}, {status_span.begin(), status_span.end()}};
    };
    for (uint32_t variant = 0; variant < 3; ++variant) {
        ConstraintSettings constraints;
        if (variant == 1) constraints = {.NormalMin = 1, .NormalMax = 1, .ConstantAlpha = 0, .GaussianHalfWidth = 0};
        if (variant == 2)
            for (auto &sample : motion) sample.VelocityY = 0;
        const auto *ball = variant == 2 ? &rolling : nullptr;
        std::vector<Trajectory> expected(frames);
        std::vector<float> expected_force(frames);
        PrepareTrajectory(surface, motion, constraints, expected);
        GenerateForces(expected, motion, scraping, ball, expected_force);
        const auto actual = run(PrepareGpuTrajectoryData(surface, motion, constraints, scraping, ball));
        ValidateGpuTrajectoryStatus(actual.Status);
        std::array<double, 5> maximum_error{};
        double force_error = 0;
        for (uint32_t i = 0; i < frames; ++i) {
            const std::array<double, 5> reference{expected[i].Height, expected[i].SlopeX, expected[i].SlopeY, expected[i].CurvatureX, expected[i].CurvatureY};
            for (uint32_t plane = 0; plane < 5; ++plane) maximum_error[plane] = std::max(maximum_error[plane], std::abs(actual.Trajectory[plane * frames + i] - reference[plane]));
            force_error = std::max(force_error, std::abs(double(actual.Force[i]) - expected_force[i]));
        }
        std::cout << "Agarwal GPU trajectory variant " << variant << ": errors=";
        for (double error : maximum_error) std::cout << ' ' << error;
        std::cout << ", force=" << force_error << '\n';
        Check(maximum_error[0] < 2e-9 && maximum_error[1] < 2e-6 && maximum_error[2] < 2e-6 && maximum_error[3] < 0.003 && maximum_error[4] < 0.003, "Metal trajectory agrees with FP64 texture preparation");
        Check(force_error < 2e-6, "Metal scraping and rolling force agrees with FP64 preparation");
        const auto first = run(PrepareGpuTrajectoryData(surface, std::span(motion).first(23), constraints, scraping, ball));
        const auto second = run(PrepareGpuTrajectoryData(surface, std::span(motion).subspan(23), constraints, scraping, ball));
        ValidateGpuTrajectoryStatus(first.Status);
        ValidateGpuTrajectoryStatus(second.Status);
        for (uint32_t i = 0; i < frames; ++i) Near(i < 23 ? first.Force[i] : second.Force[i - 23], actual.Force[i], 0, "GPU force block continuity");
    }
    std::ranges::fill(heights, -0.02);
    const auto invalid = run(PrepareGpuTrajectoryData(surface, std::span(motion).first(1), {}, scraping, &rolling));
    // The first frame is released, so negative penetration does not produce contact force.
    ValidateGpuTrajectoryStatus(invalid.Status);
    motion[0].NormalForce = 1;
    const auto tensile = run(PrepareGpuTrajectoryData(surface, std::span(motion).first(1), {}, scraping, &rolling));
    bool rejected = false;
    try {
        ValidateGpuTrajectoryStatus(tensile.Status);
    } catch (const std::runtime_error &) { rejected = true; }
    Check(rejected && tensile.Status[0] == 1, "GPU invalid rolling penetration is reported to the caller");
}

double Integrate(auto function, double begin, double end, uint32_t intervals) {
    const double step = (end - begin) / intervals;
    double sum = function(begin) + function(end);
    for (uint32_t i = 1; i < intervals; ++i) sum += (i % 2 ? 4 : 2) * function(begin + i * step);
    return sum * step / 3;
}

void TestGridConvergence() {
    constexpr double length = 0.01, position = 0.0067, half_width = 0.000625, sigma = 0.4 * half_width;
    const auto curvature = [](double x) { return std::tanh(0.03 * 12e5 * x * x) / 0.03; };
    const auto smoothed = [&](double x) {
        const double begin = std::max(0., x - half_width), end = std::min(length, x + half_width);
        const auto weight = [&](double s) { return std::exp(-0.5 * std::pow((s - x) / sigma, 2)); };
        return Integrate([&](double s) { return weight(s) * curvature(s); }, begin, end, 256) / Integrate(weight, begin, end, 256);
    };
    const std::array reference{Integrate([&](double x) { return (position - x) * smoothed(x); }, 0, position, 1024), Integrate(smoothed, 0, position, 1024), smoothed(position)};
    double previous_error = 1;
    for (uint32_t width : {33u, 65u, 129u, 257u}) {
        const double spacing = length / (width - 1);
        std::vector<double> heights(3 * width), left_slopes(3, 0), top_slopes(width, 0);
        for (uint32_t row = 0; row < 3; ++row)
            for (uint32_t x = 0; x < width; ++x) heights[row * width + x] = 1e5 * std::pow(x * spacing, 4);
        const SurfaceGrid surface{width, 3, spacing, spacing, heights, left_slopes, top_slopes};
        const ConstraintSettings constraints{.NormalMin = 1, .NormalMax = 1, .ConstantAlpha = 0.03, .ReferenceAlpha = 0.03, .GaussianHalfWidth = half_width / spacing};
        const auto sample = SampleTrajectory(surface, {.X = position, .Y = spacing, .NormalForce = 1}, constraints);
        const std::array actual{sample.Height, sample.SlopeX, sample.CurvatureX};
        double error = 0;
        for (uint32_t i = 0; i < 3; ++i) error = std::max(error, std::abs((actual[i] - reference[i]) / reference[i]));
        std::cout << "Agarwal fixed-width Gaussian grid " << width << ": relative error=" << error << '\n';
        Check(error < previous_error * 0.8, "Texture refinement converges toward independent continuous quadrature");
        Check(sample.Height >= 0 && sample.Height <= 0.5 * position * position / constraints.ConstantAlpha, "Anchored convex trajectory respects its integrated curvature bound");
        previous_error = error;
    }
    Check(previous_error < 0.001, "Refined constrained surface agrees with continuous quadrature");
}

void TestTemporalConvergence() {
    constexpr double duration = 0.1, decay = 25, a = 2 * std::numbers::pi * 73, b = 2 * std::numbers::pi * 137;
    const auto exponential_integral = [&](double frequency) { const std::complex<double> exponent{-decay, frequency}; return (std::exp(exponent * duration) - 1.) / exponent; };
    const double reference = 0.5 * std::real(std::exp(std::complex<double>{0, -a * duration}) * exponential_integral(a + b) - std::exp(std::complex<double>{0, a * duration}) * exponential_integral(b - a));
    double previous_error = 1;
    for (uint32_t sample_rate : {1000u, 2000u, 4000u}) {
        const uint32_t taps = uint32_t(duration * sample_rate) + 1;
        const std::array frequency{137.}, amplitude{1. / sample_rate}, decay_seconds{1. / decay};
        const auto envelope = ExponentialEnvelopes(amplitude, decay_seconds, taps, sample_rate);
        const ImpulseResponses responses{.SampleRate = double(sample_rate), .TapCount = taps, .Surface0 = {frequency, envelope}, .Surface1 = {frequency, envelope}, .Object = {}};
        std::vector<float> coefficients(taps), excitation(taps);
        for (uint32_t i = 0; i < taps; ++i) excitation[i] = float(std::sin(a * i / sample_rate));
        BuildImpulseResponse(responses, 0, coefficients);
        std::array<float, 1> output{};
        Convolve({taps, 1}, coefficients, excitation, output);
        const double error = std::abs(output[0] - reference);
        std::cout << "Agarwal continuous-IR sampling " << sample_rate << ": error=" << error << '\n';
        Check(error < previous_error * 0.3, "Sampled continuous convolution converges quadratically");
        previous_error = error;
    }
    Check(previous_error < 5e-6, "Sample-rate-scaled IR agrees with analytic continuous convolution");
}

void TestPhysicalScope() {
    const ScrapingSettings scraping;
    const RollingSettings rolling{.Radius = 0.01, .Eccentricity = 0.0001, .Stiffness = 400, .Dissipation = 3};
    const auto stationary = ScrapingForce({.SlopeX = 2, .CurvatureX = 10}, {.NormalForce = 1}, scraping);
    Near(stationary.Horizontal + stationary.Vertical, 0, 0, "Stationary scraping generates no excitation");
    const auto released = RollingForce({}, {.NormalForce = 0}, scraping, rolling);
    Near(released.Horizontal + released.Vertical + released.Rolling, 0, 0, "Released rolling generates no excitation");
    const auto static_contact = RollingForce({}, {.NormalForce = 1}, scraping, rolling);
    Near(static_contact.Rolling, rolling.Stiffness * std::pow(rolling.Radius - rolling.Eccentricity, 1.5), 1e-15, "Printed rolling equations retain static elastic load");
    constexpr double period = 0.2, mean = 0.003, amplitude = 0.0002, omega = 2 * std::numbers::pi / period;
    const auto power = [&](double t) {
        const double rho = mean + amplitude * std::cos(omega * t), rho_dot = -amplitude * omega * std::sin(omega * t);
        const Trajectory trajectory{.Height = rho - rolling.Radius + rolling.Eccentricity * std::cos(t / rolling.Radius), .SlopeX = rho_dot - rolling.Eccentricity / rolling.Radius * std::sin(t / rolling.Radius)};
        const auto force = RollingForce(trajectory, {.X = t, .VelocityX = 1, .NormalForce = 1}, scraping, rolling);
        Check(force.Rolling >= 0, "Admissible slow rolling cycle has no attractive rolling force");
        return force.Rolling * rho_dot;
    };
    const double dissipation = Integrate([&](double t) { const double rho = mean + amplitude * std::cos(omega * t), velocity = -amplitude * omega * std::sin(omega * t); return rolling.Dissipation * std::pow(rho, 1.5) * velocity * velocity; }, 0, period, 2048);
    const double work = Integrate(power, 0, period, 1024);
    Check(dissipation > 0, "Rolling damping dissipates positive work");
    Near(work, dissipation, 1e-15, "Closed prescribed penetration cycle stores no net elastic energy");
}
void TestTemporalAtlasPrimitive() {
    constexpr uint32_t count = 17, levels = 2, frames = 101;
    constexpr double spacing = .002;
    std::vector<float> atlas(3 * count * levels), motion(4 * frames);
    struct Location {
        uint32_t Cell;
        float Fraction;
    };
    std::vector<Location> coordinates(frames);
    const auto primitive = [](double x, double multiplier) { return std::array{1e-5 + multiplier * (.003 * x + 2 * x * x + 10 * x * x * x / 6), multiplier * (.003 + 4 * x + 5 * x * x), multiplier * (4 + 10 * x)}; };
    for (uint32_t level = 0; level < levels; ++level)
        for (uint32_t x = 0; x < count; ++x) {
            const auto value = primitive(x * spacing, level + 1);
            for (uint32_t field = 0; field < 3; ++field) atlas[level * 3 * count + field * count + x] = float(value[field]);
        }
    for (uint32_t frame = 0; frame < frames; ++frame) {
        motion[4 * frame] = float((count - 1) * spacing * (frame + .37) / (frames + 1));
        motion[4 * frame + 2] = float(1 + 5. * frame / (frames - 1));
        const double grid_x = motion[4 * frame] / spacing;
        coordinates[frame] = {uint32_t(grid_x), float(grid_x - uint32_t(grid_x))};
    }
    struct Block {
        uint32_t Count, Levels, Frames;
        float Spacing, Mass, Beta1, Beta2, Radius, Eccentricity;
        float AlphaScale{1};
        uint32_t VerticalMode{};
    };
    auto gpu = CreateGpu();
    const auto parameters = Upload(gpu, Block{count, levels, frames, float(spacing), .1f, .05f, 1, .01f, .0001f}), surface = Upload<float>(gpu, atlas), path = Upload<float>(gpu, motion), output = CreateBuffer(gpu, 3 * frames * sizeof(float)), status = CreateBuffer(gpu, frames * sizeof(uint32_t));
    const auto kernel = CreateKernel(gpu, "AgarwalAtlasTrajectoryV2");
    const auto locations = Upload<Location>(gpu, coordinates);
    const std::array bindings{GpuBinding{parameters, 0}, GpuBinding{surface, 1}, GpuBinding{path, 2}, GpuBinding{output, 3}, GpuBinding{status, 4}, GpuBinding{locations, 5}};
    BeginGpu(gpu);
    DispatchGpu(gpu, kernel, bindings, {frames});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto result = BufferSpan<float>(output);
    for (uint32_t frame = 0; frame < frames; ++frame) {
        Check(BufferSpan<uint32_t>(status)[frame] == 0, "Temporal polynomial atlas path valid");
        const double alpha = Alpha(motion[4 * frame + 2], {.NormalMin = 1, .NormalMax = 6}), weight = (alpha - .01) / .04;
        const auto expected = primitive(motion[4 * frame], 1 + weight);
        for (uint32_t field = 0; field < 3; ++field) Near(result[field * frames + frame], expected[field], field == 0 ? 2e-9 : field == 1 ? 2e-7 :
                                                                                                                                            2e-5,
                                                          "Integrated-curvature polynomial primitive");
    }
    // Full microscope scans require retaining fractions beyond the precision of an absolute FP32 grid coordinate.
    constexpr uint32_t large_count = 400003;
    std::vector<float> large_atlas(3 * large_count * levels);
    for (uint32_t level = 0; level < levels; ++level) {
        large_atlas[level * 3 * large_count + 2 * large_count + 400000] = 1000;
        large_atlas[level * 3 * large_count + 2 * large_count + 400001] = -2000;
    }
    for (uint32_t frame = 0; frame < frames; ++frame) coordinates[frame] = {400000, float((frame + .37) / (frames + 1))};
    const auto large_parameters = Upload(gpu, Block{large_count, levels, frames, 1.394068e-6f, .1f, .05f, 1, .01f, .0001f}), large_surface = Upload<float>(gpu, large_atlas), large_locations = Upload<Location>(gpu, coordinates);
    const std::array large_bindings{GpuBinding{large_parameters, 0}, GpuBinding{large_surface, 1}, GpuBinding{path, 2}, GpuBinding{output, 3}, GpuBinding{status, 4}, GpuBinding{large_locations, 5}};
    BeginGpu(gpu);
    DispatchGpu(gpu, kernel, large_bindings, {frames});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    for (uint32_t frame = 0; frame < frames; ++frame) Near(BufferSpan<float>(output)[2 * frames + frame], 1000 - 3000. * coordinates[frame].Fraction, .0002, "Full-profile integer cell and fractional interpolation");
}
void TestTemporalGaussian() {
    constexpr uint32_t frames = 513, rate = 44100;
    constexpr double amplitude = 1e-8, wavenumber = 2 * std::numbers::pi / 20e-6;
    struct Block {
        uint32_t Frames, SampleRate;
        float HalfWidth, Mass, Beta1, Beta2, Radius, Eccentricity;
        uint32_t VerticalMode{};
    };
    const Block block{frames, rate, 5, .1f, .05f, 1, .01f, .0001f};
    auto gpu = CreateGpu();
    const auto kernel = CreateKernel(gpu, "AgarwalTemporalForcesV2");
    for (bool reversing : {false, true}) {
        std::vector<float> trajectory(3 * frames), motion(4 * frames);
        std::vector<double> positions(frames), normal(frames);
        for (uint32_t n = 0; n < frames; ++n) {
            const double phase = 2 * std::numbers::pi * (double(n) - 256) / 256;
            positions[n] = reversing ? .001 + .0002 * std::cos(phase) : .001 + .2 * n / rate;
            normal[n] = reversing ? 3.5 + 2.5 * std::cos(phase) : 1 + 5 * std::pow((.05 - .028) / .04, 1 / .95);
            trajectory[n] = float(amplitude * std::sin(wavenumber * positions[n]));
            trajectory[frames + n] = float(amplitude * wavenumber * std::cos(wavenumber * positions[n]));
            trajectory[2 * frames + n] = float(-amplitude * wavenumber * wavenumber * std::sin(wavenumber * positions[n]));
            motion[4 * n] = float(positions[n]);
            motion[4 * n + 1] = reversing ? (n % 128 == 0 ? 0 : float(-.0002 * (2 * std::numbers::pi * rate / 256) * std::sin(phase))) : .2f;
            motion[4 * n + 2] = float(normal[n]);
            motion[4 * n + 3] = float(n) / frames;
        }
        const auto parameters = Upload(gpu, block), input = Upload<float>(gpu, trajectory), path = Upload<float>(gpu, motion), output = CreateBuffer(gpu, 4 * frames * sizeof(float)), status = CreateBuffer(gpu, frames * sizeof(uint32_t)), filtered = CreateBuffer(gpu, 3 * frames * sizeof(float));
        const std::array bindings{GpuBinding{parameters, 0}, GpuBinding{input, 1}, GpuBinding{path, 2}, GpuBinding{output, 3}, GpuBinding{status, 4}, GpuBinding{filtered, 5}};
        BeginGpu(gpu);
        DispatchGpu(gpu, kernel, bindings, {frames});
        SubmitGpu(gpu);
        WaitGpu(gpu);
        for (auto value : BufferSpan<uint32_t>(status)) Check(value == 0, "Temporal Gaussian valid penetration");
        const auto result = BufferSpan<float>(filtered), forces = BufferSpan<float>(output);
        for (uint32_t n = 0; n < frames; ++n) {
            const double alpha = Alpha(normal[n], {.NormalMin = 1, .NormalMax = 6}), half_width = 5 * alpha / .03, sigma = .4 * half_width;
            const int radius = int(std::ceil(half_width));
            std::array<double, 3> expected{};
            double normalization = 0;
            const auto height_at_shift = [&](double shift) {double sum=0,weights=0;for(int m=std::max(0,int(n)-radius);m<=std::min(int(frames)-1,int(n)+radius);++m){const double d=m-double(n),weight=std::exp(-.5*d*d/(sigma*sigma));sum+=weight*amplitude*std::sin(wavenumber*(positions[m]+shift));weights+=weight;}return sum/weights; };
            for (int m = std::max(0, int(n) - radius); m <= std::min(int(frames) - 1, int(n) + radius); ++m) {
                const double d = m - double(n), weight = std::exp(-.5 * d * d / (sigma * sigma));
                for (uint32_t field = 0; field < 3; ++field) expected[field] += weight * trajectory[field * frames + m];
                normalization += weight;
            }
            for (uint32_t field = 0; field < 3; ++field) Near(result[field * frames + n], expected[field] / normalization, field == 0 ? 3e-14 : field == 1 ? 1e-8 :
                                                                                                                                                             .004,
                                                              "Temporal Gaussian CPU reference");
            if (!reversing && n > uint32_t(radius) && n + uint32_t(radius) < frames) {
                double transfer = 0, weights = 0;
                for (int d = -radius; d <= radius; ++d) {
                    const double weight = std::exp(-.5 * d * d / (sigma * sigma));
                    transfer += weight * std::cos(wavenumber * .2 / rate * d);
                    weights += weight;
                }
                Near(result[n], trajectory[n] * transfer / weights, 3e-14, "Constant-speed temporal/spatial Gaussian equivalence");
            }
            if (reversing && n % 128 == 0) Near(forces[n], 0, 0, "Temporal smoothing preserves zero scraping force at reversal");
            constexpr double step = 1e-8;
            Near(result[frames + n], (height_at_shift(step) - height_at_shift(-step)) / (2 * step), 2e-8, "Translated field first derivative");
            Near(result[2 * frames + n], (height_at_shift(step) - 2 * height_at_shift(0) + height_at_shift(-step)) / (step * step), .005, "Translated field second derivative");
        }
    }
    std::cout << "Agarwal temporal Gaussian constant-speed, reversal and translated-derivative tests passed\n";
}
void TestVerticalForceAblation(uint32_t mode) {
    constexpr uint32_t count = 17, frames = 257, rate = 44100;
    constexpr double curvature = 40, mass = .003;
    struct SampleBlock {
        uint32_t Count, Levels, Frames;
        float Spacing, Mass, Beta1, Beta2, Radius, Eccentricity, AlphaScale;
        uint32_t VerticalMode;
    };
    struct TemporalBlock {
        uint32_t Frames, SampleRate;
        float HalfWidth, Mass, Beta1, Beta2, Radius, Eccentricity;
        uint32_t VerticalMode;
    };
    struct Location {
        uint32_t Cell;
        float Fraction;
    };
    std::vector<float> raw(count);
    for (uint32_t cell = 0; cell < count; ++cell) raw[cell] = cell % 2 ? float(curvature) : -float(curvature);
    std::vector<float> motion(4 * frames);
    std::vector<Location> locations(frames);
    for (uint32_t frame = 0; frame < frames; ++frame) {
        const double phase = 2 * std::numbers::pi * frame / (frames - 1);
        motion[4 * frame] = float(.01 + .001 * std::sin(phase));
        motion[4 * frame + 1] = float(.2 * std::sin(phase));
        motion[4 * frame + 2] = float(3.5 + 2.5 * std::cos(phase));
        motion[4 * frame + 3] = float(frame) / (frames - 1);
        locations[frame] = {frame % (count - 1), .37f};
    }
    auto gpu = CreateGpu();
    const auto sample_parameters = Upload(gpu, SampleBlock{count, 0, frames, .001f, float(mass), 0, 1, .01f, .0001f, 1, mode});
    const auto temporal_parameters = Upload(gpu, TemporalBlock{frames, rate, 5, float(mass), 0, 1, .01f, .0001f, mode});
    const auto raw_buffer = Upload<float>(gpu, raw), path = Upload<float>(gpu, motion), location = Upload<Location>(gpu, locations);
    const auto sampled = CreateBuffer(gpu, 3 * frames * sizeof(float)), filtered = CreateBuffer(gpu, 3 * frames * sizeof(float));
    const auto output = CreateBuffer(gpu, 4 * frames * sizeof(float)), status = CreateBuffer(gpu, frames * sizeof(uint32_t));
    const auto sample_kernel = CreateKernel(gpu, "AgarwalVerticalTrajectoryV2"), temporal_kernel = CreateKernel(gpu, "AgarwalTemporalForcesV2");
    const std::array sample_bindings{GpuBinding{sample_parameters, 0}, GpuBinding{raw_buffer, 1}, GpuBinding{path, 2}, GpuBinding{sampled, 3}, GpuBinding{status, 4}, GpuBinding{location, 5}};
    const std::array temporal_bindings{GpuBinding{temporal_parameters, 0}, GpuBinding{sampled, 1}, GpuBinding{path, 2}, GpuBinding{output, 3}, GpuBinding{status, 4}, GpuBinding{filtered, 5}};
    BeginGpu(gpu);
    DispatchGpu(gpu, sample_kernel, sample_bindings, {frames});
    DispatchGpu(gpu, temporal_kernel, temporal_bindings, {frames});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto force = BufferSpan<float>(output), trajectory = BufferSpan<float>(filtered);
    for (uint32_t frame = 0; frame < frames; ++frame) {
        const double alpha = .05 - .04 * std::pow((motion[4 * frame + 2] - 1.) / 5, .95);
        const double width = 5 * alpha / .03, sigma = .4 * width;
        const int half = int(std::ceil(width));
        double sum = 0, normalization = 0;
        for (int sample = std::max(0, int(frame) - half); sample <= std::min(int(frames) - 1, int(frame) + half); ++sample) {
            const double local_alpha = .05 - .04 * std::pow((motion[4 * sample + 2] - 1.) / 5, .95);
            const double distance = (sample - double(frame)) / sigma, weight = std::exp(-.5 * distance * distance);
            const auto position = locations[sample];
            const double left = raw[position.Cell], right = raw[position.Cell + 1], fraction = position.Fraction;
            const double constrained = mode == 2 ? std::tanh(local_alpha * std::lerp(left, right, fraction)) / local_alpha : std::lerp(std::tanh(local_alpha * left), std::tanh(local_alpha * right), fraction) / local_alpha;
            sum += weight * constrained;
            normalization += weight;
        }
        const double expected_curvature = sum / normalization, velocity = motion[4 * frame + 1];
        Near(trajectory[2 * frames + frame], expected_curvature, 2e-5, "Direct requested-alpha curvature and temporal Gaussian");
        Near(force[frame], mass * expected_curvature * velocity * velocity, 4e-9, "Vertical-only Eq3 force");
        for (uint32_t field = 0; field < 2; ++field) Near(trajectory[field * frames + frame], 0, 0, "Unused trajectory field remains zero");
        Near(force[frames + frame], 0, 0, "Vertical ablation omits elastic force");
        Near(force[2 * frames + frame], 0, 0, "Vertical ablation omits dissipative force");
        Near(force[3 * frames + frame], motion[4 * frame + 3], 0, "Vertical ablation preserves response location");
        Check(BufferSpan<uint32_t>(status)[frame] == 0, "Vertical-only path valid");
    }
}
void TestOversampledVertical() {
    constexpr uint32_t frames{129}, rate{44100}, count{260 * (frames - 1) + 3};
    constexpr double omega{2 * std::numbers::pi * 65 / 64}, width{4.9};
    struct DenseBlock {
        uint32_t Frames, VerticalMode;
        float AlphaScale;
    };
    struct FilterBlock {
        uint32_t Frames, DenseFrames, Factor, SampleRate;
        float HalfWidth, Mass, SigmaRatio;
    };
    struct Location {
        uint32_t Cell;
        float Fraction;
    };
    for (double sigma_ratio : {.4, .2}) {
        const double sigma = sigma_ratio * width;
        // Independent continuous truncated-Gaussian quadrature. The production sum
        // includes ceil(width * factor), so its support converges from outside.
        double integral{0}, normalization{0};
        constexpr uint32_t intervals{20000};
        for (uint32_t i = 0; i <= intervals; ++i) {
            const double t = -width + 2 * width * i / intervals;
            const double weight = (i == 0 || i == intervals ? 1 : i % 2 ? 4 :
                                                                          2) *
                std::exp(-.5 * t * t / (sigma * sigma));
            integral += weight * std::cos(omega * t);
            normalization += weight;
        }
        const double continuous = integral / normalization;
        auto gpu = CreateGpu();
        const auto dense_kernel = CreateKernel(gpu, "AgarwalDenseVerticalCurvature"), filter_kernel = CreateKernel(gpu, "AgarwalDownsampleVerticalForceV2");
        for (bool reversing : {false, true}) {
            for (uint32_t mode : {1u, 2u}) {
                double previous_error{1};
                for (uint32_t factor : {1u, 4u, 8u, 16u}) {
                    const uint32_t dense_frames = (frames - 1) * factor + 1;
                    const float alpha_scale = reversing ? 1.f : 1e-6f;
                    std::vector<float> raw(count), motion(4 * frames), normals(dense_frames);
                    std::vector<double> positions(frames), expected_dense(dense_frames);
                    std::vector<Location> locations(dense_frames);
                    for (uint32_t cell = 0; cell < count; ++cell) raw[cell] = float((reversing ? 40 : 1) * std::cos(2 * std::numbers::pi * cell / 256));
                    for (uint32_t n = 0; n < frames; ++n) {
                        const double phase = 2 * std::numbers::pi * n / (frames - 1);
                        positions[n] = reversing ? 500 + 300 * std::cos(phase) : 260. * n;
                        motion[4 * n + 1] = reversing ? (n % 64 == 0 ? 0 : float(std::sin(phase))) : 1;
                        motion[4 * n + 2] = reversing ? float(3.5 + 2.5 * std::cos(phase)) : 1;
                        motion[4 * n + 3] = float(n) / (frames - 1);
                    }
                    for (uint32_t j = 0; j < dense_frames; ++j) {
                        const uint32_t left = j / factor, right = std::min(left + 1, frames - 1);
                        const double fraction = double(j % factor) / factor;
                        const double x = std::lerp(positions[left], positions[right], fraction);
                        const uint32_t cell = uint32_t(x);
                        locations[j] = {cell, float(x - cell)};
                        normals[j] = float(std::lerp(double(motion[4 * left + 2]), double(motion[4 * right + 2]), fraction));
                        const double alpha = alpha_scale * Alpha(normals[j], {.NormalMin = 1, .NormalMax = 6});
                        expected_dense[j] = mode == 2 ? std::tanh(alpha * std::lerp(double(raw[cell]), double(raw[cell + 1]), double(locations[j].Fraction))) / alpha : std::lerp(std::tanh(alpha * raw[cell]), std::tanh(alpha * raw[cell + 1]), double(locations[j].Fraction)) / alpha;
                    }
                    const auto dense_parameters = Upload(gpu, DenseBlock{dense_frames, mode, alpha_scale}), filter_parameters = Upload(gpu, FilterBlock{frames, dense_frames, factor, rate, 2.94f, .1f, float(sigma_ratio)});
                    const auto surface = Upload<float>(gpu, raw), coordinates = Upload<Location>(gpu, locations), normal = Upload<float>(gpu, normals), path = Upload<float>(gpu, motion);
                    const auto dense = CreateBuffer(gpu, dense_frames * sizeof(float)), force = CreateBuffer(gpu, frames * sizeof(float)), filtered = CreateBuffer(gpu, frames * sizeof(float));
                    const std::array dense_bindings{GpuBinding{dense_parameters, 0}, GpuBinding{surface, 1}, GpuBinding{coordinates, 2}, GpuBinding{normal, 3}, GpuBinding{dense, 4}};
                    const std::array filter_bindings{GpuBinding{filter_parameters, 0}, GpuBinding{dense, 1}, GpuBinding{path, 2}, GpuBinding{force, 3}, GpuBinding{filtered, 4}};
                    BeginGpu(gpu);
                    DispatchGpu(gpu, dense_kernel, dense_bindings, {dense_frames});
                    DispatchGpu(gpu, filter_kernel, filter_bindings, {frames});
                    SubmitGpu(gpu);
                    WaitGpu(gpu);
                    const auto curvature = BufferSpan<float>(filtered), forces = BufferSpan<float>(force), dense_values = BufferSpan<float>(dense);
                    for (uint32_t j = 0; j < dense_frames; ++j) Near(dense_values[j], expected_dense[j], reversing ? 2e-5 : 3e-7, "Dense constrained spatial curvature");
                    double correlation{0}, energy{0};
                    for (uint32_t n = 0; n < frames; ++n) {
                        const double local_width = double(2.94f) * Alpha(motion[4 * n + 2], {.NormalMin = 1, .NormalMax = 6}) / .03 * factor;
                        const double local_sigma = double(float(sigma_ratio)) * local_width;
                        const int center = int(n * factor), radius = int(std::ceil(local_width));
                        double sum{0}, weights{0};
                        for (int j = std::max(0, center - radius); j <= std::min(int(dense_frames) - 1, center + radius); ++j) {
                            const double d = double(j - center) / local_sigma, weight = std::exp(-.5 * d * d);
                            sum += weight * expected_dense[j];
                            weights += weight;
                        }
                        const double expected = sum / weights, velocity = motion[4 * n + 1];
                        Near(curvature[n], expected, reversing ? 4e-5 : 2e-6, "Oversampled Gaussian including truncated edges and variable load");
                        Near(forces[n], .1 * expected * velocity * velocity, reversing ? 5e-6 : 3e-7, "Oversampled vertical force and reversing velocity");
                        if (velocity == 0) Near(forces[n], 0, 0, "Zero velocity remains exactly silent");
                        if (!reversing && n >= 6 && n + 6 < frames) {
                            const double carrier = std::cos(omega * n);
                            correlation += curvature[n] * carrier;
                            energy += carrier * carrier;
                        }
                    }
                    if (!reversing) {
                        const double transfer = correlation / energy, error = std::abs(transfer - continuous);
                        if (factor == 1) Check(transfer > .9, "Audio-rate sampling aliases ultrasonic curvature to low frequency");
                        else {
                            Check(std::abs(transfer) < .004, "Dense filtering suppresses the aliased carrier");
                            if (sigma_ratio == .4) Check(error < previous_error, "Factors 4, 8, 16 converge toward continuous truncated Gaussian");
                            else Near(transfer, continuous, 3e-6, "Narrower Gaussian at unchanged support agrees with continuous quadrature");
                        }
                        previous_error = error;
                        std::cout << "Vertical Gaussian mode=" << mode << " sigma_ratio=" << sigma_ratio << " factor=" << factor << " transfer=" << transfer << " continuous=" << continuous << " error=" << error << '\n';
                    }
                }
            }
        }
    }
}
void TestOversampledFullForce() {
    constexpr uint32_t count{17}, frames{65}, factor{8}, dense_frames{(frames - 1) * factor + 1}, rate{44100};
    constexpr double spacing{.001};
    struct SampleBlock {
        uint32_t Count, Levels, Frames;
        float Spacing, Mass, Beta1, Beta2, Radius, Eccentricity, AlphaScale;
        uint32_t VerticalMode;
    };
    struct FilterBlock {
        uint32_t Frames, DenseFrames, Factor, SampleRate;
        float HalfWidth, Mass, Beta1, Beta2, Radius, Eccentricity, SigmaRatio;
    };
    struct Location {
        uint32_t Cell;
        float Fraction;
    };
    const FilterBlock filter_block{frames, dense_frames, factor, rate, 2.94f, .1f, .07f, 1.3f, .01f, .0001f, .2f};
    const auto polynomial = [](double x, double level) {
        const double a = .001 + .0002 * level, b = .002 + .001 * level, c = 4 + level, d = 20 + 10 * level;
        return std::array{a + b * x + c * x * x / 2 + d * x * x * x / 6, b + c * x + d * x * x / 2, c + d * x};
    };
    std::vector<float> atlas(6 * count), motion(4 * frames), dense_motion(4 * dense_frames);
    std::vector<Location> locations(dense_frames);
    std::vector<std::array<double, 3>> expected_dense(dense_frames);
    for (uint32_t level = 0; level < 2; ++level)
        for (uint32_t cell = 0; cell < count; ++cell) {
            const auto values = polynomial(cell * spacing, level);
            for (uint32_t field = 0; field < 3; ++field) atlas[level * 3 * count + field * count + cell] = float(values[field]);
        }
    for (uint32_t n = 0; n < frames; ++n) {
        const double phase = 2 * std::numbers::pi * n / (frames - 1);
        motion[4 * n] = float(.008 + .006 * std::cos(phase));
        motion[4 * n + 1] = n % 32 == 0 ? 0 : float(.2 * std::sin(phase));
        motion[4 * n + 2] = float(3.5 + 2.5 * std::cos(phase));
        motion[4 * n + 3] = float(n) / (frames - 1);
    }
    for (uint32_t j = 0; j < dense_frames; ++j) {
        const uint32_t left = j / factor, right = std::min(left + 1, frames - 1);
        const double fraction = double(j % factor) / factor;
        for (uint32_t field = 0; field < 4; ++field) dense_motion[4 * j + field] = float(std::lerp(double(motion[4 * left + field]), double(motion[4 * right + field]), fraction));
        const double x = std::lerp(double(motion[4 * left]), double(motion[4 * right]), fraction), grid = x / spacing;
        locations[j] = {uint32_t(grid), float(grid - uint32_t(grid))};
        const double alpha = Alpha(dense_motion[4 * j + 2], {.NormalMin = 1, .NormalMax = 6});
        expected_dense[j] = polynomial(x, (alpha - .01) / .04);
    }
    auto gpu = CreateGpu();
    const auto sample_parameters = Upload(gpu, SampleBlock{count, 2, dense_frames, float(spacing), 0, 0, 0, 0, 0, 1, 0}), filter_parameters = Upload(gpu, filter_block);
    const auto surface = Upload<float>(gpu, atlas), dense_path = Upload<float>(gpu, dense_motion), path = Upload<float>(gpu, motion), coordinates = Upload<Location>(gpu, locations);
    const auto sampled = CreateBuffer(gpu, 3 * dense_frames * sizeof(float)), sample_status = CreateBuffer(gpu, dense_frames * sizeof(uint32_t)), status = CreateBuffer(gpu, frames * sizeof(uint32_t));
    const auto filtered = CreateBuffer(gpu, 3 * frames * sizeof(float)), output = CreateBuffer(gpu, 4 * frames * sizeof(float));
    const auto sample_kernel = CreateKernel(gpu, "AgarwalAtlasTrajectoryV2"), filter_kernel = CreateKernel(gpu, "AgarwalDownsampleFullForce");
    const std::array sample_bindings{GpuBinding{sample_parameters, 0}, GpuBinding{surface, 1}, GpuBinding{dense_path, 2}, GpuBinding{sampled, 3}, GpuBinding{sample_status, 4}, GpuBinding{coordinates, 5}};
    const std::array filter_bindings{GpuBinding{filter_parameters, 0}, GpuBinding{sampled, 1}, GpuBinding{path, 2}, GpuBinding{output, 3}, GpuBinding{status, 4}, GpuBinding{filtered, 5}};
    BeginGpu(gpu);
    DispatchGpu(gpu, sample_kernel, sample_bindings, {dense_frames});
    DispatchGpu(gpu, filter_kernel, filter_bindings, {frames});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    for (uint32_t value : BufferSpan<uint32_t>(sample_status)) Check(value == 0, "Dense polynomial atlas sample valid");
    for (uint32_t n = 0; n < frames; ++n) {
        const double width = double(filter_block.HalfWidth) * Alpha(motion[4 * n + 2], {.NormalMin = 1, .NormalMax = 6}) / .03 * factor;
        const double sigma = filter_block.SigmaRatio * width;
        const int center = int(n * factor), radius = int(std::ceil(width));
        std::array<double, 3> expected{};
        double weights{0};
        for (int j = std::max(0, center - radius); j <= std::min(int(dense_frames) - 1, center + radius); ++j) {
            const double distance = (j - double(center)) / sigma, weight = std::exp(-.5 * distance * distance);
            for (uint32_t field = 0; field < 3; ++field) expected[field] += weight * expected_dense[j][field];
            weights += weight;
        }
        for (uint32_t field = 0; field < 3; ++field) {
            expected[field] /= weights;
            Near(BufferSpan<float>(filtered)[field * frames + n], expected[field], field == 0 ? 2e-9 : field == 1 ? 2e-7 :
                                                                                                                    2e-5,
                 "Dense polynomial height and spatial derivatives filtered consistently");
        }
        const double x = motion[4 * n], velocity = motion[4 * n + 1], radius_m = filter_block.Radius, eccentricity = filter_block.Eccentricity;
        const double rho = radius_m - eccentricity * std::cos(x / radius_m) + expected[0];
        const double rho_dot = eccentricity / radius_m * velocity * std::sin(x / radius_m) + velocity * expected[1];
        const double scrape = filter_block.Beta1 * std::pow(std::abs(velocity * expected[1]), filter_block.Beta2) + filter_block.Mass * expected[2] * velocity * velocity;
        const auto force = BufferSpan<float>(output);
        Near(force[n], scrape, 5e-7, "Resolved full horizontal and vertical scraping terms");
        Near(force[frames + n], std::pow(rho, 1.5), 3e-9, "Resolved full elastic penetration term");
        Near(force[2 * frames + n], std::pow(rho, 1.5) * rho_dot, 2e-10, "Resolved full penetration derivative and damping term");
        Near(force[3 * frames + n], motion[4 * n + 3], 0, "Full force retains base response position");
        Check(BufferSpan<uint32_t>(status)[n] == 0, "Resolved full trajectory has nonnegative penetration");
        if (velocity == 0) {
            Near(force[n], 0, 0, "Full scraping vanishes at zero velocity");
            Near(force[2 * frames + n], 0, 0, "Full damping vanishes at zero velocity");
            Check(force[frames + n] > 0, "Printed elastic baseline retained at zero velocity");
        }
    }
    const FilterBlock horizontal_block{frames, dense_frames, factor, rate, filter_block.HalfWidth, 0, filter_block.Beta1, filter_block.Beta2, filter_block.Radius, filter_block.Eccentricity, filter_block.SigmaRatio};
    const auto horizontal_parameters = Upload(gpu, horizontal_block), horizontal_output = CreateBuffer(gpu, 4 * frames * sizeof(float));
    const std::array horizontal_bindings{GpuBinding{horizontal_parameters, 0}, GpuBinding{sampled, 1}, GpuBinding{path, 2}, GpuBinding{horizontal_output, 3}, GpuBinding{status, 4}, GpuBinding{filtered, 5}};
    BeginGpu(gpu);
    DispatchGpu(gpu, filter_kernel, horizontal_bindings, {frames});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    for (uint32_t n = 0; n < frames; ++n) {
        const double expected = horizontal_block.Beta1 * std::pow(std::abs(double(motion[4 * n + 1]) * BufferSpan<float>(filtered)[frames + n]), horizontal_block.Beta2);
        Near(BufferSpan<float>(horizontal_output)[n], expected, 2e-9, "Zero vertical coefficient preserves horizontal micro-collisions");
        for (uint32_t field = 1; field < 4; ++field)
            Near(BufferSpan<float>(horizontal_output)[field * frames + n], BufferSpan<float>(output)[field * frames + n], 0, "Horizontal isolation preserves rolling components and motion");
    }
    std::fill_n(BufferSpan<float>(sampled).begin(), dense_frames, -.02f);
    BeginGpu(gpu);
    DispatchGpu(gpu, filter_kernel, filter_bindings, {frames});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    for (uint32_t value : BufferSpan<uint32_t>(status)) Check(value == 2, "Resolved full force rejects negative penetration");
}
void TestForceBeforeDecimation() {
    constexpr uint32_t frames{257}, factor{8}, dense_frames{(frames - 1) * factor + 1}, rate{44100};
    struct FilterBlock {
        uint32_t Frames, DenseFrames, Factor, SampleRate;
        float HalfWidth, Mass, SigmaRatio;
    };
    struct RetainBlock {
        uint32_t Frames, DenseFrames, Factor, Fields;
    };
    std::vector<float> curvature(dense_frames, 1), motion(4 * dense_frames);
    for (uint32_t j = 0; j < dense_frames; ++j) {
        const double phase = 2 * std::numbers::pi * .35 * j / factor;
        motion[4 * j + 1] = float(std::cos(phase));
        motion[4 * j + 2] = 1;
    }
    auto gpu = CreateGpu();
    const auto parameters = Upload(gpu, FilterBlock{dense_frames, dense_frames, 1, rate * factor, 5, 1, .1f});
    const auto retain_parameters = Upload(gpu, RetainBlock{frames, dense_frames, factor, 1});
    const auto input = Upload<float>(gpu, curvature), path = Upload<float>(gpu, motion);
    const auto dense_force = CreateBuffer(gpu, dense_frames * sizeof(float)), filtered = CreateBuffer(gpu, dense_frames * sizeof(float));
    const auto output = CreateBuffer(gpu, frames * sizeof(float)), retained = CreateBuffer(gpu, frames * sizeof(float));
    const auto physical_kernel = CreateKernel(gpu, "AgarwalDownsampleVerticalForceV2"), retain_kernel = CreateKernel(gpu, "AgarwalRetainFields");
    const auto decimation = CreateDecimatePlan(gpu, dense_frames, 1, factor);
    const std::array physical_bindings{GpuBinding{parameters, 0}, GpuBinding{input, 1}, GpuBinding{path, 2}, GpuBinding{dense_force, 3}, GpuBinding{filtered, 4}};
    const std::array retain_bindings{GpuBinding{retain_parameters, 0}, GpuBinding{filtered, 1}, GpuBinding{retained, 2}};
    BeginGpu(gpu);
    DispatchGpu(gpu, physical_kernel, physical_bindings, {dense_frames});
    DispatchDecimateGpu(gpu, decimation, dense_force, output);
    DispatchGpu(gpu, retain_kernel, retain_bindings, {frames});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    double wrong_order_difference{0};
    for (uint32_t n = 0; n < frames; ++n) {
        Near(BufferSpan<float>(retained)[n], BufferSpan<float>(filtered)[n * factor], 0, "Retained physical Gaussian trajectory keeps exact output grid");
        if (n > 64 && n + 64 < frames) {
            // Squaring velocity creates .7 cycles/output-frame, above output Nyquist.
            // Correct dense-force filtering retains its DC term and removes the alias.
            Near(BufferSpan<float>(output)[n], .5, 3e-6, "Nonlinear velocity-squared force precedes antialias filtering");
            const double sampled_velocity = std::cos(2 * std::numbers::pi * .35 * n);
            wrong_order_difference = std::max(wrong_order_difference, std::abs(BufferSpan<float>(output)[n] - sampled_velocity * sampled_velocity));
        }
    }
    Check(wrong_order_difference > .4, "Filtering curvature before base-rate force evaluation cannot replace force decimation");
}
} // namespace

int main() {
    try {
        TestTrajectory();
        TestForces();
        TestImpulseResponses();
        TestGpuTrajectory();
        TestGridConvergence();
        TestTemporalConvergence();
        TestPhysicalScope();
        TestTemporalAtlasPrimitive();
        TestTemporalGaussian();
        TestVerticalForceAblation(1);
        TestVerticalForceAblation(2);
        TestOversampledVertical();
        TestOversampledFullForce();
        TestForceBeforeDecimation();
        std::cout << "Agarwal analytic force, constrained trajectory, modal morph and block-continuity tests passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
