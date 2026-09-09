#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace surface_audio::agarwal {

struct SurfaceGrid {
    uint32_t Width{}, Height{};
    double SpacingX{}, SpacingY{};
    std::span<const double> Heights;
    // Required integration anchors: one x slope per left-edge row, one y slope per top-edge column.
    std::span<const double> LeftSlopes, TopSlopes;
};

struct ConstraintSettings {
    double NormalMin{1}, NormalMax{10};
    double AlphaMin{0.01}, AlphaMax{0.05}, Exponent{0.95};
    double ConstantAlpha{0.03};
    double ReferenceAlpha{0.03}, GaussianHalfWidth{5}, GaussianSigmaRatio{0.4};
};

struct Motion {
    double X{}, Y{}, VelocityX{}, VelocityY{}, NormalForce{1}, Morph{};
};

struct Trajectory {
    double Height{}, SlopeX{}, SlopeY{}, CurvatureX{}, CurvatureY{};
};

struct ScrapingSettings {
    double Mass{0.1}, Beta1{0.05}, Beta2{1};
};

struct RollingSettings {
    double Radius{0.01}, Eccentricity{0.0001};
    double Stiffness{1000}, Dissipation{0.1};
};

struct Force {
    double Horizontal{}, Vertical{}, Rolling{};
};

struct NormalRange {
    double Min{}, Max{};
};

struct ModalEndpoint {
    std::span<const double> Frequencies;
    // Mode-major positive amplitude envelopes, [mode * TapCount + lag].
    std::span<const double> Amplitudes;
};

struct ImpulseResponses {
    double SampleRate{44100}, ObjectGain{1};
    uint32_t TapCount{};
    ModalEndpoint Surface0, Surface1, Object;
};

double Alpha(double normal_force, const ConstraintSettings &settings);
double ConstrainCurvature(double curvature, double alpha);
NormalRange ScrapingNormalRange(double mass, double angular_frequency, double half_length, double angle, double friction, double gravity = 9.81);
double ScrapingNormalForce(double position, double mass, double angular_frequency, double angle, double friction, double gravity = 9.81);
double RollingPosition(double angle, const RollingSettings &settings);
double RollingVelocity(double angle, double angular_velocity, const RollingSettings &settings);
double RollingNormalForce(double angle, double angular_velocity, double angular_acceleration, double mass, const RollingSettings &settings, double gravity = 9.81);

void Validate(const SurfaceGrid &surface, const ConstraintSettings &settings);
void Validate(const ImpulseResponses &responses);
// Coordinates must lie inside the finite grid.
Trajectory SampleTrajectory(const SurfaceGrid &surface, const Motion &motion, const ConstraintSettings &settings);
void PrepareTrajectory(const SurfaceGrid &surface, std::span<const Motion> motion, const ConstraintSettings &settings, std::span<Trajectory> trajectory);
Force ScrapingForce(const Trajectory &trajectory, const Motion &motion, const ScrapingSettings &settings);
Force RollingForce(const Trajectory &trajectory, const Motion &motion, const ScrapingSettings &scraping, const RollingSettings &rolling);
void GenerateForces(std::span<const Trajectory> trajectory, std::span<const Motion> motion, const ScrapingSettings &scraping, const RollingSettings *rolling, std::span<float> output);

void BuildImpulseResponse(const ImpulseResponses &responses, double morph, std::span<float> output);
// Frame-major coefficients for core/Convolution.h: [frame * TapCount + lag].
void BuildImpulseResponseBlock(const ImpulseResponses &responses, std::span<const float> morph, std::span<float> coefficients);

// Exponential envelopes are an explicit convenience for modal data without measured envelopes.
std::vector<double> ExponentialEnvelopes(std::span<const double> amplitude, std::span<const double> decay_seconds, uint32_t tap_count, double sample_rate);

} // namespace surface_audio::agarwal
