#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace surface_audio::conan {

struct ReferenceParameters {
    double SampleRate{44100};
    double Mass{0.01};
    double Stiffness{(1e10 / 3) * 0.01};
    double Dissipation{1};
    double Exponent{1.5};
    double Gravity{9.81};
    double Speed{0.2};
    double Spacing{0.0001};
    unsigned Substeps{8};
};

struct ReferenceState {
    double Position{};
    double Velocity{};
    double Distance{};
};

// Equation (4), with SI material properties. Both Poisson ratios must lie in (-1, .5).
double ContactStiffness(double radius, double young_exciter, double poisson_exciter, double young_resonator, double poisson_resonator);
double ContactForce(const ReferenceParameters &parameters, double compression, double velocity);
std::vector<double> FractalSurface(unsigned sample_count, double power_exponent, double maximum_height, uint64_t seed);
std::vector<double> RollingCurve(std::span<const double> surface, double spacing, double radius);
void RenderReference(const ReferenceParameters &parameters, ReferenceState &state, std::span<const double> rolling_curve, std::span<float> output);

struct ContactMeasurement {
    double Duration{};
    double PeakForce{};
    double ExitVelocity{};
};
ContactMeasurement MeasureContact(const ReferenceParameters &parameters, double input_velocity);

} // namespace surface_audio::conan
