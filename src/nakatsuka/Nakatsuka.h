#pragma once
#include "core/Gpu.h"
#include <array>
#include <vector>

namespace surface_audio::nakatsuka {
struct Material {
    double Modulus{}, Poisson{}, ArealDensity{}, Thickness{}, Width{}, Height{}, WidthDeviation{}, HeightDeviation{};
};
struct Patch {
    float Width{}, Height{};
};
struct Sample {
    float Speed{}, Displacement{}, Distance{}, Contact{};
};
struct AcousticSettings {
    double SampleRate{44100}, ReferenceSpeed{.1}, AirDensity{1.204}, SoundSpeed{343}, Attenuation{};
    uint32_t OddModes{3};
};
struct Scene {
    uint32_t Columns{30}, Rows{30}, Iterations{4}, ShiftedAdhesion{0};
    float SampleRate{44100}, Spacing{.004}, InverseMass{900}, Speed{.1};
    float Gravity{9.81f}, Damping{4}, Stretch{.8f}, Adhesion{.01f};
    float AdhesionDistance{.0001f}, SphereRadius{.04f}, InitialHeight{.041f}, ListenerHeight{.5f};
    float MotionStart{}, MotionDuration{}, SettlingTime{}, OriginY{}, MotionRise{}, MotionFall{};
};
struct Simulation {
    std::vector<Sample> Samples;
    std::vector<float> FinalPositions;
};
double AdhesionPotential(double distance, double degree, double effective_distance);
// The printed Z/r residual projects outward; the shifted residual has its zero at effective_distance.
double AdhesionCorrection(double distance, double degree, double effective_distance, bool shifted);
double AngularFrequency(const Material &, double width, double height, uint32_t m, uint32_t n);
std::vector<Patch> MakePatches(const Material &, uint32_t count, uint64_t seed);
// Samples are patch-major, at the simulation rate; the scene uses a rectangular sheet and a sphere (radius zero selects a floor).
Simulation SimulateGpu(Gpu &, const Scene &, uint32_t frames);
std::vector<double> Render(const Material &, const AcousticSettings &, std::span<const Patch>, std::span<const Sample>);
std::vector<float> RenderGpu(Gpu &, const Material &, const AcousticSettings &, std::span<const Patch>, std::span<const Sample>);
}
