#pragma once

#include "core/Gpu.h"
#include "core/Random.h"
#include <vector>

namespace surface_audio::traer {
inline constexpr uint32_t PaperModeCount = 15, PaperNoiseBandCount = 30;
struct Resonance {
    float Frequency{}, OnsetDb{}, DecayDbPerSecond{};
    uint32_t EndFrame{0xffffffffu};
};
struct Transient {
    float OnsetDb{}, DecayDbPerSecond{};
};
struct Gaussian {
    std::vector<double> Mean, Lower;
};
struct ResponseGpu {
    uint32_t Frames{}, Voices{};
    GpuBuffer Parameters, Resonances, Transients, Noise, Output;
    GpuKernel Synthesize;
};
// Covariance is row-major; rank-deficient positive semidefinite matrices are accepted.
Gaussian MakeGaussian(std::span<const double> mean, std::span<const double> covariance);
std::vector<double> SampleGaussian(const Gaussian &, RandomState &);
// Joint parameter order: frequency Hz, onset dB, decay dB/second, with one contiguous plane each.
std::vector<Resonance> SampleModes(const Gaussian &, double mean_spacing, double sample_rate, RandomState &, uint32_t attempts = 10000);
void PerturbOnsets(std::span<Resonance>, RandomState &);
void EvaluateResponse(std::span<const Resonance>, std::span<const Transient>, uint32_t frames, double sample_rate, std::span<const float> noise_bandmajor, std::span<double> output);
void EvaluateResponseFloat(std::span<const Resonance>, std::span<const Transient>, uint32_t frames, float sample_rate, std::span<const float> noise_bandmajor, std::span<float> output);
// Each voice has its own contiguous mode, transient, noise, and output records.
ResponseGpu CreateResponseGpu(Gpu &, uint32_t frames, uint32_t voices, float sample_rate, std::span<const Resonance>, std::span<const Transient>, std::span<const float> noise);
void EncodeResponse(Gpu &, const ResponseGpu &);
double SpringContactDuration(double mass, double stiffness);
std::vector<float> ImpactForce(double duration, double peak_force, double sample_rate);
// Signed power makes Eq. 8 real for negative slopes and nonintegral gamma.
double ScrapeForce(double slope, double curvature, double velocity, double mass, double shear_gain, double gamma);
std::vector<float> ScrapeExcitation(std::span<const double> profile, double spacing, std::span<const double> position, std::span<const double> velocity, double mass, double shear_gain, double gamma);
// Rows are contiguous measured profiles; overlap samples are blended between selected rows.
std::vector<double> QuiltProfile(std::span<const double> rows, uint32_t columns, uint32_t samples, uint32_t overlap, RandomState &);
// Responses are node-major; output-time position selects the IR mixture and the tail holds the final position.
std::vector<float> ConvolveSpatialGpu(Gpu &, std::span<const float> excitation, std::span<const float> responses, uint32_t taps, std::span<const float> nodes, std::span<const float> positions);
}
