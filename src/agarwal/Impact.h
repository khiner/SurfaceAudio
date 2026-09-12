#pragma once

#include "core/Gpu.h"
#include <limits>
#include <vector>

namespace surface_audio::agarwal {
enum class ImpactMaterial : uint32_t { Glass,
                                       Ceramic,
                                       Metal,
                                       Stone,
                                       Wood,
                                       Plastic,
                                       Cardboard };
enum class StiffnessCombination : uint32_t { Series,
                                             Sum };
enum class ImpactSampling : uint32_t { Point,
                                       CellAverage };
struct ImpactSettings {
    double Mass{.1}, Velocity{2}, StiffnessA{1e7}, StiffnessB{7.03e7};
    // The papers leave the proportionality A and material clipping limits uncalibrated.
    double Scale{1}, ForceLimit{std::numeric_limits<double>::infinity()};
    StiffnessCombination Combination{StiffnessCombination::Series};
};

double EquivalentStiffness(double a, double b, StiffnessCombination = StiffnessCombination::Series);
double PlateStiffness(double young_modulus, double poisson_ratio, double side, double thickness);
// Table 3 is retained separately because some entries disagree with the printed plate formula and Table 2.
double PublishedStiffness(ImpactMaterial, uint32_t size_category);
double ImpactDuration(const ImpactSettings &);
double ImpactForce(double time, const ImpactSettings &);
// Returns voice-major samples, including zeros after each pulse; frames must cover every complete pulse.
std::vector<float> RenderImpactForcesGpu(Gpu &, std::span<const ImpactSettings>, uint32_t frames, double sample_rate, ImpactSampling = ImpactSampling::CellAverage);
std::vector<float> MixContactForcesGpu(Gpu &, std::span<const float> scraping, std::span<const float> elastic, std::span<const float> dissipative, double stiffness, double dissipation);
// Uses linear pulses regardless of ForceLimit.
std::vector<float> RenderMicroImpactGpu(Gpu &, const ImpactSettings &, double sample_rate, ImpactSampling = ImpactSampling::CellAverage);
std::vector<float> RenderContactGpu(Gpu &, std::span<const float> excitation, const ImpactSettings &, double sample_rate, std::span<const float> response_a, std::span<const float> response_b = {});
}
