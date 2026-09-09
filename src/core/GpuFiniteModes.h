#pragma once
#include "Gpu.h"
#include <vector>

namespace surface_audio {
struct FiniteMode {
    double Frequency{}, Decay{}, Amplitude0{}, Amplitude1{};
};
// Fixed frequencies/decays with logarithmic amplitude interpolation at the output location.
// Complete finite convolution; the tail holds the last morph value. No implicit gain normalization.
std::vector<float> ConvolveFiniteModesGpu(Gpu &, std::span<const float> excitation, std::span<const float> morph, std::span<const FiniteMode>, uint32_t sample_rate, uint32_t taps, float gain = 1);
} // namespace surface_audio
