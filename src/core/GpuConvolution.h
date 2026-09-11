#pragma once
#include "Gpu.h"
#include <vector>

namespace surface_audio {
// Returns the complete linear convolution.
std::vector<float> ConvolveFftGpu(Gpu &, std::span<const float> input, std::span<const float> taps);
// Returns the complete mono linear convolution at the input gain and sample rate.
std::vector<float> ConvolveFixedGpu(Gpu &, std::span<const float> excitation, std::span<const float> response);
}
