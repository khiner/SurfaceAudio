#pragma once
#include "Gpu.h"
#include <vector>

namespace surface_audio {
// Complete mono linear convolution, including the response tail, with no implicit gain or resampling.
std::vector<float> ConvolveFixedGpu(Gpu &, std::span<const float> excitation, std::span<const float> response);
} // namespace surface_audio
