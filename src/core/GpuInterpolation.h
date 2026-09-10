#pragma once
#include "Gpu.h"
#include <vector>

namespace surface_audio {
// Interpolates node-major signals at each output position, with strictly increasing nodes and clamping beyond endpoints.
std::vector<float> InterpolateSignalsGpu(Gpu &, std::span<const float> signals, std::span<const float> nodes, std::span<const float> positions);
}
