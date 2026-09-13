#pragma once

#include "Agarwal.h"
#include "core/Gpu.h"

namespace surface_audio::agarwal {
// Requires positive lag envelopes and returns complete Eq. 13 convolution using the final morph value through the tail.
// Workspace contains a block_frames * TapCount response tile plus inputs.
std::vector<float> ConvolveSpatialGpu(Gpu &, std::span<const float> excitation, std::span<const float> morph, const ImpulseResponses &, uint32_t block_frames = 256, float gain = 1);
}
