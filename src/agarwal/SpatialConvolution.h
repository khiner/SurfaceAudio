#pragma once

#include "Agarwal.h"
#include "core/Gpu.h"

namespace surface_audio::agarwal {
// Eq. (13) at each output location, with changing endpoint frequencies and positive lag envelopes.
// Tail holds the last morph value. GPU workspace: block_frames * TapCount response tile plus inputs.
std::vector<float> ConvolveSpatialGpu(Gpu &, std::span<const float> excitation, std::span<const float> morph, const ImpulseResponses &, uint32_t block_frames = 256, float gain = 1);
} // namespace surface_audio::agarwal
