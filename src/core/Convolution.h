#pragma once

#include <cstdint>
#include <span>

namespace surface_audio {
struct FirBlock {
    uint32_t TapCount, FrameCount;
};

// Coefficients[frame*TapCount+lag], excitation[TapCount-1 history samples, FrameCount new samples].
// Evaluating the IR at the output location preserves Agarwal Eq. 13.
void Convolve(const FirBlock &, std::span<const float> coefficients, std::span<const float> excitation, std::span<float> output);
} // namespace surface_audio
