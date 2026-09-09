#include "GpuMix.h"

#include <array>
#include <cmath>

namespace surface_audio {
GpuMix CreateGpuMix(Gpu &gpu, MixBlock block) {
    if (!block.Voices || !block.Frames || block.Field >= block.Stride || !std::isfinite(block.Gain) || uint64_t(block.Voices) * block.Frames * block.Stride > UINT32_MAX) throw std::invalid_argument("Invalid GPU mix dimensions");
    return {block, Upload(gpu, block), CreateBuffer(gpu, std::size_t(block.Frames) * sizeof(float)), CreateKernel(gpu, "MixVoices")};
}
void EncodeMix(Gpu &gpu, const GpuMix &mix, GpuBuffer input) {
    if (input.Size < uint64_t(mix.Block.Voices) * mix.Block.Frames * mix.Block.Stride * sizeof(float)) throw std::invalid_argument("GPU mix input is too small");
    const std::array bindings{GpuBinding{mix.Parameters, 0}, GpuBinding{input, 1}, GpuBinding{mix.Output, 2}};
    DispatchGpu(gpu, mix.Kernel, bindings, {mix.Block.Frames});
}
} // namespace surface_audio
