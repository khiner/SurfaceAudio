#include "GpuModal.h"

#include <algorithm>
#include <array>
#include <bit>

namespace surface_audio {
GpuModal CreateGpuModal(Gpu &gpu, const ModalBank &bank, uint32_t frames) {
    const auto count = uint64_t(bank.Voices) * bank.Modes;
    if (!frames || !count || count > UINT32_MAX / 3 || uint64_t(bank.Voices) * frames > UINT32_MAX || bank.Coefficients.size() != 3 * count || bank.State.size() != 2 * count || bank.Modes > 1024) throw std::invalid_argument("Invalid GPU modal dimensions");
    const auto kernel = CreateKernel(gpu, "ModalSynthesize");
    const auto threads = std::bit_ceil(std::max(bank.Modes, 32u));
    if (threads > kernel.MaxThreads) throw std::invalid_argument("Modal bank exceeds GPU kernel threadgroup limit");
    const ModalBlock block{bank.Voices, bank.Modes, frames};
    return {block, Upload(gpu, block), Upload<float>(gpu, bank.Coefficients), Upload<float>(gpu, bank.State), CreateBuffer(gpu, std::size_t(bank.Voices) * frames * sizeof(float)), kernel, threads};
}
void EncodeModal(Gpu &gpu, const GpuModal &modal, GpuBuffer excitation) {
    if (excitation.Size < std::size_t(modal.Block.Voices) * modal.Block.Frames * sizeof(float)) throw std::invalid_argument("GPU modal input is too small");
    const std::array bindings{GpuBinding{modal.Parameters, 0}, GpuBinding{modal.Coefficients, 1}, GpuBinding{modal.State, 2}, GpuBinding{excitation, 3}, GpuBinding{modal.Output, 4}};
    DispatchGroupsGpu(gpu, modal.Synthesize, bindings, {modal.Block.Voices}, {modal.Threads});
}
} // namespace surface_audio
