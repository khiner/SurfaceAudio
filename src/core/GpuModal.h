#pragma once
#include "Gpu.h"
#include "Modal.h"

namespace surface_audio {
struct GpuModal {
    ModalBlock Block;
    GpuBuffer Parameters, Coefficients, State, Output;
    GpuKernel Synthesize;
    uint32_t Threads{};
};
// One lane per mode, padded to a power of two with at least 32 lanes.
// Rejects banks exceeding 1024 lanes or the device kernel's threadgroup limit.
GpuModal CreateGpuModal(Gpu &, const ModalBank &, uint32_t frames);
void EncodeModal(Gpu &, const GpuModal &, GpuBuffer excitation);
} // namespace surface_audio
