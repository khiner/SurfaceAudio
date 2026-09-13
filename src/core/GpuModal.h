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
// Requires a mode count whose power-of-two padding fits both 1024 lanes and the device threadgroup limit.
GpuModal CreateGpuModal(Gpu &, const ModalBank &, uint32_t frames);
void EncodeModal(Gpu &, const GpuModal &, GpuBuffer excitation);
}
