// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "core/Gpu.h"
#include "matusiak/DistributedGpuTypes.h"
#include "matusiak/Lumped.h"
#include "matusiak/Matusiak.h"
#include <span>
#include <vector>
namespace surface_audio::matusiak {
struct LumpedBatch {
    std::vector<float> Displacement;
    std::vector<float> MaximumEnergyError, MaximumResidual;
    std::vector<uint32_t> FailedSteps;
};
// Fig. 3 bowed masses; one trajectory per GPU thread.
LumpedBatch RenderLumpedGpu(Gpu &, std::span<const LumpedParameters<float>>, uint32_t frames);
struct StringBatch {
    std::vector<float> BridgeForce, MaximumResidual;
    // Per voice: U, PreviousU, W, PreviousW, Hair, PreviousHair, Z, Velocity, MidpointZ.
    std::vector<float> FinalState;
    std::vector<uint32_t> FailedSteps;
};
// Voices share physical string parameters.
StringBatch RenderStringsGpu(Gpu &, Parameters, std::span<const BowDrive>, uint32_t frames);
} // namespace surface_audio::matusiak
