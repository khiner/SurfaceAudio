// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "core/Gpu.h"
#include "matusiak2024/DistributedGpuTypes.h"
#include "matusiak2024/Matusiak2024.h"
#include <span>
#include <vector>
namespace surface_audio::matusiak2024 {
struct StringBatch {
    std::vector<float> BridgeForce, MaximumResidual, StoredEnergy, EnergyError;
    // Per voice: U, PreviousU, W, PreviousW, Hair, PreviousHair, Z, Velocity, MidpointZ.
    std::vector<float> FinalState;
    std::vector<uint32_t> FailedSteps;
};
StringBatch RenderStringsGpu(Gpu &, Parameters, std::span<const BowDrive>, uint32_t frames);
}
