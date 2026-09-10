#pragma once
#include "core/Gpu.h"
#include "falaize/Interaction.h"
#include <vector>
namespace surface_audio::falaize {
Model<double> MakeModel(Parameters<double> = {});
Model<float> MakeFloatModel(Parameters<double> = {});
struct Batch {
    std::vector<float> Velocity, FinalState, EnergyError, Residual;
    std::vector<unsigned> FailedSteps;
};
Batch RenderGpu(Gpu &, Parameters<double>, std::span<const float> drives, unsigned frames, bool hammer);
}
