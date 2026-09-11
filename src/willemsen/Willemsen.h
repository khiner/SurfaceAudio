#pragma once
#include "core/Gpu.h"
#include "willemsen/Interaction.h"
#include <vector>
namespace surface_audio::willemsen {
enum class Scheme { Paper,
                    AuthorFigure };
struct Parameters {
    double SampleRate{44100}, Fundamental{440}, Length{1}, Density{7850}, Radius{.0005}, Young{2e11};
    double Loss0{1}, Loss1{.005}, BowPosition{.25}, PickupPosition{2. / 3}, Noise{.02};
    Scheme Discretization{Scheme::Paper};
};
Model<double> MakeModel(Parameters = {});
Model<float> MakeFloatModel(Parameters = {});
struct Drive {
    float Velocity{.1f}, NormalForce{5};
    unsigned Seed{2019};
};
struct Batch {
    std::vector<float> Displacement, FinalState, Residual;
    std::vector<unsigned> FailedSteps;
};
Batch RenderGpu(Gpu &, Parameters, std::span<const Drive>, unsigned frames);
}
