#pragma once
#include "Response.h"
#include "core/Gaussian.h"

namespace surface_audio::agarwal {
struct ObjectResponseDistribution {
    Gaussian<3> Modes;
    std::array<Gaussian<2>, ObjectResponseNoiseCount> Noise;
};
struct ObjectResponseSampleStats {
    uint64_t ModeRejections{}, NoiseRejections{};
};

// Fit one material/size category after the paper's frequency and RT60 exclusions.
ObjectResponseDistribution FitObjectResponseDistribution(std::span<const ObjectResponseParameters>);
ObjectResponseParameters SampleObjectResponse(const ObjectResponseDistribution &, RandomState &, ObjectResponseSampleStats * = nullptr);
}
