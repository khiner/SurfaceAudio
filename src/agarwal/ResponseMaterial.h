#pragma once

#include "Response.h"
#include "core/Random.h"

namespace surface_audio::agarwal {

struct ResponseMaterial {
    uint32_t Records{};
    std::array<double, 3> ModeMean{}; // Frequency Hz, amplitude dB, RT60 seconds.
    std::array<double, 20> NoiseMean{}; // Ten amplitudes dB, then ten RT60 seconds.
    // Observation-major centered data divided by sqrt(observation count). Covariance is factor^T * factor.
    std::vector<std::array<double, 3>> ModeFactor;
    std::vector<std::array<double, 20>> NoiseFactor;
};

struct ResponseMaterialSampleStats {
    uint64_t ModeRejections{}, NoiseRejections{};
};

// Requires >=2 records. Pools modes into one 3D Gaussian and noise into one 20D Gaussian.
// MLE covariance (1/n) retains deficient rank without regularization.
ResponseMaterial FitResponseMaterial(std::span<const ResponseParameters> records);
// Draws ten independent triples and one joint noise vector. Rejects whole draws with Hz outside [20,20000] or RT60<=0.
// Statistics accumulate rejections; throws after one million attempts per vector.
ResponseParameters SampleResponseMaterial(const ResponseMaterial &, RandomState &, ResponseMaterialSampleStats * = nullptr);

} // namespace surface_audio::agarwal
