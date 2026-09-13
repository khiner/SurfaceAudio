#pragma once

#include "Response.h"
#include "core/Random.h"

namespace surface_audio::agarwal {

struct ResponseMaterial {
    uint32_t Records{};
    std::array<double, 3> ModeMean{}; // Frequency Hz, amplitude dB, RT60 seconds.
    std::array<double, 20> NoiseMean{}; // Ten amplitudes dB, then ten RT60 seconds.
    // Covariance is factor^T * factor, with observation-major centered factors scaled by 1/sqrt(count).
    std::vector<std::array<double, 3>> ModeFactor;
    std::vector<std::array<double, 20>> NoiseFactor;
};

struct ResponseMaterialSampleStats {
    uint64_t ModeRejections{}, NoiseRejections{};
};

// Requires at least two records and returns one modal 3D Gaussian and one noise 20D Gaussian with 1/n covariance.
ResponseMaterial FitResponseMaterial(std::span<const ResponseParameters> records);
// Returns ten independent mode triples and one joint noise vector with Hz in [20,20000] and positive RT60.
// Accumulates rejection counts and throws after one million attempts per vector.
ResponseParameters SampleResponseMaterial(const ResponseMaterial &, RandomState &, ResponseMaterialSampleStats * = nullptr);

}
