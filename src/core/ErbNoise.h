#pragma once
#include "Gpu.h"
#include <vector>

namespace surface_audio {
struct ErbNoiseSettings {
    uint32_t TapCount{513};
    uint64_t Seed{2023};
    double LowHz{}, HighHz{};
};
// Returns band-major noise with unit-L2 Hamming-sinc filters and shared Gaussian input extended through both FIR margins.
// HighHz=0 selects Nyquist, and TapCount must be odd and at least three.
std::vector<float> CreateErbNoise(Gpu &, uint32_t bands, uint32_t frames, double sample_rate, const ErbNoiseSettings & = {});
std::vector<double> ErbNoiseReference(uint32_t bands, uint32_t frames, double sample_rate, const ErbNoiseSettings & = {});
}
