#pragma once
#include "core/Gpu.h"
#include <array>
#include <vector>

namespace surface_audio::rough {
struct SurfaceSpectrum {
    double Rms{1}, Low{.46}, High{1}, Exponent{4}, Minimum{};
};
struct SurfaceView {
    std::span<const float> Height;
    unsigned XNodes, YNodes;
};
struct SeparationStatistics {
    double Mean, Deviation, Skewness;
};
// Frequencies are cycles per unit length; moments include the radial Jacobian and integrate to Rms squared.
std::array<double, 3> SurfaceMoments(SurfaceSpectrum);
// Requires power-of-two Y nodes and power-of-two X nodes, optionally multiplied by five.
std::vector<float> SelfAffineSurface(Gpu &, unsigned x_nodes, unsigned y_nodes, double spacing, SurfaceSpectrum, uint64_t seed);
// Returns max(track(x+u,y)-slider(x,y)) at every whole-grid offset, with fixed orientation and complete overlap.
std::vector<float> SlidingSeparation(Gpu &, SurfaceView track, SurfaceView slider);
SeparationStatistics ExtremeSeparation(double patches);
SeparationStatistics CorrelatedSeparation(double patches, double bandwidth);
// Returns the empirical crossover frequency, output bandwidth and normalized central wavelength from Equations 9–11.
std::array<double, 3> SeparationSpectrum(double size_over_wavelength, double low_over_high, double bandwidth);
}
