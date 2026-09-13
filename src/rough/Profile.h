#pragma once
#include "core/Gpu.h"
#include <vector>

namespace surface_audio::rough {
// Returns the requested N-point periodic Gaussian convolution from Bergström's rsgeng1D formula.
// RMS and correlation length specify the ensemble; individual profiles retain their sample mean and variance.
std::vector<double> GaussianProfile(Gpu &, unsigned nodes, double spacing, double rms, double correlation, uint64_t seed);
// Returns x-contiguous samples of the thesis's separable periodic 2D Gaussian convolution at the requested grid size.
std::vector<double> GaussianSurface(Gpu &, unsigned x_nodes, unsigned y_nodes, double step_x, double step_y, double rms, double correlation_x, double correlation_y, uint64_t seed);
}
