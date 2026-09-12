#pragma once
#include <span>
#include <vector>

namespace surface_audio {
// Returns a monic polynomial from strictly increasing line frequencies in (0, pi).
std::vector<double> LineSpectrumPolynomial(std::span<const double> frequencies);
}
