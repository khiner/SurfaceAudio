#pragma once
#include "HattLayout.h"
#include "core/Gpu.h"
#include <array>
#include <vector>

namespace surface_audio::hatt {
struct Model {
    double Speed{}, Force{}, Variance{}, Gain{1};
    std::vector<double> ArLsf, MaLsf;
};
struct Texture {
    double SampleRate{}, Friction{}, MaximumSpeed{}, MaximumForce{};
    std::vector<Model> Models;
    std::vector<std::array<uint32_t, 3>> Triangles;
};
struct Control {
    double Speed{}, Force{};
};
struct Filter {
    std::array<double, PolynomialSize> Ar{1}, Ma{1};
    double Variance{};
    uint32_t ArOrder{}, MaOrder{};
};
struct State {
    std::array<double, MaximumOrder> Output{}, Excitation{};
};
void Validate(const Texture &);
// Controls use mm/s and N; saturation follows the author renderer when upstream_bounds is true.
// Requires a texture accepted by Validate.
Filter Interpolate(const Texture &, Control, bool upstream_bounds = false);
// Requires a filter returned by Interpolate; state persists across control changes.
double Tick(const Filter &, State &, double standard_normal);
double FrictionForce(double velocity_mm_s, double normal_force, double coefficient, bool upstream = false);
// Controls and innovations are texture-major; interpolation and filtering run on Metal from zero history.
std::vector<float> RenderGpu(Gpu &, std::span<const Texture>, std::span<const Control>, std::span<const float> standard_normal, bool upstream_bounds = false);
}
