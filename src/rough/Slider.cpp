#include "Slider.h"
#include "core/PivotedSolve.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace surface_audio::rough {
SliderParameters<double> InstrumentedSlider(double damping_ratio) {
    if (!std::isfinite(damping_ratio) || damping_ratio < 0) throw std::invalid_argument("Invalid slider damping ratio");
    constexpr double gravity = 9.81, mass = 2.5 / gravity, radius = .00075, young = 210e9, poisson = .3;
    const double stiffness = 4. / 3 * std::sqrt(radius) * young / (2 * (1 - poisson * poisson));
    const double inertia = mass * (.06 * .06 + .029 * .029) / 12;
    const double damping = std::pow(1944 * stiffness * stiffness / (mass * mass * std::pow(gravity, 5)), 1. / 6) * damping_ratio;
    return {{{.01742, .01595, 107e-6}, {.01753, -.00112, -30e-6}, {.01745, -.01787, -96e-6}, {.00059, .01609, 14e-6}, {.00044, -.00104, -30e-6}, {.00043, -.01799, 55e-6}, {-.01644, .01600, -68e-6}, {-.01649, .00092, -46e-6}, {-.01642, -.01811, 94e-6}}, mass, inertia, inertia, gravity, stiffness, damping};
}
SliderState<double> EquilibrateSlider(const SliderParameters<double> &p, const SliderTrack<double> &track, double tolerance) {
    for (double x : {p.Mass, p.InertiaX, p.InertiaY, p.Gravity, p.Stiffness, tolerance})
        if (!std::isfinite(x) || x <= 0) throw std::invalid_argument("Invalid slider equilibrium parameters");
    constexpr double scale = .03;
    const double load = p.Mass * p.Gravity;
    double lowest = p.Points[0].Height + track.Height[0];
    for (unsigned j = 0; j < 9; ++j) {
        for (double x : {p.Points[j].X, p.Points[j].Y, p.Points[j].Height, track.Height[j]})
            if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite slider geometry");
        if (track.Velocity[j] != 0) throw std::invalid_argument("Static slider equilibrium requires stationary track inputs");
        lowest = std::min(lowest, p.Points[j].Height + track.Height[j]);
    }
    std::array<double, 3> q{lowest - std::pow(load / (9 * p.Stiffness), 2. / 3), 0, 0};
    const auto evaluate = [&](const std::array<double, 3> &position, std::array<double, 3> &gradient, std::array<double, 9> &hessian) {
        gradient = {load, 0, 0};
        hessian = {};
        double potential = load * position[0];
        for (unsigned j = 0; j < 9; ++j) {
            const auto point = p.Points[j];
            const std::array b{1., point.Y / scale, -point.X / scale};
            const double indentation = track.Height[j] + point.Height - b[0] * position[0] - b[1] * position[1] - b[2] * position[2];
            const auto c = EvaluateNormalContact(indentation, 0., p.Stiffness, 0., 1.5, 1.5);
            potential += c.Energy;
            for (unsigned a = 0; a < 3; ++a) {
                gradient[a] -= b[a] * c.Force;
                for (unsigned b_index = 0; b_index < 3; ++b_index) hessian[3 * a + b_index] += c.ElasticTangent * b[a] * b[b_index];
            }
        }
        return potential;
    };
    const auto norm = [](const std::array<double, 3> &g) { return std::max({std::abs(g[0]), std::abs(g[1]), std::abs(g[2])}); };
    for (unsigned iteration = 0; iteration < 200; ++iteration) {
        std::array<double, 3> gradient{};
        std::array<double, 9> hessian{};
        const double energy = evaluate(q, gradient, hessian), error = norm(gradient);
        if (error < tolerance) return {{q[0], q[1] / scale, q[2] / scale}, {}};
        const double regularization = 1e-12 * std::max({hessian[0], hessian[4], hessian[8], 1.});
        for (unsigned k = 0; k < 3; ++k) hessian[4 * k] += regularization;
        auto direction = gradient;
        if (!Solve(hessian.data(), direction.data(), 3, 0.)) throw std::runtime_error("Singular slider equilibrium");
        double descent{};
        for (unsigned k = 0; k < 3; ++k) descent += gradient[k] * direction[k];
        bool accepted{};
        for (double step = 1; step > 1e-12; step *= .5) {
            const std::array candidate{q[0] - step * direction[0], q[1] - step * direction[1], q[2] - step * direction[2]};
            std::array<double, 3> g{};
            std::array<double, 9> h{};
            const double e = evaluate(candidate, g, h);
            if (e < energy - 1e-4 * step * descent || (error < 1e-4 * load && norm(g) < error)) {
                q = candidate;
                accepted = true;
                break;
            }
        }
        if (!accepted) throw std::runtime_error("Slider equilibrium line search failed");
    }
    throw std::runtime_error("Slider equilibrium did not converge");
}
double SensorStep(double time, double amplitude, double cutoff, double quality) {
    if (!std::isfinite(time) || !std::isfinite(amplitude) || !std::isfinite(cutoff) || cutoff <= 0 || !std::isfinite(quality) || quality <= 0) throw std::invalid_argument("Invalid sensor response parameters");
    if (time < 0) return 0;
    const double omega = 2 * std::numbers::pi * cutoff, decay = omega / (2 * quality), discriminant = omega * omega - decay * decay;
    if (discriminant == 0) return amplitude * std::exp(-decay * time) * (1 - decay * time);
    if (discriminant > 0) {
        const double frequency = std::sqrt(discriminant);
        return amplitude * std::exp(-decay * time) * (std::cos(frequency * time) - decay / frequency * std::sin(frequency * time));
    }
    const double rate = std::sqrt(-discriminant), a = -decay + rate, b = -decay - rate;
    return amplitude * (a * std::exp(a * time) - b * std::exp(b * time)) / (a - b);
}
}
