#include "Flexural.h"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <queue>
#include <stdexcept>
#include <tuple>

namespace surface_audio {
namespace {
constexpr double Pi = std::numbers::pi;
bool Positive(double x) { return std::isfinite(x) && x > 0; }
bool Nonnegative(double x) { return std::isfinite(x) && x >= 0; }
void Validate(BeamProperties p) {
    if (!Positive(p.Length) || !Positive(p.MassPerLength) || !Positive(p.BendingStiffness) || !Nonnegative(p.DampingRatio) ||
        (p.Boundary != BeamBoundary::SimplySupported && p.Boundary != BeamBoundary::Free)) throw std::invalid_argument("Invalid beam properties");
}
void Validate(PlateProperties p) {
    if (!Positive(p.Length) || !Positive(p.Width) || !Positive(p.MassPerArea) || !Positive(p.BendingStiffness) || !Nonnegative(p.DampingRatio)) throw std::invalid_argument("Invalid plate properties");
}
double FreeRoot(unsigned mode) {
    // cos(lambda) = sech(lambda) avoids overflow in cosh(lambda)*cos(lambda) = 1.
    const double center = (mode + 1.5) * Pi;
    double left = center - .3, right = center + .3;
    const auto residual = [](double x) { const double e = std::exp(-x); return std::cos(x) - 2 * e / (1 + e * e); };
    const double sign = residual(left);
    for (unsigned i = 0; i < 60; ++i) {
        const double middle = (left + right) / 2;
        if (residual(middle) * sign > 0) left = middle;
        else right = middle;
    }
    return (left + right) / 2;
}
}
std::vector<BeamMode> MakeBeamModes(BeamProperties p, unsigned bending_modes) {
    Validate(p);
    if (bending_modes > 65536) throw std::invalid_argument("Too many beam modes");
    std::vector<BeamMode> modes;
    modes.reserve(bending_modes + (p.Boundary == BeamBoundary::Free ? 2 : 0));
    if (p.Boundary == BeamBoundary::Free) {
        modes.push_back({.Scale = 1 / std::sqrt(p.Length), .Kind = 1});
        modes.push_back({.Scale = std::sqrt(12 / std::pow(p.Length, 3)), .Kind = 2});
    }
    for (unsigned i = 0; i < bending_modes; ++i) {
        const double lambda = p.Boundary == BeamBoundary::Free ? FreeRoot(i) : (i + 1) * Pi;
        const double wave = lambda / p.Length, omega = wave * wave * std::sqrt(p.BendingStiffness / p.MassPerLength);
        if (!Positive(omega)) throw std::invalid_argument("Beam frequency is outside FP64 range");
        if (p.Boundary == BeamBoundary::SimplySupported) modes.push_back({wave, omega, std::sqrt(2 / p.Length), 0});
        else {
            const bool odd = i & 1;
            const double half = lambda / 2, e = std::exp(-half);
            const double ratio = (odd ? std::sin(half) : std::cos(half)) * 2 * e / (1 + (odd ? -1 : 1) * e * e);
            modes.push_back({wave, omega, std::sqrt(2 / p.Length / (1 + (odd ? -1 : 1) * ratio * ratio)), odd ? 4u : 3u});
        }
    }
    return modes;
}
double BeamShape(BeamProperties p, const BeamMode &m, double x, unsigned derivative) {
    if (!std::isfinite(x) || x < 0 || x > p.Length || derivative > 4) throw std::invalid_argument("Invalid beam coordinate or derivative");
    if (m.Kind == 1) return derivative ? 0 : m.Scale;
    if (m.Kind == 2) return derivative == 0 ? m.Scale * (x - p.Length / 2) : derivative == 1 ? m.Scale :
                                                                                               0;
    const double factor = m.Scale * std::pow(m.WaveNumber, derivative);
    if (m.Kind == 0) return factor * std::sin(m.WaveNumber * x + derivative * Pi / 2);
    const double z = m.WaveNumber * (x - p.Length / 2), half = m.WaveNumber * p.Length / 2;
    const bool odd = m.Kind == 4, sinh_term = odd != bool(derivative & 1);
    // Centered even/odd shapes keep the hyperbolic terms bounded at high modal order.
    const double hyperbolic = (odd ? std::sin(half) : std::cos(half)) *
        (std::exp(z - half) + (sinh_term ? -1 : 1) * std::exp(-z - half)) /
        (1 + (odd ? -1 : 1) * std::exp(-2 * half));
    const double trig = odd ? std::sin(z + derivative * Pi / 2) : std::cos(z + derivative * Pi / 2);
    return factor * (trig + hyperbolic);
}
std::vector<PlateMode> MakePlateModes(PlateProperties p, unsigned modes) {
    Validate(p);
    if (modes > 65536) throw std::invalid_argument("Too many plate modes");
    const auto omega = [p](unsigned x, unsigned y) {
        return Pi * Pi * std::sqrt(p.BendingStiffness / p.MassPerArea) * (std::pow(x / p.Length, 2) + std::pow(y / p.Width, 2));
    };
    using Entry = std::tuple<double, unsigned, unsigned>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
    for (unsigned x = 1; x <= modes; ++x) queue.emplace(omega(x, 1), x, 1);
    std::vector<PlateMode> result;
    result.reserve(modes);
    while (result.size() < modes) {
        const auto [frequency, x, y] = queue.top();
        queue.pop();
        if (!Positive(frequency)) throw std::invalid_argument("Plate frequency is outside FP64 range");
        result.push_back({x, y, frequency});
        queue.emplace(omega(x, y + 1), x, y + 1);
    }
    return result;
}
double PlateShape(PlateProperties p, PlateMode m, double x, double y) {
    if (!std::isfinite(x) || !std::isfinite(y) || x < 0 || x > p.Length || y < 0 || y > p.Width) throw std::invalid_argument("Invalid plate coordinate");
    return 2 / std::sqrt(p.Length * p.Width) * std::sin(m.X * Pi * x / p.Length) * std::sin(m.Y * Pi * y / p.Width);
}
ModalDynamics MakeModalDynamics(double mass, double omega, double damping_ratio, double dt) {
    if (!Positive(mass) || !Nonnegative(omega) || !Nonnegative(damping_ratio) || !Positive(dt) || dt * omega >= 2) throw std::invalid_argument("Invalid or unstable modal time step");
    const double damping = dt * damping_ratio * omega, denominator = 1 + damping;
    const ModalDynamics result{mass, omega, damping_ratio, std::pow(dt * omega, 2) / denominator, (1 - damping) / denominator, dt * dt / mass / denominator};
    if (!Positive(result.Compliance) || !std::isfinite(result.Stiffness) || !std::isfinite(result.Retention)) throw std::invalid_argument("Modal coefficients are outside FP64 range");
    return result;
}
double AdvanceMode(ModalDynamics m, double q, double previous, double force) { return q + (m.Retention * (q - previous) + m.Compliance * force - m.Stiffness * q); }
double PreviousMode(ModalDynamics m, double q, double velocity, double force, double dt) {
    return q - dt * velocity + dt * dt / 2 * (force / m.Mass - m.Omega * m.Omega * q - 2 * m.DampingRatio * m.Omega * velocity);
}
double ModalEnergy(ModalDynamics m, double q, double previous, double dt) {
    return m.Mass / 2 * (std::pow((q - previous) / dt, 2) + m.Omega * m.Omega * q * previous);
}
}
