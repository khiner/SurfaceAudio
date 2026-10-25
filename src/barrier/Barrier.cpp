#include "Barrier.h"

#include <array>
#include <limits>
#include <numbers>

namespace surface_audio::barrier {
template<typename T> Parameters<T> MakeParameters(const Controls &c) {
    for (double x : {c.SampleRate, c.Mass, c.Stiffness, c.Damping, c.Barrier, c.Kappa, c.Exponent, c.Restitution, c.MaxVelocity, c.Regularization, c.Friction})
        if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite mass-barrier control");
    if (c.SampleRate <= 0 || c.Mass <= 0 || c.Stiffness < 0 || c.Damping < 0 || c.Kappa < 0 || c.Exponent < 1 || c.Exponent > 2 || c.Restitution < 0 || c.Restitution > 1 || c.MaxVelocity <= 0 || c.Regularization <= 0 || c.Friction < 0) throw std::invalid_argument("Invalid mass-barrier control");
    const double dt = 1 / c.SampleRate, sigma = c.Damping / (2 * c.Mass), omega2 = c.Stiffness / c.Mass;
    if (omega2 * dt * dt >= std::numbers::pi * std::numbers::pi) throw std::invalid_argument("Mass-barrier resonance must be below Nyquist");
    double difference{}, sum{}, b{}, response{}, stiffness = c.Stiffness, damping = c.Damping;
    if (c.ModalCorrection) {
        if (omega2 >= sigma * sigma) {
            const double radius = std::exp(-sigma * dt), sine = std::sin(std::sqrt(omega2 - sigma * sigma) * dt / 2);
            difference = std::pow(std::expm1(-sigma * dt), 2) + 4 * radius * sine * sine;
            sum = (1 + radius) * (1 + radius) - 4 * radius * sine * sine;
        } else {
            const double root = std::sqrt(sigma * sigma - omega2), slow = -omega2 / (sigma + root), fast = -sigma - root;
            difference = std::expm1(slow * dt) * std::expm1(fast * dt);
            sum = (1 + std::exp(slow * dt)) * (1 + std::exp(fast * dt));
        }
        b = 1 + std::exp(-2 * sigma * dt);
        response = dt * dt * sum / (4 * c.Mass);
        stiffness = 4 * c.Mass / (dt * dt) * difference / sum;
        damping = -4 * c.Mass / dt * std::expm1(-2 * sigma * dt) / sum;
    } else {
        const double a = c.Stiffness * dt * dt / (4 * c.Mass), loss = c.Damping * dt / (2 * c.Mass);
        difference = 4 * a / (1 + a + loss);
        b = (2 + 2 * a) / (1 + a + loss);
        response = dt * dt / (c.Mass * (1 + a + loss));
    }
    const double zeta = std::sqrt(std::numbers::pi) * std::tgamma(1 + 1 / (c.Exponent + 1)) / std::tgamma(.5 + 1 / (c.Exponent + 1));
    const double cap = .5 * c.Mass * (c.Exponent + 1) * std::pow(c.MaxVelocity, 1 - c.Exponent) * std::pow(2 * zeta / (6 * dt), c.Exponent + 1);
    Parameters<T> p{T(dt), T(c.Mass), T(stiffness), T(damping), T(c.Barrier), T(c.CapStiffness ? std::min(c.Kappa, cap) : c.Kappa), T(c.Exponent), T(c.Restitution), T(c.Regularization), T(c.Friction), T(difference), T(b), T(response), std::numeric_limits<T>::epsilon(), unsigned(c.Implicit)};
    for (T x : {p.Dt, p.Mass, p.Stiffness, p.Damping, p.Barrier, p.Kappa, p.Regularization, p.Friction, p.C, p.Difference, p.B})
        if (!std::isfinite(x)) throw std::invalid_argument("Mass-barrier coefficients exceed precision range");
    if (p.C <= 0 || p.Dt <= 0 || p.Mass <= 0) throw std::invalid_argument("Mass-barrier coefficients underflow");
    return p;
}

template<typename T> void Render(const Parameters<T> &p, State<T> &s, std::span<const Force<T>> input, std::span<Sample<T>> output) {
    if (input.size() != output.size()) throw std::invalid_argument("Mass-barrier block size mismatch");
    for (const auto &f : input)
        if (!std::isfinite(f.X) || !std::isfinite(f.Y)) throw std::invalid_argument("Nonfinite mass-barrier force");
    for (size_t n = 0; n < input.size(); ++n) {
        output[n] = Step(p, s, input[n]);
        if (!std::isfinite(output[n].Energy)) throw std::runtime_error("Nonfinite mass-barrier state");
    }
}
template Parameters<double> MakeParameters(const Controls &);
template Parameters<float> MakeParameters(const Controls &);
template void Render(const Parameters<double> &, State<double> &, std::span<const Force<double>>, std::span<Sample<double>>);
template void Render(const Parameters<float> &, State<float> &, std::span<const Force<float>>, std::span<Sample<float>>);

GpuSource CreateGpuSource(Gpu &gpu, std::span<const Parameters<float>> parameters, unsigned frames) {
    if (parameters.empty() || !frames || parameters.size() * uint64_t(frames) > UINT32_MAX) throw std::invalid_argument("Invalid mass-barrier GPU dimensions");
    std::vector<State<float>> states;
    states.reserve(parameters.size());
    for (const auto &p : parameters) states.push_back(MakeState(p));
    return {unsigned(parameters.size()), frames, Upload<Parameters<float>>(gpu, parameters), Upload<State<float>>(gpu, states), CreateBuffer(gpu, parameters.size() * frames * sizeof(Force<float>)), CreateBuffer(gpu, parameters.size() * frames * sizeof(Sample<float>)), CreateKernel(gpu, "MassBarrierSource")};
}
void EncodeSource(Gpu &gpu, const GpuSource &s) {
    const auto block = BatchUpload(gpu, std::array{s.Voices, s.Frames});
    const std::array bindings{GpuBinding{s.Config, 0}, GpuBinding{s.States, 1}, GpuBinding{s.Input, 2}, GpuBinding{s.Output, 3}, GpuBinding{block, 4}};
    DispatchGpu(gpu, s.Kernel, bindings, {s.Voices});
}
}
