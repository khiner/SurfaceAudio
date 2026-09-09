// SPDX-License-Identifier: GPL-3.0-or-later
// Adapted from SkAT-VG/SDT, commit 0509de418e7bebc8b37866b3b4458e0acc8cf1f4.
#include "Sdt.h"

#include <numbers>
#include <numeric>
#include <stdexcept>

namespace surface_audio::sdt {
namespace {
void UpdatePrevious(Body &body, size_t mode) { body.PreviousPosition[mode] = (body.Velocity[mode] - body.B0V[mode] * body.Position[mode]) / body.B1V[mode]; }

std::pair<double, double> NextState(const Body &body, size_t mode, double force) {
    const double position = body.PFromP[mode] * body.Position[mode] + body.PFromV[mode] * body.Velocity[mode] + body.B1[mode] * force;
    const double velocity = body.VFromP[mode] * body.Position[mode] + body.VFromV[mode] * body.Velocity[mode] + body.B0V[mode] * body.B1[mode] * force;
    if (!std::isfinite(position) || !std::isfinite(velocity) || std::abs(position) >= 10000) throw std::domain_error("SDT predicted mode state exceeds the affine contact domain");
    return {position, velocity};
}

} // namespace

ContactEnergyT<double> ContactEnergy(const Body &body) {
    ContactEnergyT<double> energy;
    for (size_t mode = 0; mode < body.Position.size(); ++mode) {
        const auto [position, velocity] = NextState(body, mode, body.Force[mode]);
        const double dp = body.B1[mode] * body.ForceGain[mode], dv = body.B0V[mode] * dp;
        const double mass = body.Mass[mode], stiffness = body.Stiffness[mode], gain = body.ContactGain[mode];
        energy.Free += 0.5 * (stiffness * position * position + mass * velocity * velocity) * gain;
        energy.Linear += (stiffness * position * dp + mass * velocity * dv) * gain;
        energy.Quadratic += 0.5 * (stiffness * dp * dp + mass * dv * dv) * gain;
    }
    return energy;
}

namespace {
ContactSample Advance(Body &body0, Body &body1, ContactState &state, double force) {
    for (const auto [body, direction] : {std::pair{&body0, 1.0}, std::pair{&body1, -1.0}}) {
        for (size_t mode = 0; mode < body->Position.size(); ++mode) NextState(*body, mode, body->Force[mode] + direction * force * body->ForceGain[mode]);
    }
    const auto first = ContactEnergy(body0), second = ContactEnergy(body1);
    if (!std::isfinite(first.Free + second.Free) || !std::isfinite(first.Linear - second.Linear) || !std::isfinite(first.Quadratic + second.Quadratic)) throw std::domain_error("SDT contact energy exceeds the numeric domain");
    force = LimitContactForce(ContactEnergyT<double>{first.Free + second.Free, first.Linear - second.Linear, first.Quadratic + second.Quadratic}, state.Energy, force);
    ApplyForce(body0, force);
    ApplyForce(body1, -force);
    StepBody(body0);
    StepBody(body1);
    return {force, Position(body0), Position(body1), Velocity(body0), Velocity(body1), Output(body0), Output(body1)};
}
} // namespace

Body MakeBody(std::span<const ModeParameters> modes, double sample_rate, double fragment_size) {
    if (modes.empty() || !std::isfinite(sample_rate) || sample_rate <= 0 || !std::isfinite(fragment_size) || fragment_size <= 0 || fragment_size > 1) throw std::invalid_argument("Invalid SDT body dimensions or sample rate");
    Body body;
    body.SampleRate = sample_rate;
    const double time_step = 1 / sample_rate, scale = std::sqrt(fragment_size);
    for (const auto &mode : modes) {
        if (!std::isfinite(mode.Frequency) || mode.Frequency < 0 || !std::isfinite(mode.Mass) || mode.Mass * fragment_size <= 1e-6 || !std::isfinite(mode.Decay) || mode.Decay < 0 || !std::isfinite(mode.ContactGain) || mode.ContactGain < 0 || !std::isfinite(mode.OutputGain)) throw std::invalid_argument("Invalid SDT mode");
        const double omega = 2 * std::numbers::pi * mode.Frequency;
        const double phase = omega * time_step / scale;
        if (phase >= std::acos(-0.9995)) throw std::invalid_argument("SDT mode exceeds impulse-invariance frequency limit");
        const double damping = mode.Decay > 0 ? 2 / (mode.Decay * scale) : 0;
        const double radius = std::exp(-damping * time_step), cosine = std::cos(phase);
        const double sinc = phase > 0 ? std::sin(phase) / phase : 1;
        body.Mass.push_back(mode.Mass * fragment_size);
        body.Stiffness.push_back(omega * omega * mode.Mass);
        body.B1.push_back(radius * sinc * time_step * time_step / body.Mass.back());
        body.A1.push_back(-2 * radius * cosine);
        body.A2.push_back(radius * radius);
        body.B0V.push_back(cosine / (sinc * time_step) - damping);
        body.B1V.push_back(-radius / (sinc * time_step));
        const double p_from_p = -body.A1.back() + body.A2.back() * body.B0V.back() / body.B1V.back();
        const double p_from_v = -body.A2.back() / body.B1V.back();
        body.PFromP.push_back(p_from_p);
        body.PFromV.push_back(p_from_v);
        body.VFromP.push_back(body.B0V.back() * p_from_p + body.B1V.back());
        body.VFromV.push_back(body.B0V.back() * p_from_v);
        body.ContactGain.push_back(mode.ContactGain);
        body.OutputGain.push_back(mode.OutputGain);
        body.GainSum += mode.ContactGain;
    }
    for (double gain : body.ContactGain) body.ForceGain.push_back(body.GainSum > 0 ? gain / body.GainSum : 1.0 / modes.size());
    body.Position.resize(modes.size());
    body.PreviousPosition.resize(modes.size());
    body.Velocity.resize(modes.size());
    body.Force.resize(modes.size());
    return body;
}

void SetPosition(Body &body, double position) {
    if (body.GainSum <= 0) return;
    if (!std::isfinite(position) || std::abs(position / body.GainSum) >= 10000) throw std::domain_error("SDT initial mode position exceeds the affine contact domain");
    for (size_t mode = 0; mode < body.Position.size(); ++mode) {
        body.Position[mode] = position / body.GainSum;
        UpdatePrevious(body, mode);
    }
}

void SetVelocity(Body &body, double velocity) {
    if (body.GainSum <= 0) return;
    if (!std::isfinite(velocity)) throw std::domain_error("SDT initial velocity is not finite");
    for (size_t mode = 0; mode < body.Position.size(); ++mode) {
        body.Velocity[mode] = velocity / body.GainSum;
        UpdatePrevious(body, mode);
    }
}

double Position(const Body &body) { return std::inner_product(body.Position.begin(), body.Position.end(), body.ContactGain.begin(), 0.); }
double Velocity(const Body &body) { return std::inner_product(body.Velocity.begin(), body.Velocity.end(), body.ContactGain.begin(), 0.); }
double Output(const Body &body) { return std::inner_product(body.Position.begin(), body.Position.end(), body.OutputGain.begin(), 0.); }

void ApplyForce(Body &body, double force) {
    if (!std::isnormal(force)) return;
    for (size_t mode = 0; mode < body.Position.size(); ++mode) body.Force[mode] += force * body.ForceGain[mode];
}

double PredictedEnergy(const Body &body, double contact_force) {
    double energy{};
    for (size_t mode = 0; mode < body.Position.size(); ++mode) {
        const auto [position, velocity] = NextState(body, mode, body.Force[mode] + contact_force * body.ForceGain[mode]);
        energy += 0.5 * (body.Stiffness[mode] * position * position + body.Mass[mode] * velocity * velocity) * body.ContactGain[mode];
    }
    return energy;
}

void StepBody(Body &body) {
    for (size_t mode = 0; mode < body.Position.size(); ++mode) {
        const auto [position, velocity] = NextState(body, mode, body.Force[mode]);
        body.Velocity[mode] = velocity;
        body.PreviousPosition[mode] = body.Position[mode];
        body.Position[mode] = position;
        body.Force[mode] = 0;
    }
}

ContactSample StepImpact(Body &body0, Body &body1, ContactState &state, ImpactParameters parameters, double external0, double external1) {
    ApplyForce(body0, external0);
    ApplyForce(body1, external1);
    const double compression = Position(body1) - Position(body0);
    if (compression <= 0) state.Energy = 0;
    return Advance(body0, body1, state, ImpactForce(parameters, compression, Velocity(body1) - Velocity(body0)));
}

ContactSample StepFriction(Body &body0, Body &body1, ContactState &state, FrictionParameters parameters, double noise_sample, double external0, double external1) {
    ApplyForce(body0, external0);
    ApplyForce(body1, external1);
    state.Energy = 0;
    return Advance(body0, body1, state, FrictionForce(parameters, state.Friction, Velocity(body1) - Velocity(body0), noise_sample, 1 / body0.SampleRate));
}
} // namespace surface_audio::sdt
