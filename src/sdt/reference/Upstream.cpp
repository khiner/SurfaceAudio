// SPDX-License-Identifier: GPL-3.0-or-later
// Numerical reference copied from SDT 0509de418e7bebc8b37866b3b4458e0acc8cf1f4. See ../NOTICE.md.
#include "Upstream.h"

namespace surface_audio::sdt::reference {
namespace {
double NextPosition(const Body &body, size_t mode, double force) { return std::clamp(body.B1[mode] * force - body.A1[mode] * body.Position[mode] - body.A2[mode] * body.PreviousPosition[mode], -10000.0, 10000.0); }
double DistributedForce(const Body &body, size_t mode, double force) { return body.GainSum > 0 ? force * body.ContactGain[mode] / body.GainSum : force / body.Position.size(); }
} // namespace
void ApplyForce(Body &body, double force) {
    if (!std::isnormal(force)) return;
    for (size_t mode = 0; mode < body.Position.size(); ++mode) body.Force[mode] += DistributedForce(body, mode, force);
}
void StepBody(Body &body) {
    for (size_t mode = 0; mode < body.Position.size(); ++mode) {
        const double position = NextPosition(body, mode, body.Force[mode]);
        body.Velocity[mode] = body.B0V[mode] * position + body.B1V[mode] * body.Position[mode];
        body.PreviousPosition[mode] = body.Position[mode];
        body.Position[mode] = position;
        body.Force[mode] = 0;
    }
}
namespace {
ContactSample Advance(Body &body0, Body &body1, ContactState &state, double force) {
    force = LimitForce(body0, body1, state.Energy, force);
    reference::ApplyForce(body0, force);
    reference::ApplyForce(body1, -force);
    reference::StepBody(body0);
    reference::StepBody(body1);
    return {force, Position(body0), Position(body1), Velocity(body0), Velocity(body1), Output(body0), Output(body1)};
}
} // namespace

double PredictedEnergy(const Body &body, double contact_force) {
    double energy{};
    if (!std::isnormal(contact_force)) contact_force = 0;
    for (size_t mode = 0; mode < body.Position.size(); ++mode) {
        const double position = NextPosition(body, mode, body.Force[mode] + DistributedForce(body, mode, contact_force));
        const double velocity = body.B0V[mode] * position + body.B1V[mode] * body.Position[mode];
        energy += 0.5 * (body.Stiffness[mode] * position * position + body.Mass[mode] * velocity * velocity) * body.ContactGain[mode];
    }
    return energy;
}

double LimitForce(const Body &body0, const Body &body1, double &energy, double force, double tolerance) {
    const double available = reference::PredictedEnergy(body0, 0) + reference::PredictedEnergy(body1, 0) + energy;
    double excess = reference::PredictedEnergy(body0, force) + reference::PredictedEnergy(body1, -force) - available;
    if (excess > 0) {
        double lower = 0, upper = force;
        for (unsigned iteration = 0; (excess > 0 || excess < -tolerance * available) && iteration < 50; ++iteration) {
            force = (lower + upper) * 0.5;
            excess = reference::PredictedEnergy(body0, force) + reference::PredictedEnergy(body1, -force) - available;
            if (excess < 0) lower = force;
            else upper = force;
        }
    }
    energy = -excess;
    return force;
}

ContactSample StepImpact(Body &body0, Body &body1, ContactState &state, ImpactParameters parameters, double external0, double external1) {
    reference::ApplyForce(body0, external0);
    reference::ApplyForce(body1, external1);
    const double compression = Position(body1) - Position(body0);
    if (compression <= 0) state.Energy = 0;
    const double force = compression > 0 ? parameters.Stiffness * std::pow(compression, parameters.Shape) * (1 + parameters.Dissipation * (Velocity(body1) - Velocity(body0))) : 0;
    return Advance(body0, body1, state, force);
}

ContactSample StepFriction(Body &body0, Body &body1, ContactState &state, FrictionParameters parameters, double noise_sample, double external0, double external1) {
    reference::ApplyForce(body0, external0);
    reference::ApplyForce(body1, external1);
    state.Energy = 0;
    return Advance(body0, body1, state, FrictionForce(parameters, state.Friction, Velocity(body1) - Velocity(body0), noise_sample, 1 / body0.SampleRate));
}
} // namespace surface_audio::sdt::reference
