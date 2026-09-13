#include "rough/Contact.h"
#include "rough/ContactGpu.h"
#include "rough/PlateContact.h"
#include "rough/PlateContactGpu.h"
#include "rough/Profile.h"
#include "core/Random.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <numeric>
#include <stdexcept>

using namespace surface_audio;
using namespace surface_audio::rough;
namespace {
void Check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
Model RigidSlider(double dt, double stiffness, double gravity = 0) {
    const BeamProperties bottom{.Length = .4, .MassPerLength = 1., .BendingStiffness = 100.};
    const BeamProperties top{.Length = .2, .MassPerLength = 1., .BendingStiffness = 100., .Boundary = BeamBoundary::Free};
    return MakeModel(MakeBeamSurface(bottom, 0, std::vector<double>(41), dt), MakeBeamSurface(top, 0, std::vector<double>(21), dt, gravity), dt, stiffness);
}

void InterpolationRules() {
    for (double x : {0., .2, 1., 1.1, 3.5, 5.2, 6.}) {
        const auto p = Interpolate(7, x);
        double sum{}, position{}, quadratic{};
        for (unsigned i = 0; i < 4; ++i) {
            Check(p.Node[i] < 7, "Interpolation stays inside profile");
            sum += p.Weight[i];
            position += p.Weight[i] * p.Node[i];
            quadratic += p.Weight[i] * p.Node[i] * p.Node[i];
        }
        Check(std::abs(sum - 1) < 1e-14 && std::abs(position - x) < 1e-14, "Interpolation preserves force and first moment");
        if (x >= 1 && x < 5) Check(std::abs(quadratic - x * x) < 1e-13, "Interior Hermite interpolation reproduces quadratic profiles");
    }
}

void ForceBalance() {
    constexpr double dt = 1e-6;
    const BeamProperties bottom{.Length = .4, .MassPerLength = 1., .BendingStiffness = 100., .Boundary = BeamBoundary::Free};
    const BeamProperties top{.Length = .2, .MassPerLength = 2., .BendingStiffness = 100., .Boundary = BeamBoundary::Free};
    std::vector<double> profile(21);
    for (unsigned i = 0; i < profile.size(); ++i) profile[i] = 1e-4 * std::exp(-std::pow((double(i) - 6) / 3, 2));
    const auto m = MakeModel(MakeBeamSurface(bottom, 2, std::vector<double>(41), dt), MakeBeamSurface(top, 2, profile, dt), dt, 1e7);
    for (double offset : {-.033, 0., .097, .233, .367}) {
        auto s = MakeState(m);
        const auto r = Step(m, s, ContactMethod::Penalty, offset, 0);
        const double bottom_force = s.Force[0] * std::sqrt(bottom.Length), top_force = s.Force[4] * std::sqrt(top.Length);
        const double bottom_moment = s.Force[1] * std::sqrt(std::pow(bottom.Length, 3) / 12);
        const double top_moment = s.Force[5] * std::sqrt(std::pow(top.Length, 3) / 12);
        const double moment = bottom_moment + bottom.Length / 2 * bottom_force + top_moment + (offset + top.Length / 2) * top_force;
        Check(r.Contacts > 0 && r.NormalForce > 0, "Penetration produces compressive contact");
        Check(std::abs(bottom_force + top_force) < 1e-12, "Equal and opposite total world force");
        Check(std::abs(moment) < 1e-12, "Contact reactions conserve world moment including clipped overlap");
        Check(std::abs(top_force - r.NormalForce) < 1e-12, "Reported force includes both contact passes");
    }
}

void SupportedSlider() {
    const auto m = RigidSlider(1e-5, 1e8, -9.81);
    auto s = MakeState(m);
    for (unsigned n = 0; n < 30; ++n) {
        const auto r = Step(m, s, ContactMethod::Multiplier, .103, 0, 1e-14, 2000);
        Check(r.Converged && r.Residual <= 1e-14 && r.MaxPenetration <= 1e-14, "Forward multiplier contact complementarity");
        if (n > 3) Check(std::abs(r.NormalForce - .2 * 9.81) < 5e-4, "Normal reaction balances physical slider weight");
    }
    auto separated = MakeState(m);
    const auto r = Step(m, separated, ContactMethod::Multiplier, .1, .01);
    Check(r.Converged && r.NormalForce == 0, "Open gap has no multiplier force");
    const double falling = separated.Displacement[0] / std::sqrt(.2);
    Check(std::abs(falling + 9.81 * 1e-10 / 2) < 1e-20, "Separated slider follows gravity without contact impulse");
}

void FailureReporting() {
    const auto m = RigidSlider(1e-5, 1e8, -9.81);
    auto s = MakeState(m);
    const auto displacement = s.Displacement, previous = s.Previous;
    const auto r = Step(m, s, ContactMethod::Multiplier, .103, 0, 1e-25, 1);
    Check(!r.Converged && r.Residual > 1e-25, "Unresolved multiplier constraints are reported");
    Check(s.Displacement == displacement && s.Previous == previous, "Unconverged contact preserves the physical state");
    const auto retry = Step(m, s, ContactMethod::Multiplier, .103, 0, 1e-14, 2000);
    auto reference = MakeState(m);
    const auto expected = Step(m, reference, ContactMethod::Multiplier, .103, 0, 1e-14, 2000);
    Check(retry.Converged && expected.Converged && s.Displacement == reference.Displacement && s.Previous == reference.Previous, "A failed step can be retried without restoring scratch buffers");
}

double Potential(const Model &m, std::span<const double> q, double offset, double separation) {
    // Integrate displaced surface heights independently of the contact Jacobian and force scatter.
    std::array<std::vector<double>, 2> heights{m.Bottom.Height, m.Top.Height};
    unsigned first{};
    for (unsigned body = 0; body < 2; ++body) {
        const auto &surface = body ? m.Top : m.Bottom;
        for (unsigned k = 0; k < surface.Modes.size(); ++k)
            for (unsigned n = 0; n < heights[body].size(); ++n)
                heights[body][n] += (body ? -1 : 1) * q[first + k] * surface.Shapes[k * heights[body].size() + n];
        first += surface.Modes.size();
    }
    double energy{};
    for (unsigned body = 0; body < 2; ++body) {
        const auto &slave = body ? m.Top : m.Bottom, &master = body ? m.Bottom : m.Top;
        for (unsigned n = 0; n < heights[body].size(); ++n) {
            const double x = (n * slave.Step + (body ? offset : -offset)) / master.Step;
            if (x < 0 || x > heights[1 - body].size() - 1) continue;
            const auto p = Interpolate(unsigned(heights[1 - body].size()), x);
            double h = heights[body][n] - separation;
            for (unsigned j = 0; j < 4; ++j) h += p.Weight[j] * heights[1 - body][p.Node[j]];
            const double weight = slave.Step * (n == 0 || n + 1 == heights[body].size() ? .5 : 1);
            energy += .5 * m.Penalty * weight * std::pow(std::max(0., h), 2);
        }
    }
    return energy;
}

Model SlidingModel(double dt) {
    const BeamProperties bottom{.Length = .08, .MassPerLength = 1, .BendingStiffness = .02, .DampingRatio = .02};
    const BeamProperties top{.Length = .02, .MassPerLength = 2, .BendingStiffness = .005, .DampingRatio = .02, .Boundary = BeamBoundary::Free};
    std::vector<double> lower(257), upper(65);
    for (unsigned n = 0; n < lower.size(); ++n) lower[n] = 1e-5 * std::sin(2 * std::numbers::pi * n * bottom.Length / (lower.size() - 1) / .004);
    for (unsigned n = 0; n < upper.size(); ++n) upper[n] = 3e-5 - .1 * std::pow(n * top.Length / (upper.size() - 1) - .01, 2);
    return MakeModel(MakeBeamSurface(bottom, 6, lower, dt), MakeBeamSurface(top, 2, upper, dt, -9.81), dt, 1e10);
}

void PotentialGradient() {
    const auto m = SlidingModel(1e-6);
    auto s = MakeState(m);
    for (unsigned k = 0; k < s.Displacement.size(); ++k) s.Displacement[k] = 1e-8 * std::sin(double(k + 1));
    const auto q = s.Displacement;
    constexpr double offset = .025413, separation = 2e-5, epsilon = 1e-11;
    const auto contact = Step(m, s, ContactMethod::Penalty, offset, separation);
    Check(std::abs(contact.ElasticEnergy - Potential(m, q, offset, separation)) < 1e-12, "Independent surface energy matches assembled penalty energy");
    for (unsigned k = 0; k < q.size(); ++k) {
        auto a = q, b = q;
        a[k] += epsilon;
        b[k] -= epsilon;
        const double derivative = (Potential(m, a, offset, separation) - Potential(m, b, offset, separation)) / (2 * epsilon);
        const double gravity = k < m.Bottom.Modes.size() ? m.Bottom.Gravity[k] : m.Top.Gravity[k - m.Bottom.Modes.size()];
        Check(std::abs(derivative + s.Force[k] - gravity) < 1e-6 * std::max(1., std::abs(derivative)), "Modal force matches the potential gradient");
    }
}

void Impact(ContactMethod method, double dt, double penalty) {
    constexpr double mass = .2, velocity = -.5, separation = .001;
    const Surface bottom{.Step = .1, .Height = {0, 0}, .Shapes = {}, .Gravity = {}, .Modes = {}};
    const Surface top{.Step = .1, .Height = {0, 0}, .Shapes = {1, 1}, .Gravity = {0}, .Modes = {MakeModalDynamics(mass, 0, 0, dt)}};
    const auto m = MakeModel(bottom, top, dt, penalty);
    auto s = MakeState(m, {}, std::array{velocity});
    double impulse{}, work{};
    for (unsigned n = 0; n < unsigned(std::llround(.006 / dt)); ++n) {
        const double previous = s.Previous[0];
        const auto result = Step(m, s, method, 0, separation, 1e-14);
        Check(result.Converged, "Single impact converges");
        impulse += result.NormalForce * dt;
        work += result.NormalForce * (s.Displacement[0] - previous) / 2;
    }
    const double final_velocity = (s.Displacement[0] - s.Previous[0]) / dt;
    const double restitution = -final_velocity / velocity, energy_change = .5 * mass * (final_velocity * final_velocity - velocity * velocity);
    std::cout << "impact method=" << (method == ContactMethod::Penalty ? "penalty" : "multiplier") << " dt=" << dt << " stiffness=" << penalty
              << " restitution=" << restitution << " impulse=" << impulse << " contact_work=" << work << '\n';
    Check(std::abs(impulse - mass * (final_velocity - velocity)) < 1e-10, "Impact impulse balances momentum");
    Check(std::abs(energy_change - work) < 1e-10, "Impact work balances kinetic energy");
    Check(method == ContactMethod::Penalty ? std::abs(restitution - 1) < .002 : std::abs(restitution) < 1e-8,
          "Penalty impact rebounds elastically; forward displacement constraints stop rigid normal approach");
}

void PlateReactions() {
    constexpr double dt = 1e-6;
    const RigidPlateProperties bottom{.4, .3, 2, .015, .02666666666666667}, top{.08, .06, .2, .00006, .00010666666666666667};
    const PlateGrid bg{21, 16, .02, .02, 0, 0}, tg{9, 7, .01, .01, 0, 0};
    std::vector<double> height(tg.XNodes * tg.YNodes);
    for (unsigned y = 0; y < tg.YNodes; ++y)
        for (unsigned x = 0; x < tg.XNodes; ++x) height[y * tg.XNodes + x] = 1e-5 + 4e-6 * std::cos(x * .7 + y * .3);
    for (bool two_pass : {false, true}) {
        const auto model = MakePlateContact(MakeRigidPlateSurface(bottom, bg, std::vector<double>(336), dt), MakeRigidPlateSurface(top, tg, height, dt), dt, {1e10, 1e8}, two_pass);
        for (const auto offset : {std::array{.103, .087}, std::array{-.03, .02}, std::array{.373, .273}}) {
            auto state = MakePlateState(model);
            const auto result = StepPlateContact(model, state, offset[0], offset[1], 0);
            Check(result.Solve.Converged && result.NormalForce > 0, "Penetrating plate contact converges");
            const double fb = state.Force[0] * std::sqrt(bottom.Length * bottom.Width), ft = state.Force[3] * std::sqrt(top.Length * top.Width);
            const double mx = state.Force[1] * std::sqrt(bottom.Length * std::pow(bottom.Width, 3) / 12) + bottom.Width / 2 * fb +
                state.Force[4] * std::sqrt(top.Length * std::pow(top.Width, 3) / 12) + (offset[1] + top.Width / 2) * ft;
            const double my = state.Force[2] * std::sqrt(bottom.Width * std::pow(bottom.Length, 3) / 12) - bottom.Length / 2 * fb +
                state.Force[5] * std::sqrt(top.Width * std::pow(top.Length, 3) / 12) - (offset[0] + top.Length / 2) * ft;
            Check(std::abs(fb + ft) < 1e-12 && std::abs(ft - result.NormalForce) < 1e-12, "One-pass and two-pass bilinear contact conserve total force");
            Check(std::abs(mx) < 1e-12 && std::abs(my) < 1e-12, "Bilinear reactions conserve both world moments including clipped patches");
            for (unsigned k = 0; k < 6; ++k) Check(std::abs(state.Velocity[k] - state.Damping.Velocity[k]) < 1e-12, "Damping and modal update use the same centered velocity");
        }
        auto open = MakePlateState(model);
        const auto result = StepPlateContact(model, open, 2, 2, 0);
        Check(result.Solve.Converged && result.Contacts == 0 && result.NormalForce == 0, "Disjoint plate patches have no contact");
    }
}

void BeamGpu(Gpu &gpu) {
    constexpr double dt = 1e-6;
    constexpr unsigned steps = 513;
    const BeamProperties bottom{.Length = .4, .MassPerLength = 1., .BendingStiffness = 100., .DampingRatio = .01};
    const BeamProperties top{.Length = .2, .MassPerLength = 1., .BendingStiffness = 100., .Boundary = BeamBoundary::Free};
    std::vector<double> bottom_height(81), top_height(41);
    for (unsigned i = 0; i < bottom_height.size(); ++i) bottom_height[i] = 1e-5 * std::sin(i * .71) + 3e-6 * std::cos(i * 1.9);
    for (unsigned i = 0; i < top_height.size(); ++i) top_height[i] = 3e-6 * std::cos(i * .43);
    const auto m = MakeModel(MakeBeamSurface(bottom, 5, bottom_height, dt), MakeBeamSurface(top, 2, top_height, dt, -9.81), dt, 1e9);
    std::vector<double> receiver(9);
    for (unsigned k = 0; k < 5; ++k) receiver[k] = m.Bottom.Shapes[k * bottom_height.size() + 37];
    for (const auto [separation, speed, offset] : {std::array{1e-4, .13, .103}, std::array{1e-5, .13, .103}, std::array{1e-5, 20., -.191}}) {
        auto cpu = MakeState(m);
        const auto device = MakeGpuContact(gpu, m, cpu, receiver);
        const auto full = MakeGpuContact(gpu, m, cpu, receiver, 1, 1);
        const std::array runs{GpuContactRun{&device, steps, offset, separation, speed}, GpuContactRun{&full, steps, offset, separation, speed}};
        const auto traces = RenderGpu(gpu, runs);
        const auto &actual = traces[0], &reference = traces[1];
        const auto serial = MakeGpuContact(gpu, m, cpu, receiver);
        Check(actual == RenderGpu(gpu, serial, steps, offset, separation, speed), "Batched and serial contact trajectories agree exactly");
        Check(actual == reference, "Conservative contact screening preserves every output, modal state and event exactly");
        double error{}, norm{}, force_error{}, force_norm{}, state_error{}, square_error{}, square_norm{};
        std::array<double, 2> contact_work{};
        for (unsigned n = 0; n < steps; ++n) {
            const auto old_previous = cpu.Previous;
            const auto result = Step(m, cpu, ContactMethod::Penalty, offset + n * dt * speed, separation);
            double expected{}, square{};
            for (unsigned k = 0; k < receiver.size(); ++k) {
                const double v = (cpu.Displacement[k] - old_previous[k]) / (2 * dt);
                expected += receiver[k] * v;
                contact_work[k >= m.Bottom.Modes.size()] += (cpu.Force[k] - (k < m.Bottom.Modes.size() ? m.Bottom.Gravity[k] : m.Top.Gravity[k - m.Bottom.Modes.size()])) * v * dt;
                if (k < m.Bottom.Modes.size()) square += v * v / bottom.Length;
            }
            square_error += std::pow(actual.MeanSquareVelocity[n] - square, 2);
            square_norm += square * square;
            error += std::pow(actual.Velocity[n] - expected, 2);
            norm += expected * expected;
            force_error += std::pow(actual.Force[n] - result.NormalForce, 2);
            force_norm += result.NormalForce * result.NormalForce;
        }
        for (unsigned k = 0; k < receiver.size(); ++k) state_error = std::max(state_error, std::abs(actual.Displacement[k] - cpu.Displacement[k]));
        const double relative = std::sqrt(error / std::max(1e-40, norm)), force_relative = std::sqrt(force_error / std::max(1e-40, force_norm));
        std::cout << "GPU separation=" << separation << " velocity RMS=" << relative << " force RMS=" << force_relative << " state=" << state_error << '\n';
        Check(relative < 1e-5 && force_relative < 1e-5 && state_error < 1e-10, "GPU full contact trajectory matches FP64 reference");
        Check(square_error < 1e-10 * std::max(1e-40, square_norm), "GPU spatial mean square matches normalized modal velocities");
        for (unsigned body = 0; body < 2; ++body)
            Check(std::abs(contact_work[body] - actual.Events[body].Work) < 1e-10, "Each flexible body's nodal work equals its modal contact work");
    }
}

bool Near(std::span<const double> actual, std::span<const double> expected) {
    if (actual.size() != expected.size()) return false;
    double error{}, norm{};
    for (unsigned k = 0; k < actual.size(); ++k) {
        error += std::pow(actual[k] - expected[k], 2);
        norm += expected[k] * expected[k];
    }
    return error <= 1e-14 * std::max(1e-40, norm);
}
void PlateGpu(Gpu &gpu) {
    constexpr double dt = 1e-6;
    const PlateProperties plate{.2, .12, 5, .5, .01};
    const RigidPlateProperties slider{.05, .04, .02, 2.8e-6, 4.3e-6};
    std::vector<double> height(273), top_height(30);
    for (unsigned n = 0; n < height.size(); ++n) height[n] = 3e-6 * std::sin(n * .37);
    for (unsigned n = 0; n < top_height.size(); ++n) top_height[n] = 1e-6 * std::cos(n * .73);
    auto model = MakePlateContact(MakePlateSurface(plate, 8, {21, 13, .01, .01, 0, 0}, height, dt),
        MakeRigidPlateSurface(slider, {6, 5, .01, .01, 0, 0}, top_height, dt, -9.81), dt, {1e11, 3e10});
    auto full = MakePlateContactGpu(gpu, model, 1, false), cached = MakePlateContactGpu(gpu, model, 4096, true);
    for (bool two_pass : {false, true}) {
        model.TwoPass = two_pass;
        for (const auto motion : {std::array{.0713, .0437, .7, -.3}, std::array{-.032, .02, 2., 0.}, std::array{1., 1., 0., 0.}}) {
            auto cpu = MakePlateState(model), actual = MakePlateState(model), reused = MakePlateState(model);
            for (unsigned n = 0; n < 32; ++n) {
                const double x = motion[0] + n * dt * motion[2], y = motion[1] + n * dt * motion[3];
                PreparePlateContact(model, cpu, x, y, -1e-6);
                PreparePlateContactGpu(gpu, full, model, actual, x, y, -1e-6);
                PreparePlateContactGpu(gpu, cached, model, reused, x, y, -1e-6);
                for (auto *state : {&actual, &reused}) {
                    Check(Near(state->Jacobian, cpu.Jacobian) && Near(state->Penetration, cpu.Penetration) && Near(state->Area, cpu.Area),
                          "Full and cached GPU detection match ordered CPU contacts through motion and pass changes");
                    Check(AdvancePlateContact(model, *state).Solve.Converged, "GPU-assembled plate contact converges");
                }
                Check(AdvancePlateContact(model, cpu).Solve.Converged, "CPU plate contact converges");
                for (const auto *state : {&actual, &reused})
                    Check(Near(state->Displacement, cpu.Displacement) && Near(state->Previous, cpu.Previous) && Near(state->Velocity, cpu.Velocity) && Near(state->Force, cpu.Force),
                          "Full and cached GPU assembly preserve the complete coupled trajectory");
                Check(reused.Displacement == cpu.Displacement && reused.Previous == cpu.Previous && reused.Force == cpu.Force,
                      "Reused candidates preserve the CPU trajectory exactly");
            }
        }
    }
    Check(full.Capacity > 1, "GPU detection grows storage without dropping contacts");
}

void DenseContact(Gpu &gpu) {
    constexpr double dt = 1e-6, length = .2, gravity = -9.81, penalty = 1e9;
    const BeamProperties bottom{.Length = length, .MassPerLength = 1, .BendingStiffness = 100};
    const BeamProperties top{.Length = length, .MassPerLength = 1, .BendingStiffness = 100, .Boundary = BeamBoundary::Free};
    const std::vector<double> profile(1025);
    const auto model = MakeModel(MakeBeamSurface(bottom, 0, profile, dt), MakeBeamSurface(top, 0, profile, dt, gravity), dt, penalty);
    auto state = MakeState(model);
    state.Displacement[0] = gravity / (2 * penalty) * std::sqrt(length);
    state.Previous = state.Displacement;
    const std::array receiver{1 / std::sqrt(length), 0.};
    const auto a = MakeGpuContact(gpu, model, state, receiver);
    const auto b = MakeGpuContact(gpu, model, state, receiver, 1, 1);
    const std::array runs{GpuContactRun{&a, 32, 0, 0, 0}, GpuContactRun{&b, 32, 0, 0, 0}};
    const auto traces = RenderGpu(gpu, runs);
    const auto split = MakeGpuContact(gpu, model, state, receiver);
    const auto prefix = RenderGpu(gpu, split, 13, 0, 0, 0), suffix = RenderGpu(gpu, split, 19, 0, 0, 0);
    Check(std::equal(prefix.Velocity.begin(), prefix.Velocity.end(), traces[0].Velocity.begin()) &&
          std::equal(suffix.Velocity.begin(), suffix.Velocity.end(), traces[0].Velocity.begin() + 13) &&
          suffix.Displacement == traces[0].Displacement && suffix.Previous == traces[0].Previous && suffix.Events == traces[0].Events,
          "Chunk boundaries preserve modal state, output and accumulated contact statistics");
    Check(traces[0].Velocity == traces[1].Velocity && traces[0].Force == traces[1].Force && traces[0].Events == traces[1].Events, "Dense contact sorting is deterministic beyond one threadgroup's lane count");
    for (float force : traces[0].Force) Check(std::abs(force / (-gravity * length) - 1) < 1e-6, "Dense contact projection preserves the loaded equilibrium");
    for (const auto &events : traces[0].Events)
        Check(events.Completed == 0 && events.LeftCensored == profile.size() && events.RightCensored == profile.size(), "Dense contact retains every loaded material node");
}

void ScreeningCancellation(Gpu &gpu) {
    constexpr double dt = 1e-8;
    const BeamProperties beam{.Length = .2, .MassPerLength = 1, .BendingStiffness = 1};
    const std::vector<double> profile(65);
    auto model = MakeModel(MakeBeamSurface(beam, 128, profile, dt), MakeBeamSurface(beam, 128, profile, dt), dt, 1e4);
    for (auto *surface : {&model.Bottom, &model.Top})
        for (unsigned k = 0; k < surface->Modes.size(); ++k) {
            surface->Modes[k] = {.Stiffness = 0, .Retention = 1, .Compliance = 1e-16};
            for (unsigned n = 0; n < profile.size(); ++n) surface->Shapes[k * profile.size() + n] = 1 + .1 * std::sin(double(n));
        }
    const std::vector<double> receiver(256, 1);
    for (double magnitude : {1e-7, 1e3}) {
        auto state = MakeState(model);
        for (unsigned k = 0; k < 256; ++k) {
            state.Displacement[k] = (k % 2 ? -1 : 1) * magnitude;
            state.Previous[k] = state.Displacement[k] + (k % 3 ? 1 : -1) * magnitude * 1e-10;
        }
        const auto full = MakeGpuContact(gpu, model, state, receiver, 3, 1);
        const auto expected = RenderGpu(gpu, full, 513, .00313, magnitude * 1e-8, .7);
        const auto screened = MakeGpuContact(gpu, model, state, receiver, 3, 32);
        const auto actual = RenderGpu(gpu, screened, 513, .00313, magnitude * 1e-8, .7);
        Check(actual == expected, "Screening preserves 256-mode trajectories with cancellation across displacement scales");
    }
}

void DirectConvolution(Gpu &gpu, unsigned nodes) {
    constexpr double spacing = 5e-6, rms = 6.02e-6;
    constexpr double correlation = 25e-6;
    constexpr uint64_t seed = 94751;
    const auto actual = GaussianProfile(gpu, nodes, spacing, rms, correlation, seed);
    const double scale = std::sqrt(2 / std::sqrt(std::numbers::pi) * (nodes - 1) * spacing / nodes / correlation);
    std::vector<double> noise(nodes), filter(nodes);
    for (unsigned n = 0; n < nodes; ++n) {
        auto random = MakeIndexedRandom(seed, n);
        noise[n] = rms * Normal(random);
        filter[n] = std::exp(-2 * std::pow((n - .5 * (nodes - 1)) * spacing / correlation, 2));
    }
    double error{}, norm{};
    for (unsigned n = 0; n < nodes; ++n) {
        double expected{};
        for (unsigned j = 0; j < nodes; ++j) expected += noise[j] * filter[(n + nodes - j) % nodes] * scale;
        error += std::pow(actual[n] - expected, 2);
        norm += expected * expected;
    }
    Check(error < 1e-10 * norm, "GPU filter matches direct published circular convolution");
    Check(actual == GaussianProfile(gpu, nodes, spacing, rms, correlation, seed), "Profile seed reproduces the same samples");
}

void SurfaceConvolution(Gpu &gpu) {
    constexpr unsigned nx = 9, ny = 8;
    constexpr double dx = 5e-6, dy = 7e-6, cx = 12e-6, cy = 21e-6, rms = 17e-6;
    constexpr uint64_t seed = 29713;
    const auto actual = GaussianSurface(gpu, nx, ny, dx, dy, rms, cx, cy, seed);
    std::vector<double> noise(nx * ny);
    for (unsigned n = 0; n < noise.size(); ++n) {
        auto random = MakeIndexedRandom(seed, n);
        noise[n] = rms * Normal(random);
    }
    const double scale = 2 / std::sqrt(std::numbers::pi) * std::sqrt(dx * dy / cx / cy);
    double error{}, norm{};
    for (unsigned y = 0; y < ny; ++y)
        for (unsigned x = 0; x < nx; ++x) {
            double expected{};
            for (unsigned j = 0; j < ny; ++j)
                for (unsigned i = 0; i < nx; ++i) {
                    const double rx = (i - .5 * (nx - 1)) * dx / cx, ry = (j - .5 * (ny - 1)) * dy / cy;
                    expected += scale * std::exp(-2 * (rx * rx + ry * ry)) * noise[((y + ny - j) % ny) * nx + (x + nx - i) % nx];
                }
            error += std::pow(actual[y * nx + x] - expected, 2);
            norm += expected * expected;
        }
    Check(error < 1e-10 * norm, "Separable GPU surface matches direct thesis 2D circular convolution");
}
}
int main() {
    try {
        InterpolationRules();
        ForceBalance();
        SupportedSlider();
        FailureReporting();
        PotentialGradient();
        for (auto method : {ContactMethod::Penalty, ContactMethod::Multiplier}) Impact(method, 2.5e-6, 1e7);
        PlateReactions();
        auto gpu = CreateGpu();
        BeamGpu(gpu);
        PlateGpu(gpu);
        DenseContact(gpu);
        ScreeningCancellation(gpu);
        for (unsigned nodes : {64u, 65u}) DirectConvolution(gpu, nodes);
        SurfaceConvolution(gpu);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
