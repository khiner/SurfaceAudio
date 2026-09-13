#include "Contact.h"
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>

namespace surface_audio::rough {
namespace {
unsigned Modes(const Model &m) { return unsigned(m.Bottom.Modes.size() + m.Top.Modes.size()); }
const ModalDynamics &Mode(const Model &m, unsigned k) { return k < m.Bottom.Modes.size() ? m.Bottom.Modes[k] : m.Top.Modes[k - m.Bottom.Modes.size()]; }
double Gravity(const Model &m, unsigned k) { return k < m.Bottom.Modes.size() ? m.Bottom.Gravity[k] : m.Top.Gravity[k - m.Bottom.Modes.size()]; }
double Dot(const double *a, const double *b, unsigned count) { return cblas_ddot(int(count), a, 1, b, 1); }
double Quadrature(const Surface &s, unsigned node) { return s.Step * (node == 0 || node + 1 == s.Height.size() ? .5 : 1); }
double Sample(const Interpolation &p, const double *values) {
    double result{};
    for (unsigned i = 0; i < 4; ++i) result += p.Weight[i] * values[p.Node[i]];
    return result;
}
void Validate(const Model &m, const State &s) {
    const unsigned modes = Modes(m);
    if (!modes || s.Displacement.size() != modes || s.Previous.size() != modes || s.Next.size() != modes || s.Force.size() != modes)
        throw std::invalid_argument("Contact state dimensions do not match model");
}
}
Surface MakeBeamSurface(BeamProperties p, unsigned bending_modes, std::span<const double> heights, double dt, double gravity) {
    if (heights.size() < 2 || heights.size() > INT_MAX || !std::isfinite(gravity)) throw std::invalid_argument("Invalid beam surface grid or gravity");
    for (double h : heights)
        if (!std::isfinite(h)) throw std::invalid_argument("Nonfinite beam profile");
    const auto modes = MakeBeamModes(p, bending_modes);
    Surface result{.Step = p.Length / (heights.size() - 1), .Height = {heights.begin(), heights.end()}, .Shapes = std::vector<double>(heights.size() * modes.size()), .Gravity = std::vector<double>(modes.size()), .Modes = {}};
    result.Modes.reserve(modes.size());
    for (unsigned k = 0; k < modes.size(); ++k) {
        const auto &mode = modes[k];
        result.Modes.push_back(MakeModalDynamics(p.MassPerLength, mode.Omega, p.DampingRatio, dt));
        for (unsigned n = 0; n < heights.size(); ++n) result.Shapes[k * heights.size() + n] = BeamShape(p, mode, p.Length * n / (heights.size() - 1));
        const double integral = mode.Kind == 1 ? std::sqrt(p.Length) : mode.Kind == 0 ? mode.Scale * (1 - std::cos(mode.WaveNumber * p.Length)) / mode.WaveNumber :
                                                                                        0;
        result.Gravity[k] = p.MassPerLength * gravity * integral;
    }
    return result;
}
Model MakeModel(Surface bottom, Surface top, double dt, double penalty) {
    if (!std::isfinite(dt) || dt <= 0 || !std::isfinite(penalty) || penalty <= 0) throw std::invalid_argument("Invalid contact time step or stiffness");
    for (const auto *s : {&bottom, &top}) {
        if (!std::isfinite(s->Step) || s->Step <= 0 || s->Height.size() < 2 || s->Shapes.size() != s->Height.size() * s->Modes.size() || s->Gravity.size() != s->Modes.size()) throw std::invalid_argument("Invalid contact surface");
        for (const auto &mode : s->Modes)
            if (mode != MakeModalDynamics(mode.Mass, mode.Omega, mode.DampingRatio, dt)) throw std::invalid_argument("Surface integration coefficients use a different time step");
    }
    if (bottom.Modes.size() + top.Modes.size() == 0 || bottom.Modes.size() + top.Modes.size() > INT_MAX) throw std::invalid_argument("Invalid contact mode count");
    return {std::move(bottom), std::move(top), dt, penalty};
}
State MakeState(const Model &m, std::span<const double> displacement, std::span<const double> velocity) {
    const unsigned modes = Modes(m);
    if ((!displacement.empty() && displacement.size() != modes) || (!velocity.empty() && velocity.size() != modes)) throw std::invalid_argument("Invalid initial contact state");
    State s{std::vector<double>(modes), std::vector<double>(modes), std::vector<double>(modes), std::vector<double>(modes), {}, {}, {}, {}};
    for (unsigned k = 0; k < modes; ++k) {
        const double q = displacement.empty() ? 0 : displacement[k], v = velocity.empty() ? 0 : velocity[k];
        if (!std::isfinite(q) || !std::isfinite(v)) throw std::invalid_argument("Nonfinite initial modal state");
        s.Displacement[k] = q;
        s.Previous[k] = PreviousMode(Mode(m, k), q, v, Gravity(m, k), m.TimeStep);
    }
    return s;
}
Interpolation Interpolate(unsigned nodes, double x) {
    if (nodes < 2 || !std::isfinite(x) || x < 0 || x > nodes - 1) throw std::invalid_argument("Contact coordinate outside profile");
    const unsigned left = std::min(unsigned(x), nodes - 2);
    const double t = x - left, t2 = t * t, t3 = t2 * t;
    if (!left || left + 2 >= nodes) return {{left, left + 1, left, left}, {1 - t, t, 0, 0}};
    return {{left - 1, left, left + 1, left + 2}, {-.5 * t + t2 - .5 * t3, 1 - 2.5 * t2 + 1.5 * t3, .5 * t + 2 * t2 - 1.5 * t3, -.5 * t2 + .5 * t3}};
}
void PrepareContact(const Model &m, State &s, double offset, double separation) {
    Validate(m, s);
    if (!std::isfinite(offset) || !std::isfinite(separation)) throw std::invalid_argument("Nonfinite contact geometry");
    const unsigned modes = Modes(m);
    const size_t rows = m.Bottom.Height.size() + m.Top.Height.size();
    s.Jacobian.resize(rows * modes);
    for (auto *values : {&s.Gap, &s.Weight, &s.Multiplier}) values->resize(rows);
    s.Rows = 0;
    for (bool top_slave : {false, true}) {
        const auto &slave = top_slave ? m.Top : m.Bottom, &master = top_slave ? m.Bottom : m.Top;
        for (unsigned n = 0; n < slave.Height.size(); ++n) {
            const double coordinate = std::fma(double(n), slave.Step / master.Step, (top_slave ? offset : -offset) / master.Step);
            const double end = master.Height.size() - 1, tolerance = 0x1p-45 * master.Height.size();
            // Retain coincident endpoints despite rounding during changes of coordinate frame.
            if (coordinate < -tolerance || coordinate > end + tolerance) continue;
            const double x = std::clamp(coordinate, 0., end);
            const auto interpolation = Interpolate(unsigned(master.Height.size()), x);
            const unsigned row = s.Rows++;
            s.Gap[row] = separation - slave.Height[n] - Sample(interpolation, master.Height.data());
            // Integrate the slave force once; transpose interpolation preserves virtual work at master endpoints.
            s.Weight[row] = Quadrature(slave, n);
            s.Multiplier[row] = 0;
            for (unsigned k = 0; k < m.Bottom.Modes.size(); ++k) {
                const auto *shape = m.Bottom.Shapes.data() + k * m.Bottom.Height.size();
                s.Jacobian[row * modes + k] = -(top_slave ? Sample(interpolation, shape) : shape[n]);
            }
            for (unsigned k = 0; k < m.Top.Modes.size(); ++k) {
                const auto *shape = m.Top.Shapes.data() + k * m.Top.Height.size();
                s.Jacobian[row * modes + m.Bottom.Modes.size() + k] = top_slave ? shape[n] : Sample(interpolation, shape);
            }
        }
    }
}
ContactResult Step(const Model &m, State &s, ContactMethod method, double offset, double separation, double tolerance, unsigned iterations) {
    if (!std::isfinite(tolerance) || tolerance <= 0 || !iterations || (method != ContactMethod::Penalty && method != ContactMethod::Multiplier)) throw std::invalid_argument("Invalid contact solver settings");
    PrepareContact(m, s, offset, separation);
    const unsigned modes = Modes(m);
    ContactResult result{.Converged = true};
    for (unsigned k = 0; k < modes; ++k) s.Force[k] = Gravity(m, k);
    if (method == ContactMethod::Penalty) {
        for (unsigned row = 0; row < s.Rows; ++row) {
            const auto *j = s.Jacobian.data() + row * modes;
            const double penetration = std::max(0., -s.Gap[row] - Dot(j, s.Displacement.data(), modes));
            const double stiffness = m.Penalty * s.Weight[row], force = stiffness * penetration;
            s.Multiplier[row] = force;
            result.NormalForce += force;
            result.ElasticEnergy += stiffness * penetration * penetration / 2;
            result.MaxPenetration = std::max(result.MaxPenetration, penetration);
            result.Contacts += force > 0;
            cblas_daxpy(int(modes), force, j, 1, s.Force.data(), 1);
        }
        for (unsigned k = 0; k < modes; ++k) s.Next[k] = AdvanceMode(Mode(m, k), s.Displacement[k], s.Previous[k], s.Force[k]);
    } else {
        for (unsigned k = 0; k < modes; ++k) s.Next[k] = AdvanceMode(Mode(m, k), s.Displacement[k], s.Previous[k], s.Force[k]);
        result.Converged = false;
        for (unsigned iteration = 0; iteration < iterations; ++iteration) {
            for (unsigned row = 0; row < s.Rows; ++row) {
                const auto *j = s.Jacobian.data() + row * modes;
                const double gap = s.Gap[row] + Dot(j, s.Next.data(), modes);
                if (gap >= 0 && s.Multiplier[row] == 0) continue;
                double diagonal{};
                for (unsigned k = 0; k < modes; ++k) diagonal += j[k] * j[k] * Mode(m, k).Compliance;
                if (diagonal <= 0) continue;
                const double next = std::max(0., s.Multiplier[row] - gap / diagonal), change = next - s.Multiplier[row];
                s.Multiplier[row] = next;
                for (unsigned k = 0; k < modes; ++k) s.Next[k] += Mode(m, k).Compliance * j[k] * change;
            }
            result.Residual = 0;
            for (unsigned row = 0; row < s.Rows; ++row) {
                const double gap = s.Gap[row] + Dot(s.Jacobian.data() + row * modes, s.Next.data(), modes);
                result.Residual = std::max(result.Residual, s.Multiplier[row] > 0 ? std::abs(gap) : std::max(0., -gap));
            }
            result.Iterations = iteration + 1;
            if (result.Residual <= tolerance) {
                result.Converged = true;
                break;
            }
        }
        for (unsigned row = 0; row < s.Rows; ++row) {
            const auto *j = s.Jacobian.data() + row * modes;
            result.NormalForce += s.Multiplier[row];
            result.MaxPenetration = std::max(result.MaxPenetration, -s.Gap[row] - Dot(j, s.Next.data(), modes));
            result.Contacts += s.Multiplier[row] > 0;
            cblas_daxpy(int(modes), s.Multiplier[row], j, 1, s.Force.data(), 1);
        }
    }
    if (!result.Converged) return result;
    for (double q : s.Next)
        if (!std::isfinite(q)) throw std::runtime_error("Contact integration became nonfinite");
    s.Previous.swap(s.Displacement);
    s.Displacement.swap(s.Next);
    return result;
}
double Energy(const Model &m, const State &s) {
    Validate(m, s);
    double result{};
    for (unsigned k = 0; k < Modes(m); ++k) result += ModalEnergy(Mode(m, k), s.Displacement[k], s.Previous[k], m.TimeStep);
    return result;
}
}
