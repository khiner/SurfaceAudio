// SPDX-License-Identifier: GPL-3.0-only
#include "matusiak2024/Matusiak2024.h"
#include "core/FiniteDifference.h"
#include "core/PivotedSolve.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace surface_audio::matusiak2024 {
namespace {
double D2(const std::vector<double> &u, size_t i) {
    return SecondDifference(i ? u[i - 1] : 0, u[i], i + 1 < u.size() ? u[i + 1] : 0);
}
double D4(const std::vector<double> &u, size_t i) {
    return (i ? D2(u, i - 1) : 0) - 2 * D2(u, i) + (i + 1 < u.size() ? D2(u, i + 1) : 0);
}
double Dot(const double *a, const double *b, size_t n) {
    double sum{};
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}
size_t Interpolate(std::vector<double> &matrix, size_t row, size_t n, double x) {
    const int node = static_cast<int>(std::floor(x));
    const double a = x - node;
    const double weights[]{-a * (a - 1) * (a - 2) / 6, (a - 1) * (a + 1) * (a - 2) / 2, -a * (a + 1) * (a - 2) / 2, a * (a + 1) * (a - 1) / 6};
    if (node < 1 || node + 2 >= static_cast<int>(n)) throw std::invalid_argument("Bow interpolation crosses boundary");
    for (int j = 0; j < 4; ++j) matrix[row * n + node - 1 + j] = weights[j];
    return node - 1;
}
}
StringState MakeString(Parameters p) {
    if (p.Convention != NumericalConvention::Paper2024 && p.Convention != NumericalConvention::AuthorArchive2025) throw std::invalid_argument("Unknown numerical convention");
    const double values[]{p.SampleRate, p.Length, p.Radius, p.MaterialDensity, p.Tension, p.Young, p.Damping0, p.Damping1, p.TorsionStiffness, p.PolarInertia, p.TorsionDamping, p.BowWidth, p.BowPosition, p.HairStiffness, p.HairDamping, p.Friction.Stiffness, p.Friction.Damping, p.Friction.StribeckVelocity, p.Friction.Dynamic, p.Friction.Static, p.Friction.BreakawayRatio};
    for (double value : values)
        if (!std::isfinite(value) || value < 0) throw std::invalid_argument("Invalid bowed string parameter");
    if (p.SampleRate < 8000 || p.Length <= 0 || p.Radius <= 0 || p.MaterialDensity <= 0 || p.Tension <= 0 || p.PolarInertia <= 0 || p.TorsionStiffness <= 0 || p.Friction.Stiffness <= 0 || p.Friction.StribeckVelocity <= 0 || p.Friction.Dynamic <= 0 || p.Friction.Static < p.Friction.Dynamic || p.Friction.BreakawayRatio >= 1 || p.Friction.Exponential > 1 || p.HairStiffness + p.HairDamping <= 0 || p.BowPoints < 1 || p.BowPoints > 32) throw std::invalid_argument("Invalid bowed string parameter range");
    const double step = 1 / p.SampleRate, density = p.MaterialDensity * std::numbers::pi * p.Radius * p.Radius;
    const double speed = std::sqrt(p.Tension / density), bending = p.Young * std::numbers::pi * std::pow(p.Radius, 4) / 4;
    const double torsion_speed = std::sqrt(p.TorsionStiffness / p.PolarInertia);
    const double minimum = StiffStringMinimumSpacing(speed, bending / density, p.Damping1, step);
    const double transverse_count = std::floor(p.Length / minimum), torsional_count = std::floor(p.Length / (torsion_speed * step));
    if (!std::isfinite(transverse_count) || !std::isfinite(torsional_count) || transverse_count < 6 || torsional_count < 6 || transverse_count > 100000 || torsional_count > 100000) throw std::invalid_argument("Invalid string grid");
    const auto n = static_cast<size_t>(transverse_count), nt = static_cast<size_t>(torsional_count);
    StringState s{.Config = p, .Step = step, .Spacing = p.Length / n, .TorsionSpacing = p.Length / nt, .Density = density, .Bending = bending, .WaveSpeed = speed, .TorsionSpeed = torsion_speed, .TorsionFeedback = p.Convention == NumericalConvention::AuthorArchive2025 ? (p.Length / n) / (p.Length / nt) : 1, .HairStiffness = p.HairStiffness, .HairDamping = p.HairDamping};
    for (auto *u : {&s.U, &s.PreviousU, &s.NextU, &s.FreeU}) u->resize(n - 1);
    for (auto *u : {&s.W, &s.PreviousW, &s.NextW, &s.FreeW}) u->resize(nt - 1);
    const size_t m = p.BowPoints;
    for (auto *u : {&s.Hair, &s.PreviousHair, &s.NextHair, &s.FreeHair, &s.Z, &s.Velocity, &s.MidpointZ, &s.Force, &s.Work}) u->resize(m);
    s.Interpolation.resize(m * (n - 1));
    s.TorsionInterpolation.resize(m * (nt - 1));
    s.Coupling.resize(m * m);
    s.Jacobian.resize(4 * m * m);
    s.Residual.resize(2 * m);
    // The archive indexes its first interior degree of freedom as coordinate zero.
    const double coordinate_offset = p.Convention == NumericalConvention::AuthorArchive2025 ? 0 : 1;
    for (size_t i = 0; i < m; ++i) {
        const double x = p.Length * p.BowPosition + (m > 1 ? p.BowWidth * (double(i) / (m - 1) - .5) : 0);
        s.InterpolationOffset.push_back(Interpolate(s.Interpolation, i, n - 1, x / s.Spacing - coordinate_offset));
        s.TorsionInterpolationOffset.push_back(Interpolate(s.TorsionInterpolation, i, nt - 1, x / s.TorsionSpacing - coordinate_offset));
    }
    const double xs = 1 / (2 / s.Step + 2 * p.Damping0), xt = 1 / (2 / s.Step + 2 * p.TorsionDamping);
    const double xh = 1 / (s.Step * s.HairStiffness + s.HairDamping);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < m; ++j)
            s.Coupling[i * m + j] = xs / (m * s.Density * s.Spacing) * Dot(&s.Interpolation[i * (n - 1)], &s.Interpolation[j * (n - 1)], n - 1) + s.TorsionFeedback * p.Radius * p.Radius * xt / (m * p.PolarInertia * s.TorsionSpacing) * Dot(&s.TorsionInterpolation[i * (nt - 1)], &s.TorsionInterpolation[j * (nt - 1)], nt - 1) + (i == j ? xh / m : 0);
    return s;
}
Energy StoredEnergy(const StringState &s) {
    Energy energy;
    const auto &p = s.Config;
    const double k = s.Step, h = s.Spacing, ht = s.TorsionSpacing;
    for (size_t i = 0; i < s.U.size(); ++i) {
        energy.String += s.Density * h / 2 * std::pow((s.U[i] - s.PreviousU[i]) / k, 2) + s.Bending / (2 * h * h * h) * D2(s.U, i) * D2(s.PreviousU, i);
    }
    for (size_t i = 0; i <= s.U.size(); ++i) energy.String += p.Tension / (2 * h) * ((i < s.U.size() ? s.U[i] : 0) - (i ? s.U[i - 1] : 0)) * ((i < s.U.size() ? s.PreviousU[i] : 0) - (i ? s.PreviousU[i - 1] : 0));
    for (size_t i = 0; i < s.W.size(); ++i) energy.Torsion += s.TorsionFeedback * p.PolarInertia * ht / 2 * std::pow((s.W[i] - s.PreviousW[i]) / k, 2);
    for (size_t i = 0; i <= s.W.size(); ++i) energy.Torsion += s.TorsionFeedback * p.TorsionStiffness / (2 * ht) * ((i < s.W.size() ? s.W[i] : 0) - (i ? s.W[i - 1] : 0)) * ((i < s.W.size() ? s.PreviousW[i] : 0) - (i ? s.PreviousW[i - 1] : 0));
    for (size_t i = 0; i < s.Z.size(); ++i) {
        energy.Hair += s.HairStiffness / 4 * (s.Hair[i] * s.Hair[i] + s.PreviousHair[i] * s.PreviousHair[i]);
        energy.Bristle += p.Friction.Stiffness / (2 * s.Z.size()) * s.Z[i] * s.Z[i];
    }
    return energy;
}
Sample Step(StringState &s, double bow_velocity, double normal_force) {
    if (!std::isfinite(bow_velocity) || !std::isfinite(normal_force) || normal_force <= 0) throw std::invalid_argument("Bow velocity must be finite and force positive");
    if (s.Steps++ == 0) s.InitialEnergy = TotalEnergy(StoredEnergy(s));
    const auto &p = s.Config;
    const size_t n = s.U.size(), nt = s.W.size(), m = s.Z.size(), dim = 2 * m;
    const double k = s.Step, h = s.Spacing, ht = s.TorsionSpacing;
    const double xs = 1 / (2 / k + 2 * p.Damping0), xt = 1 / (2 / k + 2 * p.TorsionDamping), xh = 1 / (k * s.HairStiffness + s.HairDamping);
    for (size_t i = 0; i < n; ++i) s.FreeU[i] = xs * (s.WaveSpeed * s.WaveSpeed / (h * h) * D2(s.U, i) - s.Bending / (s.Density * h * h * h * h) * D4(s.U, i) + 2 * p.Damping1 / (k * h * h) * (D2(s.U, i) - D2(s.PreviousU, i)) + 2 / (k * k) * (s.U[i] - s.PreviousU[i]));
    for (size_t i = 0; i < nt; ++i) s.FreeW[i] = xt * (s.TorsionSpeed * s.TorsionSpeed / (ht * ht) * D2(s.W, i) + 2 / (k * k) * (s.W[i] - s.PreviousW[i]));
    for (size_t i = 0; i < m; ++i) {
        const size_t a = s.InterpolationOffset[i], b = s.TorsionInterpolationOffset[i];
        s.FreeHair[i] = -xh * s.HairStiffness * s.PreviousHair[i];
        s.Work[i] = Dot(&s.Interpolation[i * n + a], s.FreeU.data() + a, 4) - s.TorsionFeedback * p.Radius * Dot(&s.TorsionInterpolation[i * nt + b], s.FreeW.data() + b, 4) + s.FreeHair[i] - bow_velocity;
    }
    Sample sample;
    for (uint32_t iteration = 0; iteration < 100; ++iteration) {
        // Reinitialize the same nonlinear equations when continuation stalls at a slip transition.
        if (iteration == 25 || iteration == 50 || iteration == 75) {
            for (size_t j = 0; j < m; ++j) {
                s.Velocity[j] = s.Work[j] * (iteration == 25 ? 1 : iteration == 50 ? .5 :
                                                                                     2);
                const double steady = Evaluate(p.Friction, normal_force, 0., s.Velocity[j]).Steady;
                double low = std::min({s.Z[j], s.Z[j] + k / 2 * s.Velocity[j], steady});
                double high = std::max({s.Z[j], s.Z[j] + k / 2 * s.Velocity[j], steady});
                for (unsigned bisection = 0; bisection < 60; ++bisection) {
                    const double mid = (low + high) / 2;
                    const double residual = Evaluate(p.Friction, normal_force, mid, s.Velocity[j]).Rate - 2 / k * (mid - s.Z[j]);
                    if (residual > 0) low = mid;
                    else high = mid;
                }
                s.MidpointZ[j] = (low + high) / 2;
            }
        }
        for (size_t j = 0; j < m; ++j) {
            const auto f = Evaluate(p.Friction, normal_force, s.MidpointZ[j], s.Velocity[j]);
            s.Force[j] = f.Force;
            s.Residual[m + j] = f.Rate - 2 / k * (s.MidpointZ[j] - s.Z[j]);
            for (size_t i = 0; i < m; ++i) {
                s.Jacobian[i * dim + j] = (i == j ? 1 : 0) + s.Coupling[i * m + j] * (f.Damping * f.RateVelocity + f.DampingVelocity * f.Rate);
                s.Jacobian[i * dim + m + j] = s.Coupling[i * m + j] * (p.Friction.Stiffness + f.Damping * f.RateBristle);
                s.Jacobian[(m + i) * dim + j] = i == j ? f.RateVelocity : 0;
                s.Jacobian[(m + i) * dim + m + j] = i == j ? f.RateBristle - 2 / k : 0;
            }
        }
        for (size_t i = 0; i < m; ++i) s.Residual[i] = s.Velocity[i] + Dot(&s.Coupling[i * m], s.Force.data(), m) - s.Work[i];
        sample.Residual = 0;
        for (double r : s.Residual) sample.Residual = std::max(sample.Residual, std::abs(r));
        if (sample.Residual < 2e-13) break;
        if (!Solve(s.Jacobian.data(), s.Residual.data(), unsigned(dim), 1e-20)) throw std::runtime_error("Singular friction Jacobian");
        double scale = 1;
        std::array<double, 32> trial_force{};
        for (unsigned search = 0; search < 24; ++search) {
            double trial_error{};
            for (size_t j = 0; j < m; ++j) {
                const double trial_z = s.MidpointZ[j] - scale * s.Residual[m + j];
                const auto trial = Evaluate(p.Friction, normal_force, trial_z, s.Velocity[j] - scale * s.Residual[j]);
                trial_force[j] = trial.Force;
                trial_error = std::max(trial_error, std::abs(trial.Rate - 2 / k * (trial_z - s.Z[j])));
            }
            for (size_t j = 0; j < m; ++j) trial_error = std::max(trial_error, std::abs(s.Velocity[j] - scale * s.Residual[j] + Dot(&s.Coupling[j * m], trial_force.data(), m) - s.Work[j]));
            if (trial_error < sample.Residual || trial_error < 2e-13) break;
            scale *= .5;
        }
        for (size_t i = 0; i < m; ++i) {
            s.Velocity[i] -= scale * s.Residual[i];
            s.MidpointZ[i] -= scale * s.Residual[m + i];
        }
        sample.Iterations = iteration + 1;
        if (iteration == 99) throw std::runtime_error("Bowed string Newton solver did not converge at step " + std::to_string(s.Steps) + ", residual " + std::to_string(sample.Residual));
    }
    for (size_t j = 0; j < m; ++j) {
        const double rate = 2 / k * (s.MidpointZ[j] - s.Z[j]);
        s.Force[j] = p.Friction.Stiffness * s.MidpointZ[j] + p.Friction.Damping * rate;
        sample.BristleDissipation += (s.Velocity[j] * s.Force[j] - p.Friction.Stiffness * s.MidpointZ[j] * rate) / m;
        sample.InputPower -= bow_velocity * s.Force[j] / m;
        s.NextHair[j] = s.PreviousHair[j] + 2 * k * (s.FreeHair[j] - xh * s.Force[j] / m);
        sample.Dissipation += s.HairDamping * std::pow((s.NextHair[j] - s.PreviousHair[j]) / (2 * k), 2);
    }
    for (size_t i = 0; i < n; ++i) {
        double force{};
        for (size_t j = 0; j < m; ++j) force += s.Interpolation[j * n + i] * s.Force[j] / (m * s.Density * h);
        s.NextU[i] = s.PreviousU[i] + 2 * k * (s.FreeU[i] - xs * force);
        const double velocity = (s.NextU[i] - s.PreviousU[i]) / (2 * k);
        sample.Dissipation += 2 * p.Damping0 * s.Density * h * velocity * velocity - 2 * p.Damping1 * s.Density / h * velocity * (D2(s.U, i) - D2(s.PreviousU, i)) / k;
    }
    for (size_t i = 0; i < nt; ++i) {
        double force{};
        for (size_t j = 0; j < m; ++j) force += s.TorsionInterpolation[j * nt + i] * s.Force[j] * p.Radius / (m * p.PolarInertia * ht);
        s.NextW[i] = s.PreviousW[i] + 2 * k * (s.FreeW[i] + xt * force);
        sample.Dissipation += s.TorsionFeedback * 2 * p.TorsionDamping * p.PolarInertia * ht * std::pow((s.NextW[i] - s.PreviousW[i]) / (2 * k), 2);
    }
    sample.Dissipation += sample.BristleDissipation;
    s.IntegratedLoss += k * (sample.Dissipation - sample.InputPower);
    for (size_t i = 0; i < m; ++i) s.Z[i] = 2 * s.MidpointZ[i] - s.Z[i];
    s.PreviousU.swap(s.U);
    s.U.swap(s.NextU);
    s.PreviousW.swap(s.W);
    s.W.swap(s.NextW);
    s.PreviousHair.swap(s.Hair);
    s.Hair.swap(s.NextHair);
    sample.Stored = StoredEnergy(s);
    sample.EnergyError = TotalEnergy(sample.Stored) + s.IntegratedLoss - s.InitialEnergy;
    sample.BridgeForce = p.Tension / h * s.U[0] - s.Bending / (h * h * h) * (s.U[1] - 2 * s.U[0]);
    sample.RelativeVelocity = s.Velocity[m / 2];
    sample.FrictionForce = s.Force[m / 2];
    return sample;
}
}
