// SPDX-License-Identifier: GPL-3.0-only
#include "matusiak/Matusiak.h"
#include "core/FiniteDifference.h"
#include "matusiak/Solve.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace surface_audio::matusiak {
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
    // Preserve the author's x=0 convention for the first interior DOF.
    if (node < 1 || node + 2 >= static_cast<int>(n)) throw std::invalid_argument("Bow interpolation crosses boundary");
    for (int j = 0; j < 4; ++j) matrix[row * n + node - 1 + j] = weights[j];
    return node - 1;
}
}
StringState MakeString(Parameters p) {
    const double values[]{p.SampleRate, p.Length, p.Radius, p.Fundamental, p.Tension, p.Young, p.Damping0, p.Damping1, p.TorsionStiffness, p.PolarInertia, p.TorsionDamping, p.BowWidth, p.BowPosition, p.HairMass, p.HairStiffness, p.HairDamping, p.Friction.Stiffness, p.Friction.Damping, p.Friction.StribeckVelocity, p.Friction.Dynamic, p.Friction.Static, p.Friction.BreakawayRatio};
    for (double value : values)
        if (!std::isfinite(value) || value < 0) throw std::invalid_argument("Invalid bowed string parameter");
    if (p.SampleRate < 8000 || p.Length <= 0 || p.Radius <= 0 || p.Fundamental <= 0 || p.Tension <= 0 || p.PolarInertia <= 0 || p.TorsionStiffness <= 0 || p.HairMass <= 0 || p.Friction.Stiffness <= 0 || p.Friction.StribeckVelocity <= 0 || p.Friction.Dynamic <= 0 || p.Friction.Static < p.Friction.Dynamic || p.Friction.BreakawayRatio >= 1 || p.BowPoints < 1 || p.BowPoints > 32) throw std::invalid_argument("Invalid bowed string parameter range");
    const double step = 1 / p.SampleRate, speed = 2 * p.Fundamental * p.Length;
    const double density = p.Tension / (speed * speed), bending = p.Young * std::numbers::pi * std::pow(p.Radius, 4) / 4;
    const double torsion_speed = std::sqrt(p.TorsionStiffness / p.PolarInertia);
    const double minimum = StiffStringMinimumSpacing(speed, bending / density, p.Damping1, step);
    const double transverse_count = std::floor(p.Length / minimum), torsional_count = std::floor(p.Length / (torsion_speed * step));
    if (!std::isfinite(transverse_count) || !std::isfinite(torsional_count) || transverse_count < 6 || torsional_count < 6 || transverse_count > 100000 || torsional_count > 100000) throw std::invalid_argument("Invalid string grid");
    const auto n = static_cast<size_t>(transverse_count), nt = static_cast<size_t>(torsional_count);
    const double width = p.BowWidth > 0 ? p.BowWidth : 1;
    StringState s{.Config = p, .Step = step, .Spacing = p.Length / n, .TorsionSpacing = p.Length / nt, .Density = density, .Bending = bending, .WaveSpeed = speed, .TorsionSpeed = torsion_speed, .HairMass = p.HairMass / width, .HairStiffness = p.HairStiffness / width, .HairDamping = p.HairDamping / width};
    for (auto *u : {&s.U, &s.PreviousU, &s.NextU, &s.FreeU}) u->resize(n - 1);
    for (auto *u : {&s.W, &s.PreviousW, &s.NextW, &s.FreeW}) u->resize(nt - 1);
    const size_t m = p.BowPoints;
    for (auto *u : {&s.Hair, &s.PreviousHair, &s.NextHair, &s.FreeHair, &s.Z, &s.Velocity, &s.MidpointZ, &s.Force, &s.Work}) u->resize(m);
    s.Interpolation.resize(m * (n - 1));
    s.TorsionInterpolation.resize(m * (nt - 1));
    s.Coupling.resize(m * m);
    s.Jacobian.resize(4 * m * m);
    s.Residual.resize(2 * m);
    for (size_t i = 0; i < m; ++i) {
        const double x = p.Length * p.BowPosition + (m > 1 ? p.BowWidth * (double(i) / (m - 1) - .5) : 0);
        s.InterpolationOffset.push_back(Interpolate(s.Interpolation, i, n - 1, x / s.Spacing));
        s.TorsionInterpolationOffset.push_back(Interpolate(s.TorsionInterpolation, i, nt - 1, x / s.TorsionSpacing));
    }
    const double xs = 1 / (2 / s.Step + 2 * p.Damping0), xt = 1 / (2 / s.Step + 2 * p.TorsionDamping);
    const double xh = 1 / (2 * s.HairMass / s.Step + s.Step * s.HairStiffness / 2 + s.HairDamping);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < m; ++j)
            s.Coupling[i * m + j] = xs / (m * s.Density * s.Spacing) * Dot(&s.Interpolation[i * (n - 1)], &s.Interpolation[j * (n - 1)], n - 1) + s.Spacing / s.TorsionSpacing * p.Radius * p.Radius * xt / (m * p.PolarInertia * s.TorsionSpacing) * Dot(&s.TorsionInterpolation[i * (nt - 1)], &s.TorsionInterpolation[j * (nt - 1)], nt - 1) + (i == j ? xh / m : 0);
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
    for (size_t i = 0; i < s.W.size(); ++i) energy.Torsion += p.PolarInertia * h / 2 * std::pow((s.W[i] - s.PreviousW[i]) / k, 2);
    for (size_t i = 0; i <= s.W.size(); ++i) energy.Torsion += p.TorsionStiffness * h / (2 * ht * ht) * ((i < s.W.size() ? s.W[i] : 0) - (i ? s.W[i - 1] : 0)) * ((i < s.W.size() ? s.PreviousW[i] : 0) - (i ? s.PreviousW[i - 1] : 0));
    for (size_t i = 0; i < s.Z.size(); ++i) {
        energy.Hair += s.HairMass / 2 * std::pow((s.Hair[i] - s.PreviousHair[i]) / k, 2) + s.HairStiffness / 2 * std::pow((s.Hair[i] + s.PreviousHair[i]) / 2, 2);
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
    const double xs = 1 / (2 / k + 2 * p.Damping0), xt = 1 / (2 / k + 2 * p.TorsionDamping), xh = 1 / (2 * s.HairMass / k + k * s.HairStiffness / 2 + s.HairDamping);
    for (size_t i = 0; i < n; ++i) s.FreeU[i] = xs * (s.WaveSpeed * s.WaveSpeed / (h * h) * D2(s.U, i) - s.Bending / (s.Density * h * h * h * h) * D4(s.U, i) + 2 * p.Damping1 / (k * h * h) * (D2(s.U, i) - D2(s.PreviousU, i)) + 2 / (k * k) * (s.U[i] - s.PreviousU[i]));
    for (size_t i = 0; i < nt; ++i) s.FreeW[i] = xt * (s.TorsionSpeed * s.TorsionSpeed / (ht * ht) * D2(s.W, i) + 2 / (k * k) * (s.W[i] - s.PreviousW[i]));
    for (size_t i = 0; i < m; ++i) {
        const size_t a = s.InterpolationOffset[i], b = s.TorsionInterpolationOffset[i];
        s.FreeHair[i] = xh * ((k * s.HairStiffness / 2 + 2 * s.HairMass / k) * (s.Hair[i] - s.PreviousHair[i]) / k - s.HairStiffness * s.Hair[i]);
        s.Work[i] = Dot(&s.Interpolation[i * n + a], s.FreeU.data() + a, 4) - p.Radius * h / ht * Dot(&s.TorsionInterpolation[i * nt + b], s.FreeW.data() + b, 4) + s.FreeHair[i] - bow_velocity;
    }
    Sample sample;
    for (uint32_t iteration = 0; iteration < 100; ++iteration) {
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
        if (iteration > 50)
            for (double &a : s.Jacobian) a *= 1.1;
        if (!Solve(s.Jacobian.data(), s.Residual.data(), unsigned(dim), 1e-20)) throw std::runtime_error("Singular friction Jacobian");
        for (size_t i = 0; i < m; ++i) {
            s.Velocity[i] -= s.Residual[i];
            s.MidpointZ[i] -= s.Residual[m + i];
        }
        sample.Iterations = iteration + 1;
        if (iteration == 99) throw std::runtime_error("Bowed string Newton solver did not converge");
    }
    for (size_t j = 0; j < m; ++j) {
        const auto f = Evaluate(p.Friction, normal_force, s.MidpointZ[j], s.Velocity[j]);
        const double rate = 2 / k * (s.MidpointZ[j] - s.Z[j]);
        s.Force[j] = p.Friction.Stiffness * s.MidpointZ[j] + f.Damping * rate;
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
        sample.Dissipation += 2 * p.TorsionDamping * p.PolarInertia * h * std::pow((s.NextW[i] - s.PreviousW[i]) / (2 * k), 2);
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
