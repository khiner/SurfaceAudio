#include "ContactDamping.h"
#include "NormalContact.h"
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>

namespace surface_audio {
namespace {
double Dot(std::span<const double> a, std::span<const double> b) { return cblas_ddot(int(a.size()), a.data(), 1, b.data(), 1); }
double Maximum(std::span<const double> v) {
    double result{};
    for (double x : v) {
        if (!std::isfinite(x)) throw std::runtime_error("Damped contact iteration became nonfinite");
        result = std::max(result, std::abs(x));
    }
    return result;
}
void SolvePositive(std::span<double> matrix, std::span<double> value) {
    const __LAPACK_int count = __LAPACK_int(value.size()), rhs = 1;
    __LAPACK_int info{};
    // Row-major lower storage is column-major upper storage.
    const char upper = 'U';
    dposv_(&upper, &count, &rhs, matrix.data(), &count, value.data(), &count, &info);
    if (info) throw std::runtime_error("Damped contact Hessian is not positive definite");
}
double Evaluate(ContactDampingState &s, std::span<const double> value) {
    const int modes = int(value.size());
    std::ranges::copy(value, s.Gradient.begin());
    double energy = Dot(value, value) / 2;
    for (size_t row = 0; row < s.Offset.size(); ++row) {
        const auto j = std::span(s.ScaledJacobian).subspan(row * modes, modes);
        const double displacement = Dot(j, value), damping = s.Damping[row];
        const double force = std::max(0., s.Offset[row] - damping * displacement);
        s.Force[row] = force;
        if (damping > 0) energy += force * force / (2 * damping);
        else energy -= force * displacement;
        cblas_daxpy(modes, -force, j.data(), 1, s.Gradient.data(), 1);
    }
    if (!std::isfinite(energy)) throw std::runtime_error("Damped contact objective became nonfinite");
    return energy;
}
}
ContactDampingResult SolveContactDamping(std::span<const double> jacobian, std::span<const double> penetration, std::span<const double> area, std::span<const double> free_velocity, std::span<const double> mobility, NormalContactLaw law, ContactDampingState &s, double tolerance, unsigned iterations) {
    const size_t modes = free_velocity.size(), rows = penetration.size();
    if (!modes || modes > INT_MAX || modes > size_t(INT_MAX) / modes || rows > SIZE_MAX / modes || jacobian.size() != rows * modes ||
        area.size() != rows || mobility.size() != modes || !std::isfinite(tolerance) || tolerance <= 0 || !iterations ||
        !std::isfinite(law.Stiffness) || law.Stiffness < 0 || !std::isfinite(law.Damping) || law.Damping < 0 ||
        !std::isfinite(law.Exponent) || law.Exponent <= 0 || !std::isfinite(law.DampingExponent) || law.DampingExponent < 0)
        throw std::invalid_argument("Invalid damped contact system");
    for (auto *v : {&s.Velocity, &s.Scale, &s.Value, &s.Trial, &s.Gradient, &s.Direction}) v->resize(modes);
    for (auto *v : {&s.Force, &s.Damping, &s.Offset}) v->resize(rows);
    s.ScaledJacobian.resize(rows * modes);
    for (size_t k = 0; k < modes; ++k) {
        if (!std::isfinite(mobility[k]) || mobility[k] <= 0 || !std::isfinite(free_velocity[k])) throw std::invalid_argument("Invalid contact mobility or free velocity");
        s.Scale[k] = std::sqrt(mobility[k]);
        s.Value[k] = 0;
    }
    for (size_t row = 0; row < rows; ++row) {
        const double delta = penetration[row];
        if (!std::isfinite(delta) || !std::isfinite(area[row]) || area[row] <= 0) throw std::invalid_argument("Invalid contact penetration or area");
        const auto j = jacobian.subspan(row * modes, modes);
        for (size_t k = 0; k < modes; ++k) {
            if (!std::isfinite(j[k])) throw std::invalid_argument("Nonfinite contact Jacobian");
            s.ScaledJacobian[row * modes + k] = j[k] * s.Scale[k];
        }
        const double elastic = delta > 0 ? area[row] * law.Stiffness * std::pow(delta, law.Exponent) : 0;
        s.Damping[row] = delta > 0 ? area[row] * law.Damping * std::pow(delta, law.DampingExponent) : 0;
        s.Offset[row] = elastic - s.Damping[row] * Dot(j, free_velocity);
        if (!std::isfinite(elastic) || !std::isfinite(s.Damping[row]) || !std::isfinite(s.Offset[row])) throw std::invalid_argument("Contact law exceeds numerical range");
    }
    double energy = Evaluate(s, s.Value);
    const double reference = std::max(1e-30, Maximum(s.Gradient));
    ContactDampingResult result{};
    for (unsigned iteration = 0; iteration < iterations; ++iteration) {
        const double norm = Maximum(s.Gradient);
        result.Residual = norm / std::max(reference, Maximum(s.Value));
        result.Iterations = iteration;
        if (result.Residual <= tolerance) break;
        s.ActiveRows.clear();
        for (size_t row = 0; row < rows; ++row)
            if (s.Force[row] > 0 && s.Damping[row] > 0) s.ActiveRows.push_back(row);
        const size_t active = s.ActiveRows.size();
        if (active < modes) {
            s.WeightedJacobian.resize(active * modes);
            s.ContactValue.resize(active);
            s.Hessian.resize(active * active);
            std::ranges::fill(s.Direction, 0);
            for (size_t row = 0; row < rows; ++row)
                if (s.Force[row] > 0 && s.Damping[row] == 0)
                    cblas_daxpy(int(modes), s.Offset[row], s.ScaledJacobian.data() + row * modes, 1, s.Direction.data(), 1);
            // Eliminate modal increments: (I + B*B^T) z = offset/sqrt(damping) - B*elastic_increment.
            for (size_t i = 0; i < active; ++i) {
                const size_t row = s.ActiveRows[i];
                const double scale = std::sqrt(s.Damping[row]);
                for (size_t k = 0; k < modes; ++k) s.WeightedJacobian[i * modes + k] = scale * s.ScaledJacobian[row * modes + k];
                s.ContactValue[i] = s.Offset[row] / scale - Dot(std::span(s.WeightedJacobian).subspan(i * modes, modes), s.Direction);
            }
            if (active) {
                cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, int(active), int(modes), 1, s.WeightedJacobian.data(), int(modes), 0, s.Hessian.data(), int(active));
                for (size_t i = 0; i < active; ++i) s.Hessian[i * active + i] += 1;
                SolvePositive(s.Hessian, s.ContactValue);
                cblas_dgemv(CblasRowMajor, CblasTrans, int(active), int(modes), 1, s.WeightedJacobian.data(), int(modes), s.ContactValue.data(), 1, 1, s.Direction.data(), 1);
            }
            for (size_t k = 0; k < modes; ++k) s.Direction[k] -= s.Value[k];
        } else {
            s.Hessian.resize(modes * modes);
            std::ranges::fill(s.Hessian, 0);
            for (size_t k = 0; k < modes; ++k) {
                s.Hessian[k * modes + k] = 1;
                s.Direction[k] = -s.Gradient[k];
            }
            for (size_t row : s.ActiveRows)
                cblas_dsyr(CblasRowMajor, CblasLower, int(modes), s.Damping[row], s.ScaledJacobian.data() + row * modes, 1, s.Hessian.data(), int(modes));
            SolvePositive(s.Hessian, s.Direction);
        }
        const double slope = Dot(s.Gradient, s.Direction);
        bool accepted{};
        for (double step = 1; step >= 0x1p-30; step *= .5) {
            for (size_t k = 0; k < modes; ++k) s.Trial[k] = s.Value[k] + step * s.Direction[k];
            const double trial_energy = Evaluate(s, s.Trial);
            if (trial_energy <= energy + 1e-4 * step * slope || Maximum(s.Gradient) < norm * (1 - 1e-4 * step)) {
                s.Value.swap(s.Trial);
                energy = trial_energy;
                accepted = true;
                break;
            }
        }
        result.Iterations = iteration + 1;
        if (!accepted) {
            Evaluate(s, s.Value);
            break;
        }
    }
    result.Residual = Maximum(s.Gradient) / std::max(reference, Maximum(s.Value));
    for (size_t k = 0; k < modes; ++k) s.Velocity[k] = free_velocity[k] + s.Scale[k] * s.Value[k];
    for (size_t row = 0; row < rows; ++row) {
        const double rate = -Dot(jacobian.subspan(row * modes, modes), s.Velocity);
        s.Force[row] = area[row] * EvaluateNormalContact(penetration[row], rate, law.Stiffness, law.Damping, law.Exponent, law.DampingExponent).Force;
        if (!std::isfinite(s.Force[row])) throw std::runtime_error("Damped contact solution became nonfinite");
    }
    std::ranges::copy(s.Value, s.Gradient.begin());
    for (size_t row = 0; row < rows; ++row) cblas_daxpy(int(modes), -s.Force[row], s.ScaledJacobian.data() + row * modes, 1, s.Gradient.data(), 1);
    result.Residual = std::max(result.Residual, Maximum(s.Gradient) / std::max(reference, Maximum(s.Value)));
    result.Converged = result.Residual <= tolerance;
    return result;
}
}
