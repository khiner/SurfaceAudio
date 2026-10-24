#include "PatchContact.h"
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace surface_audio::rough {
namespace {
void ValidateResponse(size_t n, unsigned taps, size_t values) {
    if (!n || n > INT_MAX || !taps || n > std::numeric_limits<size_t>::max() / n / taps || values != n * n * taps)
        throw std::invalid_argument("Invalid interfacial response dimensions");
}
void Validate(const PatchSpring &s) {
    if (s.size() < 2 || s.front().Compression != 0 || s.front().Force != 0)
        throw std::invalid_argument("Invalid interfacial spring dimensions or origin");
    for (size_t i = 0; i < s.size(); ++i) {
        const auto &k = s[i];
        if (!std::isfinite(k.Compression) || !std::isfinite(k.Stiffness) || k.Stiffness < 0 || !std::isfinite(k.Force))
            throw std::invalid_argument("Invalid interfacial spring sample");
        if (i && (k.Compression <= s[i - 1].Compression || k.Stiffness < s[i - 1].Stiffness ||
                  k.Force != s[i - 1].Force + (k.Compression - s[i - 1].Compression) * (.5 * s[i - 1].Stiffness + .5 * k.Stiffness)))
            throw std::invalid_argument("Interfacial spring must have increasing compression and consistent integrated stiffness");
    }
}
}
PatchSpring MakePatchSpring(std::span<const std::array<double, 2>> samples) {
    PatchSpring s;
    for (const auto [compression, stiffness] : samples) {
        const double force = s.empty() ? 0 : s.back().Force + (compression - s.back().Compression) * (.5 * s.back().Stiffness + .5 * stiffness);
        s.push_back({compression, stiffness, force});
    }
    Validate(s);
    return s;
}
std::array<double, 2> EvaluatePatchSpring(const PatchSpring &s, double compression) {
    if (s.size() < 2) throw std::invalid_argument("Invalid interfacial spring dimensions");
    if (!std::isfinite(compression) || compression > s.back().Compression) throw std::out_of_range("Interfacial compression outside spring domain");
    if (compression <= 0) return {};
    const auto upper = std::min(std::ranges::upper_bound(s, compression, {}, &PatchSpringKnot::Compression), s.end() - 1);
    const auto &lower = *(upper - 1);
    const double distance = compression - lower.Compression, width = upper->Compression - lower.Compression;
    const double increment = (upper->Stiffness - lower.Stiffness) * (distance / width);
    return {lower.Force + distance * (lower.Stiffness + .5 * increment), lower.Stiffness + increment};
}
PatchContactModel MakePatchContact(std::vector<PatchSpring> springs, unsigned taps, double dt, std::span<const double> green) {
    ValidateResponse(springs.size(), taps, green.size());
    if (!std::isfinite(dt) || dt <= 0) throw std::invalid_argument("Invalid interfacial time step");
    for (const auto &s : springs) Validate(s);
    PatchContactModel m{std::move(springs), {green.begin(), green.end()}, taps};
    for (double &value : m.Response) {
        value *= dt;
        if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite interfacial response");
    }
    return m;
}
PatchContactState MakePatchContactState(const PatchContactModel &m) {
    const size_t n = m.Springs.size();
    ValidateResponse(n, m.Taps, m.Response.size());
    return {std::vector<double>(n), std::vector<double>(n), std::vector<double>(n), std::vector<double>(n * (m.Taps - 1)), 0};
}
PatchContactResult StepPatchContact(const PatchContactModel &m, PatchContactState &s, std::span<const double> approach, double tolerance, unsigned iterations) {
    const size_t n = m.Springs.size();
    ValidateResponse(n, m.Taps, m.Response.size());
    const size_t history_length = m.Taps - 1, head = history_length ? s.Samples % history_length : 0;
    if (approach.size() != n ||
        s.Compression.size() != n || s.Force.size() != n || s.Displacement.size() != n || s.History.size() != history_length * n ||
        !std::isfinite(tolerance) || tolerance <= 0 || !iterations)
        throw std::invalid_argument("Invalid interfacial contact state or solver settings");
    for (double value : approach)
        if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite interfacial approach");
    std::vector<double> past(n), force(n), stiffness(n), residual(n), displacement(n), jacobian(n * n), direction(n), trial(n);
    std::vector<__LAPACK_int> pivots(n);
    auto compression = s.Compression;
    for (size_t lag = 1; lag <= std::min(s.Samples, uint64_t(history_length)); ++lag) {
        const size_t slot = (head + history_length - lag) % history_length;
        cblas_dgemv(CblasRowMajor, CblasNoTrans, int(n), int(n), 1., m.Response.data() + lag * n * n, int(n), s.History.data() + slot * n, 1, 1., past.data(), 1);
    }
    const auto evaluate = [&](const std::vector<double> &x) {
        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(x[i]) || x[i] > m.Springs[i].back().Compression) return std::numeric_limits<double>::infinity();
            const auto value = EvaluatePatchSpring(m.Springs[i], x[i]);
            force[i] = value[0];
            stiffness[i] = value[1];
            if (!std::isfinite(force[i]) || !std::isfinite(stiffness[i])) return std::numeric_limits<double>::infinity();
        }
        displacement = past;
        cblas_dgemv(CblasRowMajor, CblasNoTrans, int(n), int(n), 1., m.Response.data(), int(n), force.data(), 1, 1., displacement.data(), 1);
        double norm{};
        for (size_t i = 0; i < n; ++i) {
            residual[i] = x[i] + displacement[i] - approach[i];
            if (!std::isfinite(residual[i])) return std::numeric_limits<double>::infinity();
            norm = std::max(norm, std::abs(residual[i]));
        }
        return norm;
    };
    PatchContactResult result{evaluate(compression), 0, false};
    while (result.Residual > tolerance && result.Iterations < iterations) {
        if (!std::isfinite(result.Residual)) return result;
        for (size_t i = 0; i < n; ++i) {
            direction[i] = -residual[i];
            for (size_t j = 0; j < n; ++j) jacobian[j * n + i] = (i == j ? 1. : 0.) + m.Response[i * n + j] * stiffness[j];
        }
        ++result.Iterations;
        const __LAPACK_int count = __LAPACK_int(n), rhs = 1;
        __LAPACK_int info{};
        dgesv_(&count, &rhs, jacobian.data(), &count, pivots.data(), direction.data(), &count, &info);
        if (info) return result;
        bool accepted = false;
        for (double scale = 1.; scale >= 0x1p-30; scale *= .5) {
            for (size_t i = 0; i < n; ++i) trial[i] = compression[i] + scale * direction[i];
            const double norm = evaluate(trial);
            if (norm <= tolerance || norm <= (1 - 1e-4 * scale) * result.Residual) {
                compression.swap(trial);
                result.Residual = norm;
                accepted = true;
                break;
            }
        }
        if (!accepted) return result;
    }
    if (result.Residual > tolerance) return result;
    s.Compression = std::move(compression);
    s.Force = std::move(force);
    s.Displacement = std::move(displacement);
    if (history_length) std::copy(s.Force.begin(), s.Force.end(), s.History.begin() + head * n);
    ++s.Samples;
    result.Converged = true;
    return result;
}
}
