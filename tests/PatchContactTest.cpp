#include "rough/PatchContact.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace surface_audio::rough;
namespace {
void Check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
template<typename Exception> bool Throws(auto call) {
    try { call(); } catch (const Exception &) { return true; }
    return false;
}
PatchSpring Quadratic(double coefficient) {
    return MakePatchSpring(std::array{std::array{0., 0.}, std::array{.0005, .001 * coefficient}, std::array{.001, .002 * coefficient}});
}
void SpringLaw() {
    const auto spring = MakePatchSpring(std::array{std::array{0., 0.}, std::array{.001, 2000.}, std::array{.003, 6000.}});
    for (double x : {-.001, 0., .0002, .001, .002, .003}) {
        const auto value = EvaluatePatchSpring(spring, x);
        const double positive = std::max(0., x);
        Check(std::abs(value[0] - 1e6 * positive * positive) < 1e-14 && std::abs(value[1] - 2e6 * positive) < 1e-11,
              "Tabulated stiffness integrates to the analytical quadratic force across knots");
    }
    Check(Throws<std::out_of_range>([&] { EvaluatePatchSpring(spring, .00301); }), "A finite spring table does not extrapolate");
    Check(Throws<std::invalid_argument>([] { MakePatchSpring(std::array{std::array{0., 2000.}, std::array{.001, 1000.}}); }),
          "Softening is outside the monotone nonadhesive spring model");
}
void SingleContact() {
    constexpr double coefficient = 2e9, compliance = 2e-6, dt = 1. / 51200;
    const auto model = MakePatchContact({Quadratic(coefficient)}, 1, dt, std::array{compliance / dt});
    auto state = MakePatchContactState(model);
    for (double approach : {-.001, 0., .0001, .0005, .002, .004, .001, -.001}) {
        const auto result = StepPatchContact(model, state, std::array{approach}, 1e-14);
        const double compression = approach > 0 ? 2 * approach / (1 + std::sqrt(1 + 4 * compliance * coefficient * approach)) : approach;
        const double force = coefficient * std::pow(std::max(0., compression), 2);
        Check(result.Converged && std::abs(state.Compression[0] - compression) < 1e-14 && std::abs(state.Force[0] - force) < 1e-8,
              "One nonlinear spring in series with bulk compliance matches the closed-form root during loading and unloading");
    }
    constexpr double approach = .0008, delta = 1e-7;
    Check(StepPatchContact(model, state, std::array{approach}, 1e-15).Converged, "Central static load converges");
    const double local = 2 * coefficient * state.Compression[0], expected = local / (1 + compliance * local);
    Check(StepPatchContact(model, state, std::array{approach - delta}, 1e-15).Converged, "Lower static load converges");
    const double lower = state.Force[0];
    Check(StepPatchContact(model, state, std::array{approach + delta}, 1e-15).Converged, "Upper static load converges");
    Check(std::abs((state.Force[0] - lower) / (2 * delta) / expected - 1) < 1e-7,
          "Contact tangent stiffness matches the analytical series-compliance derivative");
}
void CoupledContacts() {
    const auto first = MakePatchSpring(std::array{std::array{0., 2e5}, std::array{.01, 2e5}});
    const auto second = MakePatchSpring(std::array{std::array{0., 9e5}, std::array{.01, 9e5}});
    constexpr double dt = .01, c00 = 2e-6, c01 = .3e-6, c11 = 1e-6;
    const auto model = MakePatchContact({first, second}, 1, dt, std::array{c00 / dt, c01 / dt, c01 / dt, c11 / dt});
    auto state = MakePatchContactState(model);
    constexpr double a = 1 / 2e5 + c00, b = 1 / 9e5 + c11, determinant = a * b - c01 * c01;
    const std::array approach{.001, .0008};
    const std::array expected{(b * approach[0] - c01 * approach[1]) / determinant, (a * approach[1] - c01 * approach[0]) / determinant};
    Check(StepPatchContact(model, state, approach, 1e-14).Converged, "Two coupled elements converge");
    for (unsigned i = 0; i < 2; ++i) Check(std::abs(state.Force[i] - expected[i]) < 1e-8, "Coupled forces match the independent compliance-matrix inverse");
    Check(StepPatchContact(model, state, std::array{.001, -.001}, 1e-14).Converged, "One contact can release while the other stays loaded");
    Check(std::abs(state.Force[0] - .001 / a) < 1e-8 && state.Force[1] == 0 && state.Compression[1] < 0,
          "Released elements retain bulk displacement without tensile spring force");
}
void DynamicHistory() {
    constexpr unsigned elements = 3, taps = 4, steps = 80;
    constexpr double dt = 1. / 51200;
    const std::array coefficients{2e9, 5e9, 1e9};
    const std::array weights{.4, -.1, .3, .4};
    std::vector<double> green(taps * elements * elements);
    for (unsigned lag = 0; lag < taps; ++lag)
        for (unsigned i = 0; i < elements; ++i)
            for (unsigned j = 0; j < elements; ++j) green[(lag * elements + i) * elements + j] = weights[lag] * (i == j ? 1.5e-6 : .2e-6) / dt;
    const auto model = MakePatchContact({Quadratic(coefficients[0]), Quadratic(coefficients[1]), Quadratic(coefficients[2])}, taps, dt, green);
    auto state = MakePatchContactState(model);
    std::array<std::array<double, elements>, steps> expected_force{}, expected_compression{};
    for (unsigned step = 0; step < steps; ++step)
        for (unsigned i = 0; i < elements; ++i) {
            expected_compression[step][i] = .0003 * std::sin(.23 * step + .7 * i);
            expected_force[step][i] = coefficients[i] * std::pow(std::max(0., expected_compression[step][i]), 2);
        }
    double force_error{}, displacement_error{};
    for (unsigned step = 0; step < steps; ++step) {
        std::array<double, elements> displacement{}, approach{};
        for (unsigned i = 0; i < elements; ++i) {
            for (unsigned lag = 0; lag < taps && lag <= step; ++lag)
                for (unsigned j = 0; j < elements; ++j)
                    displacement[i] += dt * green[(lag * elements + i) * elements + j] * expected_force[step - lag][j];
            approach[i] = expected_compression[step][i] + displacement[i];
        }
        if (step == 17) {
            const auto before = state;
            const auto failed = StepPatchContact(model, state, std::array{1., 1., 1.}, 1e-14);
            Check(!failed.Converged && state == before, "An out-of-domain solve preserves forces and the full convolution history");
        }
        if (step == 2) {
            const auto before = state;
            const auto limited = StepPatchContact(model, state, approach, 1e-14, 1);
            Check(!limited.Converged && state == before, "Iteration exhaustion does not advance time");
        }
        Check(StepPatchContact(model, state, approach, 1e-14).Converged, "Manufactured multipoint trajectory converges");
        for (unsigned i = 0; i < elements; ++i) {
            force_error = std::max(force_error, std::abs(state.Force[i] - expected_force[step][i]));
            displacement_error = std::max(displacement_error, std::abs(state.Displacement[i] - displacement[i]));
            Check(std::abs(state.Compression[i] - expected_compression[step][i]) < 2e-14, "Manufactured compression is recovered through contact loss and re-entry");
        }
    }
    Check(force_error < 1e-7 && displacement_error < 2e-14, "Causal history matches independent full convolution across ring-buffer wraps");
    const auto before = state;
    Check(Throws<std::invalid_argument>([&] { StepPatchContact(model, state, std::array{double(NAN), 0., 0.}); }) && state == before,
          "Nonfinite input is rejected without changing history");
    std::cout << "Manufactured trajectory: maximum force error " << force_error << " N, displacement error " << displacement_error << " m\n";
}
void PaperSizedPatch() {
    constexpr unsigned elements = 400, taps = 256, steps = 260;
    constexpr double dt = 1. / 51200, coefficient = 2e9, diagonal = 1e-6, coupling = 2e-9;
    std::array<double, taps> weights{};
    double sum{};
    for (unsigned lag = 0; lag < taps; ++lag) sum += weights[lag] = std::exp(-double(lag) / 30);
    for (double &weight : weights) weight /= sum;
    std::vector<double> green(size_t(taps) * elements * elements);
    for (unsigned lag = 0; lag < taps; ++lag)
        for (unsigned i = 0; i < elements; ++i)
            for (unsigned j = 0; j < elements; ++j)
                green[(size_t(lag) * elements + i) * elements + j] = weights[lag] * (i == j ? diagonal : coupling) / dt;
    const auto model = MakePatchContact(std::vector<PatchSpring>(elements, Quadratic(coefficient)), taps, dt, green);
    green.clear(); green.shrink_to_fit();
    auto state = MakePatchContactState(model);
    std::vector<double> approach(elements), forces(steps);
    for (unsigned step = 0; step < steps; ++step) {
        const double compression = 1e-5 * (1 + std::sin(.03 * step));
        forces[step] = coefficient * compression * compression;
        double displacement{};
        for (unsigned lag = 0; lag < taps && lag <= step; ++lag)
            displacement += weights[lag] * forces[step - lag] * (diagonal + (elements - 1) * coupling);
        std::fill(approach.begin(), approach.end(), compression + displacement);
        Check(StepPatchContact(model, state, approach, 1e-14).Converged, "The 20 by 20 patch with 256 response taps converges");
        for (double force : state.Force) Check(std::abs(force - forces[step]) < 1e-8, "The full-sized symmetric patch matches its scalar convolution oracle");
    }
    std::cout << "400-element patch: 256 taps, 260 time steps, scalar convolution oracle passed\n";
}
}
int main() {
    try {
        SpringLaw();
        SingleContact();
        CoupledContacts();
        DynamicHistory();
        PaperSizedPatch();
        std::cout << "Interfacial contact checks passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
