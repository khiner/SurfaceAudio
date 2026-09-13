#include "core/ContactDamping.h"
#include "core/Flexural.h"
#include "core/NormalContact.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>

using namespace surface_audio;
namespace {
void Check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
void BeamBasis() {
    for (const auto boundary : {BeamBoundary::SimplySupported, BeamBoundary::Free}) {
        const BeamProperties p{.Length = .45, .MassPerLength = 15.6, .BendingStiffness = 140, .Boundary = boundary};
        const auto modes = MakeBeamModes(p, 5);
        constexpr unsigned samples = 2048;
        for (unsigned a = 0; a < modes.size(); ++a) {
            const auto &m = modes[a];
            for (double x : {0., p.Length}) {
                if (boundary == BeamBoundary::SimplySupported) Check(std::abs(BeamShape(p, m, x)) < 1e-12, "Pinned displacement boundary");
                else Check(std::abs(BeamShape(p, m, x, 3)) / (1 + std::pow(m.WaveNumber, 3)) < 1e-12, "Free shear boundary");
                Check(std::abs(BeamShape(p, m, x, 2)) / (1 + m.WaveNumber * m.WaveNumber) < 1e-12, "Zero end moment");
            }
            for (unsigned b = 0; b <= a; ++b) {
                double integral{};
                for (unsigned n = 0; n <= samples; ++n) {
                    const double x = p.Length * n / samples, weight = n == 0 || n == samples ? 1 : n & 1 ? 4 :
                                                                                                           2;
                    integral += weight * BeamShape(p, m, x) * BeamShape(p, modes[b], x) * p.Length / (3 * samples);
                }
                Check(std::abs(integral - double(a == b)) < 1e-10, "Independent numerical beam Gram matrix");
            }
            const double x = .317 * p.Length, h = 1e-6;
            const double slope = (BeamShape(p, m, x + h) - BeamShape(p, m, x - h)) / (2 * h);
            Check(std::abs(slope - BeamShape(p, m, x, 1)) / (1 + std::abs(slope)) < 1e-7, "Independent shape derivative");
        }
        if (boundary == BeamBoundary::SimplySupported) {
            double compliance{};
            for (const auto &mode : MakeBeamModes(p, 64))
                compliance += std::pow(BeamShape(p, mode, p.Length / 2) / mode.Omega, 2) / p.MassPerLength;
            // The midpoint compliance of a pinned beam is L^3/(48*EI).
            Check(std::abs(compliance * 48 * p.BendingStiffness / std::pow(p.Length, 3) - 1) < 3e-6, "Modal response matches continuum beam compliance");
        } else {
            Check(std::abs(modes[2].WaveNumber * p.Length - 4.730040744862704) < 1e-13, "First free bending root");
            Check(std::abs(modes[3].WaveNumber * p.Length - 7.853204624095838) < 1e-13, "Second free bending root");
            Check(modes[0].Omega == 0 && modes[1].Omega == 0, "Free rigid modes");
            const auto high = MakeBeamModes(p, 1000);
            for (double x : {0., .001, p.Length / 2, p.Length - .001, p.Length})
                Check(std::isfinite(BeamShape(p, high.back(), x)) && std::abs(BeamShape(p, high.back(), x)) < 10, "High free modes avoid exponential cancellation");
        }
    }
}

void PlateBasis() {
    for (double width : {.013, .4}) {
        const PlateProperties p{.Length = .6, .Width = width, .MassPerArea = 31.2, .BendingStiffness = 1230};
        const auto modes = MakePlateModes(p, 8);
        std::vector<double> reference;
        for (unsigned i = 1; i <= 8; ++i)
            for (unsigned j = 1; j <= 8; ++j)
                reference.push_back(std::sqrt(p.BendingStiffness / p.MassPerArea) * (std::pow(i * std::numbers::pi / p.Length, 2) + std::pow(j * std::numbers::pi / p.Width, 2)));
        std::ranges::sort(reference);
        for (unsigned k = 0; k < modes.size(); ++k) {
            Check(std::abs(modes[k].Omega / reference[k] - 1) < 1e-14, "Plate frequency ordering across aspect ratios");
            Check(std::abs(PlateShape(p, modes[k], 0, width / 2)) < 1e-12, "Plate supported edge");
        }
        for (unsigned a = 0; a < 3; ++a)
            for (unsigned b = 0; b <= a; ++b) {
                double integral{};
                for (unsigned i = 0; i < 32; ++i)
                    for (unsigned j = 0; j < 32; ++j) {
                        const double x = (i + .5) * p.Length / 32, y = (j + .5) * p.Width / 32;
                        integral += PlateShape(p, modes[a], x, y) * PlateShape(p, modes[b], x, y) * p.Length * p.Width / (32 * 32);
                    }
                Check(std::abs(integral - double(a == b)) < 1e-12, "Independent plate Gram matrix");
            }
    }
}

void Dynamics() {
    for (double damping : {0., .02, 2.}) {
        constexpr double dt = .001;
        const auto m = MakeModalDynamics(2.3, 23., damping, dt);
        double q = .01, previous = PreviousMode(m, q, -.03, 0, dt);
        double loss{}, work{}, error{};
        const double initial = ModalEnergy(m, q, previous, dt);
        for (unsigned n = 0; n < 512; ++n) {
            const double force = .13 * std::sin(n * .011), next = AdvanceMode(m, q, previous, force);
            const double velocity = (next - previous) / (2 * dt);
            loss += 2 * m.Mass * m.DampingRatio * m.Omega * velocity * velocity * dt;
            work += force * velocity * dt;
            error = std::max(error, std::abs(ModalEnergy(m, next, q, dt) + loss - work - initial));
            previous = q;
            q = next;
        }
        Check(error < 1e-11, "Exact discrete modal energy/work/dissipation balance");
    }
    const auto falling = MakeModalDynamics(3., 0., 0., .01);
    double q{}, previous = PreviousMode(falling, 0, .2, -3 * 9.81, .01);
    for (unsigned n = 1; n <= 100; ++n) {
        const double next = AdvanceMode(falling, q, previous, -3 * 9.81), time = .01 * n;
        previous = q;
        q = next;
        Check(std::abs(q - (.2 * time - 9.81 * time * time / 2)) < 1e-12, "Rigid translation under gravity");
    }
    bool rejected{};
    try {
        MakeModalDynamics(1, 2000, 0, .001);
    } catch (const std::invalid_argument &) { rejected = true; }
    Check(rejected, "Reject modal stability boundary");
}

void Scalar() {
    const NormalContactLaw law{2e6, 8e4, 1.5, 1.5};
    const std::array j{1.7}, delta{.003}, area{.02}, mobility{.13};
    ContactDampingState state;
    for (double speed : {-100., -1., 0., 1., 100.}) {
        const std::array free{speed};
        const auto result = SolveContactDamping(j, delta, area, free, mobility, law, state);
        const double a = area[0] * law.Stiffness * std::pow(delta[0], 1.5), b = area[0] * law.Damping * std::pow(delta[0], 1.5);
        const double force = std::max(0., (a - b * j[0] * speed) / (1 + b * j[0] * j[0] * mobility[0]));
        Check(result.Converged && std::abs(state.Force[0] - force) < 1e-11, "Scalar damped contact matches closed form");
        Check(std::abs(state.Velocity[0] - speed - mobility[0] * j[0] * force) < 1e-11, "Scalar force and velocity satisfy the same update");
    }
}

std::array<double, 2> Enumerate(std::span<const double> j, std::span<const double> a, std::span<const double> b, std::array<double, 2> free, std::array<double, 2> mobility) {
    for (unsigned mask = 0; mask < (1u << a.size()); ++mask) {
        double aa = 1, ab = 0, ba = 0, bb = 1, x = free[0], y = free[1];
        for (unsigned row = 0; row < a.size(); ++row) {
            if (!(mask & (1u << row))) continue;
            const double u = j[2 * row], v = j[2 * row + 1];
            aa += mobility[0] * b[row] * u * u;
            ab += mobility[0] * b[row] * u * v;
            ba += mobility[1] * b[row] * v * u;
            bb += mobility[1] * b[row] * v * v;
            x += mobility[0] * a[row] * u;
            y += mobility[1] * a[row] * v;
        }
        const double determinant = aa * bb - ab * ba;
        const std::array velocity{(bb * x - ab * y) / determinant, (aa * y - ba * x) / determinant};
        bool consistent = true;
        for (unsigned row = 0; row < a.size(); ++row) {
            const double force = a[row] - b[row] * (j[2 * row] * velocity[0] + j[2 * row + 1] * velocity[1]);
            if ((mask & (1u << row)) ? force < -1e-12 : force > 1e-12) consistent = false;
        }
        if (consistent) return velocity;
    }
    throw std::runtime_error("No consistent enumerated contact set");
}

void CoupledDamping() {
    const std::array jacobian{1., .3, -.2, 1., -1., .4, .5, -1.};
    const std::array penetration{.001, .002, .003, .0015}, area{.1, .2, .3, .1};
    const std::array mobility{.1, .1};
    // An orthonormal embedding exercises both modal and contact-space solves against the same enumerated solution.
    for (unsigned modes : {2u, 8u}) {
        const auto basis = [modes](unsigned k) {
            return std::array{1 / std::sqrt(double(modes)), std::sqrt(2. / modes) * std::cos(std::numbers::pi * (k + .5) / modes)};
        };
        std::vector<double> j(4 * modes), free(modes), mass(modes, mobility[0]);
        for (unsigned row = 0; row < 4; ++row)
            for (unsigned k = 0; k < modes; ++k) j[row * modes + k] = jacobian[2 * row] * basis(k)[0] + jacobian[2 * row + 1] * basis(k)[1];
        ContactDampingState state;
        for (double damping : {0., 2e5, 2e8}) {
            const NormalContactLaw law{1e4, damping, 1, 1};
            std::array<double, 4> a{}, b{};
            for (unsigned row = 0; row < 4; ++row) {
                a[row] = area[row] * law.Stiffness * penetration[row];
                b[row] = area[row] * damping * penetration[row];
            }
            for (const auto input : {std::array{-3., 2.}, std::array{10., 20.}}) {
                const auto expected = Enumerate(jacobian, a, b, input, mobility);
                for (unsigned k = 0; k < modes; ++k) free[k] = input[0] * basis(k)[0] + input[1] * basis(k)[1];
                const auto result = SolveContactDamping(j, penetration, area, free, mass, law, state);
                Check(result.Converged, "Coupled damping converges");
                for (unsigned k = 0; k < modes; ++k)
                    Check(std::abs(state.Velocity[k] - expected[0] * basis(k)[0] - expected[1] * basis(k)[1]) < 1e-9, "Damping matches exhaustive active-set enumeration");
                for (unsigned row = 0; row < 4; ++row) {
                    const double force = std::max(0., a[row] - b[row] * (jacobian[2 * row] * expected[0] + jacobian[2 * row + 1] * expected[1]));
                    Check(std::abs(state.Force[row] - force) < 1e-8, "Damping force satisfies the independent constitutive solution");
                }
            }
        }
        const auto open = SolveContactDamping({}, {}, {}, free, mass, {1e4, 2e5, 1, 1}, state);
        Check(open.Converged && state.Velocity == free, "Released contact restores free velocity with reused scratch storage");
    }
}
void ContactLaw() {
    constexpr double depth = .003, epsilon = 1e-8;
    for (double exponent : {1., 1.5, 2.}) {
        const auto evaluate = [exponent](double d) { return EvaluateNormalContact(d, 0., 2e6, 8e4, exponent, exponent); };
        const auto force = evaluate(depth);
        const double gradient = (evaluate(depth + epsilon).Energy - evaluate(depth - epsilon).Energy) / (2 * epsilon);
        const double tangent = (evaluate(depth + epsilon).Force - evaluate(depth - epsilon).Force) / (2 * epsilon);
        Check(std::abs(gradient / force.Force - 1) < 1e-9 && std::abs(tangent / force.ElasticTangent - 1) < 1e-9, "Normal force and tangent match potential derivatives");
        for (double rate : {-100., 0., 100.}) {
            const auto contact = EvaluateNormalContact(depth, rate, 2e6, 8e4, exponent, exponent);
            Check(contact.Force >= 0 && contact.Dissipation >= 0, "Unilateral damping dissipates energy on approach and release");
        }
        Check(evaluate(-depth).Force == 0 && evaluate(0).Energy == 0, "Open contact has zero force and stored energy");
    }
}
}
int main() {
    try {
        BeamBasis();
        PlateBasis();
        Dynamics();
        ContactLaw();
        Scalar();
        CoupledDamping();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
