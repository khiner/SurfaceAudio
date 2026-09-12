#include "hatt/Hatt.h"
#include "core/LineSpectrum.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <numbers>

using namespace surface_audio;
namespace {
void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
}
int main() {
    try {
        constexpr double pi = std::numbers::pi;
        const auto even = LineSpectrumPolynomial(std::array{pi / 3, 2 * pi / 3});
        const auto odd = LineSpectrumPolynomial(std::array{pi / 2});
        Require(even.size() == 3 && std::abs(even[1]) + std::abs(even[2]) < 1e-14, "Even LSF polynomial");
        Require(odd.size() == 2 && std::abs(odd[1]) < 1e-14, "Odd LSF polynomial");
        for (size_t order = 1; order <= 25; ++order) {
            std::vector<double> lsf(order);
            for (size_t i = 0; i < order; ++i) lsf[i] = pi * (i + 1) / (order + 1) + .007 * std::sin(i + 1);
            const auto coefficients = LineSpectrumPolynomial(lsf);
            for (double angle : std::array{.13, .9, 2.4}) {
                const auto z = std::polar(1., -angle);
                std::complex<double> sum = 1, difference = 1;
                for (size_t i = 0; i < order; ++i) {
                    const auto root = std::polar(1., lsf[i]);
                    (i % 2 ? difference : sum) *= (1. - root * z) * (1. - std::conj(root) * z);
                }
                if (order % 2) difference *= 1. - z * z;
                else {
                    sum *= 1. + z;
                    difference *= 1. - z;
                }
                std::complex<double> evaluated = 0;
                for (size_t i = coefficients.size(); i > 0; --i) evaluated = evaluated * z + coefficients[i - 1];
                Require(std::abs(evaluated - .5 * (sum + difference)) < 1e-8, "Polynomial agrees with independent complex root products");
            }
        }
        bool rejected = false;
        try {
            LineSpectrumPolynomial(std::array{1., .5});
        } catch (const std::invalid_argument &) { rejected = true; }
        Require(rejected, "Reject unordered LSFs");
        hatt::Texture texture{1000, .5, 100, 2, {{0, 0, 0, 1, {std::acos(.5)}, {}}, {100, 0, 0, 1, {std::acos(.5)}, {}}, {0, 2, 0, 1, {std::acos(.5)}, {}}, {100, 2, 4, 1, {std::acos(.5)}, {}}}, {{0, 1, 2}, {1, 3, 2}}};
        hatt::Validate(texture);
        const auto anchor = hatt::Interpolate(texture, {100, 2});
        const auto center = hatt::Interpolate(texture, {75, 1.5});
        Require(std::abs(anchor.Ar[1] + .5) < 1e-14 && std::abs(center.Variance - 2) < 1e-14, "Barycentric interpolation");
        Require(hatt::Interpolate(texture, {-1, 1}).Variance == 0 && hatt::Interpolate(texture, {1, -1}).Variance == 0, "Silent boundary models");
        Require(hatt::Interpolate(texture, {1000, 20}).Variance == 4, "Force and speed saturation");
        hatt::State state;
        Require(std::abs(hatt::Tick(anchor, state, 1) - 2) < 1e-14, "Variance sets innovation power");
        for (int i = 1; i < 15; ++i) Require(std::abs(hatt::Tick(anchor, state, 0) - 2 * std::pow(.5, i)) < 1e-14, "Causal AR impulse response");
        Require(hatt::FrictionForce(0, 2, .5) == 0 && hatt::FrictionForce(10, 2, .5) == -.08 && hatt::FrictionForce(-200, 2, .5) == 1, "Viscous and Coulomb friction");
        Require(std::abs(hatt::FrictionForce(125 - 1e-8, 2, .5) + 1) < 1e-9, "Continuous friction threshold");
        Require(hatt::FrictionForce(10, 2, .5, true) == -.04, "Preserve upstream extra friction coefficient");
        std::vector<float> noise(4096);
        for (size_t i = 0; i < noise.size(); ++i) noise[i] = float(std::sin(i * 1.73));
        auto gpu = CreateGpu();
        const std::vector<hatt::Texture> textures(4, texture);
        const std::vector<hatt::Control> controls(noise.size(), {100, 2});
        const auto actual = hatt::RenderGpu(gpu, textures, controls, noise);
        for (uint32_t voice = 0; voice < 4; ++voice) {
            hatt::State reference;
            for (uint32_t i = 0; i < 1024; ++i) {
                const auto index = voice * 1024 + i;
                Require(std::abs(actual[index] - hatt::Tick(anchor, reference, noise[index])) < 1e-6, "Independent GPU filter histories");
            }
        }
        const std::vector<hatt::Control> stationary(noise.size(), {0, 1});
        const auto silence = hatt::RenderGpu(gpu, textures, stationary, noise);
        for (float sample : silence) Require(std::abs(sample) < 1e-10f, "Zero-speed GPU texture is silent");
        std::cout << "HaTT interpolation, filter, friction and GPU checks passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
