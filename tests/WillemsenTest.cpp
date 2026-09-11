#include "willemsen/Willemsen.h"
#include "core/Random.h"
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <sstream>
#include <stdexcept>
using namespace surface_audio;
using namespace surface_audio::willemsen;
namespace {
void Check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
void Equations() {
    const auto m = MakeModel();
    for (double v : {-.4, -.1, -.015, .015, .1, .4})
        for (double z : {-.0004, -.00018, -.00011, 0., .00011, .00018, .0004}) {
            const auto f = Evaluate(m, 5., v, z);
            const double dz = 1e-10, dv = 1e-7;
            Check(std::abs(f.RateVelocity - (Evaluate(m, 5., v + dv, z).Rate - Evaluate(m, 5., v - dv, z).Rate) / (2 * dv)) < 2e-7, "Eq15 velocity Jacobian");
            Check(std::abs(f.RateState - (Evaluate(m, 5., v, z + dz).Rate - Evaluate(m, 5., v, z - dz).Rate) / (2 * dz)) < 2e-5, "Eq15 bristle Jacobian");
            const auto opposite = Evaluate(m, 5., -v, -z);
            Check(std::abs(f.Rate + opposite.Rate) < 1e-14 && f.Adhesion == opposite.Adhesion, "Sign symmetric adhesion map");
        }
    double sum{};
    for (double w : m.Weight) sum += w;
    Check(std::abs(sum - 1) < 1e-15, "Cubic partition of unity");
    Parameters p;
    for (const auto pair : {std::pair{196., 95u}, {293.66, 71u}, {440., 49u}, {659.26, 33u}}) {
        p.Fundamental = pair.first;
        const auto model = MakeModel(p);
        Check(model.Intervals == pair.second, "Published Section5 grid counts");
    }
    State<double> s;
    for (unsigned j = 0; j <= m.Intervals; ++j) s.U[j] = s.Previous[j] = 1e-4 * std::sin(std::numbers::pi * j / m.Intervals);
    const double theta = std::numbers::pi / m.Intervals;
    const double expected = m.Update[0] + 2 * m.Update[1] * std::cos(theta) + 2 * m.Update[2] * std::cos(2 * theta) + m.Update[3] + 2 * m.Update[4] * std::cos(theta);
    Step(m, s, 0., 0., 0., 1e-7);
    for (unsigned j = 1; j < m.Intervals; ++j) Check(std::abs(s.U[j] - expected * 1e-4 * std::sin(theta * j)) < 1e-18, "Simply-supported mode stencil including endpoints");
    s = {};
    double residual{}, kinematic{};
    for (unsigned i = 0; i < 44100; ++i) {
        const double old_z = s.Z, old_rate = s.Rate;
        const auto a = Step(m, s, .1, 5., 0., 1e-7);
        Check(!a.Failed && std::isfinite(a.Displacement), "One-second Figure8 trajectory converges");
        residual = std::max(residual, a.Residual);
        kinematic = std::max(kinematic, a.KinematicResidual);
        Check(std::abs(s.Z - old_z - (s.Rate + old_rate) / (2 * m.SampleRate)) < 1e-12, "Independent trapezoidal bristle update");
    }
    Check(residual < 1e-7 && kinematic < 1e-7, "Nonlinear and interpolation kinematic residual");
}
void Figure() {
    const auto m = MakeModel({.Noise = 0, .Discretization = Scheme::AuthorFigure});
    auto s = MakeState(m, .1);
    std::vector<double> pickup(23000), bow(23000);
    for (unsigned i = 0; i < pickup.size(); ++i) {
        const auto a = Step(m, s, .1, 5., 0., 1e-7);
        pickup[i] = a.Displacement;
        bow[i] = a.BowDisplacement;
        Check(!a.Failed, "Author figure Newton convergence");
    }
    const auto directory = std::filesystem::path(__FILE__).parent_path().parent_path() / "repros/willemsen";
    for (const auto name : {"figure8_pickup.csv", "figure8_bow.csv"}) {
        std::ifstream input(directory / name);
        Check(bool(input), "Published EPS fixture exists");
        std::string line;
        std::getline(input, line);
        double error{}, norm{};
        unsigned count{};
        while (std::getline(input, line)) {
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream row(line);
            double time{}, value{};
            row >> time >> value;
            const unsigned index = unsigned(std::round(time * 44100)) - 1;
            const double actual = (std::string_view(name) == "figure8_pickup.csv" ? pickup : bow)[index], d = actual - value;
            error += d * d;
            norm += value * value;
            ++count;
        }
        Check(count == 442 && std::sqrt(error / norm) < 3e-5, "Unaligned published Figure8 waveform within EPS coordinate precision");
    }
}
void Metal(Scheme scheme) {
    Parameters p{.Noise = 0, .Discretization = scheme};
    const auto m = MakeFloatModel(p);
    const std::vector<Drive> drives{{.1f, 5, 2019}, {-.1f, 5, 2019}, {.12f, 4, 2020}, {0, 0, 2019}};
    auto gpu = CreateGpu();
    const unsigned frames = scheme == Scheme::AuthorFigure ? 44100 : 4096;
    const auto batch = RenderGpu(gpu, p, drives, frames);
    double wave_error{}, wave_norm{}, state_error{}, state_norm{};
    std::array<double, 5> field_error{}, field_norm{};
    for (unsigned j = 0; j < drives.size(); ++j) {
        auto s = MakeState(m, drives[j].Velocity);
        auto random = MakeRandom(drives[j].Seed);
        for (unsigned i = 0; i < frames; ++i) {
            const auto sample = Step(m, s, drives[j].Velocity, drives[j].NormalForce, 2 * Uniform(random) - 1, 2e-6f);
            const double value = sample.Displacement, d = value - batch.Displacement[j * frames + i];
            wave_error += d * d;
            wave_norm += value * value;
        }
        size_t index = j * (2 * (m.Intervals + 1) + 3);
        unsigned field{};
        const auto compare = [&](double value) { const double d=value-batch.FinalState[index++]; state_error+=d*d; state_norm+=value*value; field_error[field]+=d*d; field_norm[field]+=value*value; };
        for (unsigned i = 0; i <= m.Intervals; ++i) compare(s.U[i]);
        ++field;
        for (unsigned i = 0; i <= m.Intervals; ++i) compare(s.Previous[i]);
        ++field;
        compare(s.Z);
        ++field;
        compare(s.Rate);
        ++field;
        compare(s.Velocity);
        Check(batch.FailedSteps[j] == 0, "Metal nonlinear solve converges");
    }
    const double wave = std::sqrt(wave_error / wave_norm), state = std::sqrt(state_error / state_norm);
    std::cout << "Metal wave relative L2 " << wave << ", full state relative L2 " << state << '\n';
    Check(wave < .001 && state < .002, "Complete CPU/Metal trajectory agreement");
    for (unsigned i = 0; i < 5; ++i) Check(std::sqrt(field_error[i] / std::max(field_norm[i], 1e-100)) < .002, "Each complete state field agrees");
}
}
int main(int argc, char **) {
    try {
        Equations();
        Figure();
        if (argc == 1) {
            Metal(Scheme::Paper);
            Metal(Scheme::AuthorFigure);
        }
        std::cout << "Willemsen tests passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
