#include "core/AudioFile.h"
#include "core/Random.h"
#include "willemsen/Willemsen.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace surface_audio;
using namespace surface_audio::willemsen;
namespace {
void WriteListen(const std::filesystem::path &path, std::vector<float> wave) {
    float peak{};
    for (float x : wave) peak = std::max(peak, std::abs(x));
    if (peak > 0)
        for (auto &x : wave) x *= .8f / peak;
    WriteWave(path, 44100, 1, wave);
}
void Benchmark(const std::filesystem::path &output, unsigned frames, unsigned voices) {
    if (!voices || voices > 4096) throw std::invalid_argument("Willemsen benchmark requires 1 to 4096 voices");
    const Parameters p{.Noise = 0, .Discretization = Scheme::AuthorFigure};
    std::vector<Drive> drives(voices);
    for (unsigned i = 0; i < voices; ++i) drives[i].Velocity += float(i % 17) * .00001f;
    auto gpu = CreateGpu();
    const auto begin = std::chrono::steady_clock::now();
    const auto batch = RenderGpu(gpu, p, drives, frames);
    const double gpu_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    const auto model = MakeFloatModel(p);
    double cpu_seconds{}, wave_error{}, wave_norm{}, state_error{}, state_norm{}, residual{};
    std::array<double, 5> field_error{}, field_norm{};
    unsigned failures{};
    for (unsigned j = 0; j < voices; ++j) {
        auto state = MakeState(model, drives[j].Velocity);
        auto random = MakeRandom(drives[j].Seed);
        const auto start = std::chrono::steady_clock::now();
        std::vector<float> wave(frames);
        for (unsigned i = 0; i < frames; ++i) wave[i] = Step(model, state, drives[j].Velocity, drives[j].NormalForce, 2 * Uniform(random) - 1, 2e-6f).Displacement;
        cpu_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        for (unsigned i = 0; i < frames; ++i) {
            const double d = double(wave[i]) - batch.Displacement[size_t(j) * frames + i];
            wave_error += d * d;
            wave_norm += double(wave[i]) * wave[i];
        }
        size_t index = j * (2 * (model.Intervals + 1) + 3);
        unsigned field{};
        const auto compare = [&](double value) { const double d = value - batch.FinalState[index++]; state_error += d*d; state_norm += value*value; field_error[field] += d*d; field_norm[field] += value*value; };
        for (unsigned i = 0; i <= model.Intervals; ++i) compare(state.U[i]);
        ++field;
        for (unsigned i = 0; i <= model.Intervals; ++i) compare(state.Previous[i]);
        ++field;
        compare(state.Z);
        ++field;
        compare(state.Rate);
        ++field;
        compare(state.Velocity);
        residual = std::max(residual, double(batch.Residual[j]));
        failures += batch.FailedSteps[j];
    }
    std::ofstream report(output / "gpu.json");
    report << std::setprecision(17) << "{\"device\":\"" << DeviceName(gpu) << "\",\"voices\":" << voices << ",\"frames\":" << frames
           << ",\"cpu_fp32_seconds\":" << cpu_seconds << ",\"gpu_fp32_seconds\":" << gpu_seconds
           << ",\"wave_relative_l2\":" << std::sqrt(wave_error / wave_norm) << ",\"full_state_relative_l2\":" << std::sqrt(state_error / state_norm)
           << ",\"max_residual_m_per_s\":" << residual << ",\"failed_steps\":" << failures
           << ",\"state_fields_relative_l2\":[";
    for (unsigned field = 0; field < 5; ++field) report << (field ? "," : "") << std::sqrt(field_error[field] / std::max(field_norm[field], 1e-100));
    report << "],\"state_field_order\":[\"u\",\"previous_u\",\"z\",\"rate\",\"velocity\"],"
           << "\"scope\":\"Matched FP32 complete trajectories; GPU includes allocation, kernel setup, dispatch, synchronization and readback.\"}\n";
}
}
int main(int argc, char **argv) {
    try {
        const std::filesystem::path output = argc > 1 ? argv[1] : "outputs/reproduction/willemsen";
        const unsigned frames = argc > 2 ? std::stoul(argv[2]) : 44100;
        if (!frames || frames > 4410000) throw std::invalid_argument("Invalid Willemsen duration");
        std::filesystem::create_directories(output);
        if (argc > 3) {
            Benchmark(output, frames, std::stoul(argv[3]));
            return 0;
        }
        const Parameters p{.Noise = 0, .Discretization = Scheme::AuthorFigure};
        const auto m = MakeModel(p);
        auto state = MakeState(m, .1);
        std::ofstream trace(output / "trajectory.csv");
        trace << "frame,displacement,bow_displacement,force,velocity,z,rate\n"
              << std::setprecision(17);
        std::vector<float> wave(frames);
        double residual{}, kinematic{};
        unsigned failures{}, iterations{};
        for (unsigned i = 0; i < frames; ++i) {
            const auto s = Step(m, state, .1, 5., 0., 1e-7);
            wave[i] = float(s.Displacement);
            trace << i << ',' << s.Displacement << ',' << s.BowDisplacement << ',' << s.Force << ',' << s.Velocity << ',' << state.Z << ',' << state.Rate << '\n';
            residual = std::max(residual, s.Residual);
            kinematic = std::max(kinematic, s.KinematicResidual);
            failures += s.Failed;
            iterations = std::max(iterations, s.Iterations);
        }
        WriteWave(output / "figure_displacement_m.wav", 44100, 1, wave);
        WriteListen(output / "figure_listen.wav", wave);
        const auto literal = MakeModel({.Noise = 0});
        State<double> literal_state;
        std::vector<float> literal_wave(frames);
        for (unsigned i = 0; i < frames; ++i) literal_wave[i] = float(Step(literal, literal_state, .1, 5., 0., 1e-7).Displacement);
        WriteWave(output / "paper_displacement_m.wav", 44100, 1, literal_wave);
        WriteListen(output / "paper_listen.wav", literal_wave);
        std::vector<float> strings;
        for (double fundamental : {196., 293.66, 440., 659.26}) {
            const auto model = MakeModel({.Fundamental = fundamental, .Noise = .02, .Discretization = Scheme::AuthorFigure});
            auto note = MakeState(model, .1);
            auto random = MakeRandom(2019);
            for (unsigned i = 0; i < 88200; ++i) {
                const double time = i / 44100.;
                const double velocity = .1 * std::min(time / .03, 1.);
                const auto s = Step(model, note, velocity, time < 1.6 ? 5. : 0., double(2 * Uniform(random) - 1), 1e-7);
                strings.push_back(float(s.Displacement));
                failures += s.Failed;
            }
        }
        WriteListen(output / "four_open_strings_listen.wav", strings);
        std::ofstream report(output / "cpu.json");
        report << std::setprecision(17) << "{\"intervals\":" << m.Intervals << ",\"frames\":" << frames << ",\"failed_steps\":" << failures
               << ",\"max_newton_iterations\":" << iterations << ",\"max_residual_m_per_s\":" << residual
               << ",\"max_kinematic_residual_m_per_s\":" << kinematic << "}\n";
        std::cout << "Willemsen: " << m.Intervals << " intervals, " << failures << " failed steps, residual " << residual << '\n';
        return failures ? 1 : 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
