#include "core/AudioFile.h"
#include "matusiak2024/Matusiak2024.h"
#include "matusiak2024/Matusiak2024Gpu.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
using namespace surface_audio;
using namespace surface_audio::matusiak2024;
namespace {
void Benchmark(const std::filesystem::path &output, Parameters p, double force, double acceleration, unsigned frames, unsigned voices) {
    if (voices > 4096) throw std::invalid_argument("At most 4096 GPU voices");
    auto gpu = CreateGpu();
    std::vector<BowDrive> drives(voices);
    for (unsigned j = 0; j < voices; ++j) drives[j] = {static_cast<float>(force) * (1 + .0001f * j), 100.f, static_cast<float>(acceleration)};
    const auto start = std::chrono::steady_clock::now();
    const auto batch = RenderStringsGpu(gpu, p, drives, frames);
    const double gpu_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double cpu_seconds{}, error{}, norm{}, transient_error{}, transient_norm{}, energy_error{}, energy_norm{}, balance{};
    unsigned first_difference = frames, failures{}, maximum_iterations{}, restarts{};
    std::array<double, 9> state_error{}, state_norm{};
    for (unsigned j = 0; j < voices; ++j) {
        const auto cpu_start = std::chrono::steady_clock::now();
        auto state = MakeString(p);
        std::vector<double> wave(frames), energy(frames);
        for (unsigned i = 1; i < frames; ++i) {
            const auto sample = Step(state, double(drives[j].Acceleration) * i / p.SampleRate, drives[j].NormalForce);
            wave[i] = sample.BridgeForce;
            energy[i] = TotalEnergy(sample.Stored);
            maximum_iterations = std::max(maximum_iterations, sample.Iterations);
            restarts += sample.Iterations / 25;
        }
        cpu_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - cpu_start).count();
        for (unsigned i = 1; i < frames; ++i) {
            const size_t index = size_t(j) * frames + i;
            const double d = batch.BridgeForce[index] - wave[i];
            error += d * d;
            norm += wave[i] * wave[i];
            if (i / p.SampleRate < .12) {
                transient_error += d * d;
                transient_norm += wave[i] * wave[i];
            }
            if (std::abs(d) > .01) first_difference = std::min(first_difference, i);
            const double ed = batch.StoredEnergy[index] - energy[i];
            energy_error += ed * ed;
            energy_norm += energy[i] * energy[i];
            balance = std::max(balance, std::abs(double(batch.EnergyError[index])));
        }
        if (j == 0) {
            std::vector<float> rounded(wave.begin(), wave.end());
            WriteWave(output / "cpu_rounded_bridge_force.wav", static_cast<uint32_t>(p.SampleRate), 1, rounded);
        }
        failures += batch.FailedSteps[j];
        size_t offset = j * (2 * (state.U.size() + state.W.size()) + 5 * state.Z.size()), field{};
        for (const auto *v : {&state.U, &state.PreviousU, &state.W, &state.PreviousW, &state.Hair, &state.PreviousHair, &state.Z, &state.Velocity, &state.MidpointZ}) {
            for (double x : *v) {
                const double d = batch.FinalState[offset++] - x;
                state_error[field] += d * d;
                state_norm[field] += x * x;
            }
            ++field;
        }
    }
    WriteWave(output / "gpu_bridge_force.wav", static_cast<uint32_t>(p.SampleRate), 1, std::span<const float>(batch.BridgeForce.data(), frames));
    std::ofstream json(output / "gpu.json");
    json << std::setprecision(17) << "{\"voices\":" << voices << ",\"frames\":" << frames << ",\"cpu_fp64_seconds\":" << cpu_seconds << ",\"gpu_fp32_seconds\":" << gpu_seconds
         << ",\"relative_l2\":" << std::sqrt(error / norm) << ",\"first_120ms_relative_l2\":" << std::sqrt(transient_error / transient_norm)
         << ",\"energy_relative_l2\":" << std::sqrt(energy_error / energy_norm) << ",\"gpu_max_energy_balance_j\":" << balance
         << ",\"first_force_difference_above_0_01n_frame\":" << first_difference << ",\"failed_steps\":" << failures
         << ",\"cpu_max_iterations\":" << maximum_iterations << ",\"cpu_branch_initializations\":" << restarts << ",\"final_state_relative_l2\":[";
    for (size_t i = 0; i < 9; ++i) json << (i ? "," : "") << std::sqrt(state_error[i] / std::max(state_norm[i], 1e-100));
    json << "],\"scope\":\"Both paths include full mechanics, stored energy and loss accounting; GPU includes kernel setup, allocation, dispatch and readback. Precisions differ.\"}\n";
}
}
int main(int argc, char **argv) {
    try {
        const std::filesystem::path output = argc > 1 ? argv[1] : "outputs/reproduction/matusiak2024";
        const unsigned frames = argc > 2 ? std::stoul(argv[2]) : 17200;
        const double force = argc > 3 ? std::stod(argv[3]) : 2.3433, acceleration = argc > 4 ? std::stod(argv[4]) : .8722;
        const unsigned curve = argc > 5 ? std::stoul(argv[5]) : 0;
        if (frames < 2 || frames > 441000 || curve > 1) throw std::invalid_argument("Invalid render shape");
        std::filesystem::create_directories(output);
        const unsigned case_index = argc > 6 ? std::stoul(argv[6]) : 0;
        if (case_index > 3) throw std::invalid_argument("Unknown paper case");
        const FrictionParameters<double> calibrated[2][4]{
            {{318600, .0027, .228, .5071, 1.0207, .7, 0}, {370600, .0082, .0513, .7727, 1.0902, .7, 0}, {333200, .05, .1603, .7722, 1.17, .7, 0}, {302670, .0026, .228, .5071, 1.0207, .7, 0}},
            {{240990, .0115, .4, .3382, 1.1489, .7, 1}, {259700, .0034, .1, .6951, 1.0925, .7, 1}, {189700, .00074, .4, .5308, 1.1725, .7, 1}, {331470, .0228, .4, .3365, .9116, .7, 1}}
        };
        const unsigned convention = argc > 9 ? std::stoul(argv[9]) : 0;
        if (convention > 1) throw std::invalid_argument("Unknown numerical convention");
        const Parameters defaults;
        const Parameters p{.SampleRate = argc > 7 ? std::stod(argv[7]) : defaults.SampleRate, .MaterialDensity = convention ? defaults.Tension / (137.2 * 137.2 * std::numbers::pi * defaults.Radius * defaults.Radius) : defaults.MaterialDensity, .Friction = calibrated[curve][case_index], .Convention = static_cast<NumericalConvention>(convention)};
        auto state = MakeString(p);
        std::vector<float> wave(frames);
        std::vector<double> trace(size_t(frames) * 8);
        double max_energy{}, max_residual{};
        unsigned max_iterations{}, branch_initializations{};
        const auto start = std::chrono::steady_clock::now();
        for (unsigned i = 1; i < frames; ++i) {
            const auto s = Step(state, acceleration * i / p.SampleRate, force);
            wave[i] = static_cast<float>(s.BridgeForce);
            const double row[]{s.BridgeForce, s.RelativeVelocity, s.FrictionForce, TotalEnergy(s.Stored), s.EnergyError, s.BristleDissipation, s.InputPower, s.Residual};
            std::copy(std::begin(row), std::end(row), trace.begin() + size_t(i) * 8);
            max_energy = std::max(max_energy, std::abs(s.EnergyError));
            max_residual = std::max(max_residual, s.Residual);
            max_iterations = std::max(max_iterations, s.Iterations);
            branch_initializations += s.Iterations / 25;
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        WriteWave(output / "bridge_force.wav", static_cast<uint32_t>(p.SampleRate), 1, wave);
        std::ofstream raw(output / "trace.f64", std::ios::binary);
        raw.write(reinterpret_cast<const char *>(trace.data()), trace.size() * sizeof(double));
        std::ofstream json(output / "native.json");
        json << std::setprecision(17) << "{\"cpu_seconds\":" << seconds << ",\"frames\":" << frames << ",\"max_energy_error_j\":" << max_energy << ",\"max_residual\":" << max_residual << ",\"max_iterations\":" << max_iterations << ",\"branch_initializations\":" << branch_initializations << "}\n";
        if (argc > 8 && std::stoul(argv[8])) Benchmark(output, p, force, acceleration, frames, std::stoul(argv[8]));
        std::cout << "2024 curve " << curve << ": " << seconds << " s, energy balance " << max_energy << " J\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
