#include "core/AudioFile.h"
#include "matusiak/Matusiak.h"
#include "matusiak/MatusiakGpu.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace {
void Benchmark(const std::filesystem::path &output, uint32_t voices, uint32_t frames, double normal_force) {
    using namespace surface_audio;
    using namespace surface_audio::matusiak;
    auto gpu = CreateGpu();
    std::vector<BowDrive> drives(voices);
    for (uint32_t i = 0; i < voices; ++i) drives[i].NormalForce = static_cast<float>(normal_force) * (1 + .0001f * i);
    const auto start = std::chrono::steady_clock::now();
    const auto batch = RenderStringsGpu(gpu, {}, drives, frames);
    const double gpu_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    WriteWave(output / "gpu_bridge_force.wav", 44100, 1, std::span<const float>(batch.BridgeForce.data(), frames));
    double cpu_seconds{}, error{}, norm{}, maximum{}, worst_voice{}, final_error{}, final_norm{}, max_residual{};
    std::array<double, 9> field_error{}, field_norm{};
    uint64_t failures{};
    for (uint32_t voice = 0; voice < voices; ++voice) {
        const auto cpu_start = std::chrono::steady_clock::now();
        auto state = MakeString();
        std::vector<double> cpu_wave(frames);
        const double ramp = std::ceil(double(drives[voice].Velocity) / double(drives[voice].Acceleration) * 44100) - 1;
        double voice_error{}, voice_norm{};
        for (uint32_t i = 1; i < frames; ++i) {
            const auto sample = Step(state, std::min(double(drives[voice].Velocity), double(drives[voice].Velocity) * i / ramp), drives[voice].NormalForce);
            cpu_wave[i] = sample.BridgeForce;
        }
        cpu_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - cpu_start).count();
        for (uint32_t i = 1; i < frames; ++i) {
            const double d = batch.BridgeForce[size_t(voice) * frames + i] - cpu_wave[i];
            voice_error += d * d;
            voice_norm += cpu_wave[i] * cpu_wave[i];
            maximum = std::max(maximum, std::abs(d));
        }
        error += voice_error;
        norm += voice_norm;
        worst_voice = std::max(worst_voice, std::sqrt(voice_error / std::max(voice_norm, 1e-100)));
        const size_t stride = 2 * (state.U.size() + state.W.size()) + 5 * state.Z.size();
        size_t offset = size_t(voice) * stride;
        size_t field{};
        for (const auto *v : {&state.U, &state.PreviousU, &state.W, &state.PreviousW, &state.Hair, &state.PreviousHair, &state.Z, &state.Velocity, &state.MidpointZ}) {
            for (double x : *v) {
                const double d = batch.FinalState[offset++] - x;
                final_error += d * d;
                final_norm += x * x;
                field_error[field] += d * d;
                field_norm[field] += x * x;
            }
            ++field;
        }
        failures += batch.FailedSteps[voice];
        max_residual = std::max(max_residual, double(batch.MaximumResidual[voice]));
    }
    std::ofstream json(output / "gpu.json");
    json << std::setprecision(17) << "{\n  \"device\": \"" << DeviceName(gpu) << "\", \"voices\": " << voices << ", \"frames\": " << frames << ",\n"
         << "  \"gpu_seconds\": " << gpu_seconds << ", \"cpu_seconds\": " << cpu_seconds << ",\n"
         << "  \"gpu_scope\": \"FP32 mechanics with nonlinear residual checks\", \"cpu_scope\": \"FP64 reference with full energy accounting\",\n"
         << "  \"relative_l2\": " << std::sqrt(error / std::max(norm, 1e-100)) << ", \"worst_voice_relative_l2\": " << worst_voice << ", \"max_force_error_n\": " << maximum << ",\n"
         << "  \"final_state_relative_l2\": " << std::sqrt(final_error / std::max(final_norm, 1e-100)) << ", \"failed_steps\": " << failures << ", \"max_residual\": " << max_residual << ",\n  \"final_state_relative_l2_by_field\": {";
    const std::array names{"U", "PreviousU", "W", "PreviousW", "Hair", "PreviousHair", "Z", "Velocity", "MidpointZ"};
    for (size_t i = 0; i < names.size(); ++i) json << (i ? ", " : "") << "\"" << names[i] << "\": " << std::sqrt(field_error[i] / std::max(field_norm[i], 1e-100));
    json << "}\n}\n";
    std::cout << "Distributed GPU: " << voices << " voices, " << gpu_seconds << " s; CPU reference " << cpu_seconds << " s, relative L2 " << std::sqrt(error / std::max(norm, 1e-100)) << ", failed " << failures << '\n';
    if (failures) throw std::runtime_error("GPU trajectory contains unconverged steps");
}
}
int main(int argc, char **argv) {
    try {
        const std::filesystem::path output = argc > 1 ? argv[1] : "outputs/reproduction/matusiak";
        const double seconds = argc > 2 ? std::stod(argv[2]) : .5;
        const double normal_force = argc > 3 ? std::stod(argv[3]) : 2.3433;
        const unsigned long requested_voices = argc > 4 ? std::stoul(argv[4]) : 0;
        if (requested_voices > 4096) throw std::invalid_argument("At most 4096 GPU voices");
        const unsigned gpu_voices = static_cast<unsigned>(requested_voices);
        if (!std::isfinite(seconds) || seconds <= 0 || seconds > 60) throw std::invalid_argument("Duration must be in (0,60]");
        std::filesystem::create_directories(output);
        auto state = surface_audio::matusiak::MakeString();
        const size_t frames = static_cast<size_t>(std::ceil(seconds * state.Config.SampleRate));
        if (frames < 2) throw std::invalid_argument("At least two samples are required");
        const double ramp = std::ceil(.3439 / .8722 * state.Config.SampleRate) - 1;
        std::vector<float> wave(frames);
        std::ofstream trace(output / "trace.f64", std::ios::binary);
        const double zero[8]{};
        trace.write(reinterpret_cast<const char *>(zero), sizeof zero);
        double energy_error{}, residual{}, minimum_bristle{}, peak{};
        uint64_t iterations{};
        uint32_t maximum_iterations{};
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 1; i < frames; ++i) {
            const auto sample = surface_audio::matusiak::Step(state, std::min(.3439, .3439 * i / ramp), normal_force);
            wave[i] = static_cast<float>(sample.BridgeForce);
            if (!std::isfinite(wave[i])) throw std::runtime_error("Nonfinite bridge waveform");
            const double row[]{sample.BridgeForce, sample.RelativeVelocity, sample.FrictionForce, surface_audio::matusiak::TotalEnergy(sample.Stored), sample.EnergyError, sample.BristleDissipation, sample.Residual, double(sample.Iterations)};
            trace.write(reinterpret_cast<const char *>(row), sizeof row);
            energy_error = std::max(energy_error, std::abs(sample.EnergyError));
            residual = std::max(residual, sample.Residual);
            minimum_bristle = std::min(minimum_bristle, sample.BristleDissipation);
            peak = std::max(peak, std::abs(double(wave[i])));
            iterations += sample.Iterations;
            maximum_iterations = std::max(maximum_iterations, sample.Iterations);
        }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        surface_audio::WriteWave(output / "bridge_force.wav", 44100, 1, wave);
        const double listening_gain = peak > 0 ? .95 / peak : 1;
        for (float &v : wave) v = static_cast<float>(double(v) * listening_gain);
        surface_audio::WriteWave(output / "bridge_listening.wav", 44100, 1, wave);
        std::ofstream manifest(output / "native.json");
        manifest << std::setprecision(17) << "{\n  \"frames\": " << frames << ", \"sample_rate\": 44100, \"normal_force\": " << normal_force << ",\n  \"seconds\": " << elapsed << ", \"real_time_factor\": " << elapsed / seconds << ",\n  \"max_energy_error_j\": " << energy_error << ", \"max_residual\": " << residual << ", \"minimum_bristle_dissipation_w\": " << minimum_bristle << ",\n  \"max_iterations\": " << maximum_iterations << ", \"mean_iterations\": " << double(iterations) / (frames - 1) << ",\n  \"transverse_intervals\": " << state.U.size() + 1 << ", \"torsional_intervals\": " << state.W.size() + 1 << ", \"contacts\": " << state.Z.size() << ", \"listening_gain\": " << listening_gain << "\n}\n";
        std::cout << "Matusiak: " << frames << " frames, " << elapsed << " s, energy error " << energy_error << " J, residual " << residual << '\n';
        if (gpu_voices) Benchmark(output, gpu_voices, static_cast<uint32_t>(frames), normal_force);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
