#include "core/AudioFile.h"
#include "nakatsuka/Nakatsuka.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace surface_audio;
int main(int argc, char **argv) {
    try {
        if (argc < 3 || argc > 4) throw std::invalid_argument("Usage: nakatsukaReproduce settings.txt output [seconds=2]");
        std::ifstream settings(argv[1]);
        settings.exceptions(std::ios::badbit | std::ios::failbit);
        uint32_t count{};
        settings >> count;
        if (!count || count > 16) throw std::invalid_argument("Invalid scene count");
        const std::filesystem::path output(argv[2]);
        std::filesystem::create_directories(output);
        std::ofstream metrics(output / "render.json");
        metrics << std::setprecision(12) << "{\"cases\":[";
        for (uint32_t case_index = 0; case_index < count; ++case_index) {
            nakatsuka::Scene scene;
            nakatsuka::Material material;
            nakatsuka::AcousticSettings acoustic;
            settings >> scene.Columns >> scene.Rows >> scene.Speed >> scene.SphereRadius >> scene.InitialHeight >> scene.ShiftedAdhesion >> scene.MotionStart >> scene.MotionDuration >> scene.SettlingTime >> scene.OriginY >> scene.MotionRise >> scene.MotionFall >> scene.Adhesion >> scene.AdhesionDistance;
            settings >> material.Modulus >> material.Poisson >> material.ArealDensity >> material.Thickness >> material.Width >> material.Height >> material.WidthDeviation >> material.HeightDeviation;
            settings >> acoustic.ReferenceSpeed >> acoustic.OddModes;
            const uint32_t frames = uint32_t((argc > 3 ? std::stod(argv[3]) : 2) * scene.SampleRate);
            auto gpu = CreateGpu();
            const auto patches = nakatsuka::MakePatches(material, scene.Columns * scene.Rows, 42);
            const auto start = std::chrono::steady_clock::now();
            const auto simulation = nakatsuka::SimulateGpu(gpu, scene, frames);
            const double simulation_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const auto begin = std::chrono::steady_clock::now();
            const auto sound = nakatsuka::RenderGpu(gpu, material, acoustic, patches, simulation.Samples);
            const double synthesis_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            double square = 0, active = 0;
            for (float x : sound) {
                if (!std::isfinite(x)) throw std::runtime_error("Nonfinite Nakatsuka sound");
                square += double(x) * x;
            }
            for (const auto &sample : simulation.Samples) active += sample.Contact;
            if (!square || !active) throw std::runtime_error("Nakatsuka scene produced no contact sound");
            WriteWave(output / (std::to_string(case_index) + ".wav"), 44100, 1, sound);
            const size_t checked = std::min<size_t>(16, patches.size() - patches.size() / 2);
            const size_t offset = patches.size() / 2;
            const auto checked_patches = std::span(patches).subspan(offset, checked);
            const auto checked_samples = std::span(simulation.Samples).subspan(offset * frames, checked * frames);
            const auto reference = nakatsuka::Render(material, acoustic, checked_patches, checked_samples);
            const auto actual = nakatsuka::RenderGpu(gpu, material, acoustic, checked_patches, checked_samples);
            double error = 0, power = 0;
            for (uint32_t i = 0; i < frames; ++i) {
                error += std::pow(actual[i] - reference[i], 2);
                power += reference[i] * reference[i];
            }
            const double relative = std::sqrt(error / std::max(power, 1e-30));
            if (relative > .0001) throw std::runtime_error("Nakatsuka CPU/GPU radiation mismatch: " + std::to_string(relative));
            if (case_index) metrics << ',';
            metrics << "{\"simulation_seconds\":" << simulation_seconds << ",\"synthesis_seconds\":" << synthesis_seconds << ",\"rms\":" << std::sqrt(square / frames) << ",\"contact_fraction\":" << active / simulation.Samples.size() << ",\"radiation_relative_l2\":" << relative << '}';
            std::cout << "Nakatsuka scene " << case_index << ": simulation " << simulation_seconds << " s, synthesis " << synthesis_seconds << " s\n";
        }
        metrics << "]}\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
