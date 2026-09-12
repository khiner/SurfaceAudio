#include "core/AudioFile.h"
#include "hatt_input.h"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace surface_audio;
int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::invalid_argument("Usage: hattReproduce input.txt output");
        const auto input = hatt::ReadInput(argv[1]);
        const std::filesystem::path output(argv[2]);
        std::filesystem::create_directories(output);
        const auto frames = input.Frames;
        const auto &textures = input.Textures;
        for (const auto &texture : textures) hatt::Validate(texture);
        auto gpu = CreateGpu();
        const auto begin = std::chrono::steady_clock::now();
        const auto rendered = hatt::RenderGpu(gpu, textures, input.Controls, input.Noise, true);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        const auto reference_start = std::chrono::steady_clock::now();
        std::ofstream coefficients(output / "filters.f64", std::ios::binary), metrics(output / "render.json");
        metrics << std::setprecision(12) << "{\"gpu_seconds\":" << seconds << ",\"voices\":" << textures.size() << ",\"cases\":[";
        for (size_t voice = 0; voice < textures.size(); ++voice) {
            hatt::State state;
            double error = 0, power = 0;
            for (size_t i = 0; i < frames; ++i) {
                const auto index = voice * frames + i;
                const auto f = hatt::Interpolate(textures[voice], input.Controls[index], true);
                coefficients.write(reinterpret_cast<const char *>(f.Ar.data()), sizeof(f.Ar));
                coefficients.write(reinterpret_cast<const char *>(f.Ma.data()), sizeof(f.Ma));
                coefficients.write(reinterpret_cast<const char *>(&f.Variance), sizeof(double));
                const double reference = hatt::Tick(f, state, input.Noise[index]);
                const double delta = reference - rendered[index];
                if (!std::isfinite(reference) || !std::isfinite(rendered[index])) throw std::runtime_error("Nonfinite HaTT render");
                error += delta * delta;
                power += reference * reference;
            }
            const double relative = std::sqrt(error / std::max(power, 1e-30));
            if (relative > 1e-4) throw std::runtime_error("HaTT CPU/GPU mismatch: " + std::to_string(relative));
            const auto rate = uint32_t(textures[voice].SampleRate);
            WriteWave(output / (std::to_string(voice) + ".wav"), rate, 1, std::span(rendered).subspan(voice * frames, frames));
            if (voice) metrics << ',';
            metrics << "{\"relative_l2\":" << relative << '}';
        }
        coefficients.flush();
        metrics << "],\"reference_seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now() - reference_start).count() << "}\n";
        std::cout << "HaTT: " << textures.size() << " voices, " << frames << " frames, GPU " << seconds << " s\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
