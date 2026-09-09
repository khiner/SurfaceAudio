#include "agarwal/SpatialConvolution.h"
#include "core/AudioFile.h"
#include "core/BinaryFile.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace surface_audio;
namespace {

}

int main(int argc, char **argv) {
    try {
        if (argc != 3 && !(argc == 4 && std::string(argv[3]) == "--basis")) throw std::invalid_argument("Usage: agarwalSpatialReproduce CASE_DIRECTORY OUTPUT_PREFIX [--basis]");
        const std::filesystem::path directory = argv[1];
        std::ifstream parameters(directory / "parameters.txt");
        uint32_t sample_rate{}, frames{}, levels{}, taps{};
        double spacing{}, half_width{}, mass{}, beta1{}, beta2{}, radius{}, eccentricity{}, stiffness{}, dissipation{}, gain{};
        parameters >> sample_rate >> frames >> levels >> taps >> spacing >> half_width >> mass >> beta1 >> beta2 >> radius >> eccentricity >> stiffness >> dissipation >> gain;
        if (!parameters || !frames) throw std::runtime_error("Invalid reproduction parameters");
        auto force = ReadBinary<float>(directory / "scrape.f32");
        const auto elastic = ReadBinary<float>(directory / "elastic.f32"), damping = ReadBinary<float>(directory / "damping.f32"), morph = ReadBinary<float>(directory / "morph.f32");
        if (force.size() != frames || elastic.size() != frames || damping.size() != frames || morph.size() != frames) throw std::runtime_error("Prepared force extent mismatch");
        for (uint32_t frame = 0; frame < frames; ++frame) force[frame] += float(stiffness * elastic[frame] + dissipation * damping[frame]);
        const auto frequencies0 = ReadBinary<double>(directory / "surface0_frequencies.f64"), frequencies1 = ReadBinary<double>(directory / "surface1_frequencies.f64"), object_frequencies = ReadBinary<double>(directory / "object_frequencies.f64");
        const auto amplitudes0 = ReadBinary<double>(directory / "surface0_amplitudes.f64"), amplitudes1 = ReadBinary<double>(directory / "surface1_amplitudes.f64"), object_amplitudes = ReadBinary<double>(directory / "object_amplitudes.f64");
        const agarwal::ImpulseResponses responses{double(sample_rate), 1, taps, {frequencies0, amplitudes0}, {frequencies1, amplitudes1}, {object_frequencies, object_amplitudes}};
        auto gpu = CreateGpu();
        if (argc == 4) {
            agarwal::Validate(responses);
            std::ofstream file(std::string(argv[2]) + "-basis.f32", std::ios::binary);
            if (!file) throw std::runtime_error("Cannot write modal basis");
            const std::span<const double> f0 = frequencies0, f1 = frequencies1, fo = object_frequencies, a0 = amplitudes0, a1 = amplitudes1, ao = object_amplitudes;
            for (size_t mode = 0; mode < f0.size() + fo.size(); ++mode) {
                auto one = responses;
                if (mode < f0.size()) {
                    one.Surface0 = {f0.subspan(mode, 1), a0.subspan(mode * taps, taps)};
                    one.Surface1 = {f1.subspan(mode, 1), a1.subspan(mode * taps, taps)};
                    one.Object = {};
                } else {
                    one.Surface0 = one.Surface1 = {};
                    one.Object = {fo.subspan(mode - f0.size(), 1), ao.subspan((mode - f0.size()) * taps, taps)};
                }
                const auto output = agarwal::ConvolveSpatialGpu(gpu, force, morph, one, 256);
                file.write(reinterpret_cast<const char *>(output.data()), std::streamsize(output.size() * sizeof(float)));
            }
            if (!file) throw std::runtime_error("Cannot write modal basis");
            std::cout << "Rendered independent spatial mode basis: " << f0.size() + fo.size() << " modes\n";
            return 0;
        }
        const auto output = agarwal::ConvolveSpatialGpu(gpu, force, morph, responses, 128, float(gain));
        WriteWave(std::string(argv[2]) + ".wav", sample_rate, 1, output);
        std::cout << "Rendered full spatial IR: " << output.size() << " samples, " << frequencies0.size() << " surface modes, " << object_frequencies.size() << " object modes\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
