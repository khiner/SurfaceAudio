#include "collision/Collision.h"
#include "core/BinaryFile.h"
#include <algorithm>
#include <iostream>
using namespace surface_audio;
using namespace surface_audio::collision;
int main(int argc, char **argv) {
    try {
        if (argc < 4 || argc > 5 || (argc == 5 && std::string_view(argv[4]) != "--gpu")) throw std::invalid_argument("Usage: ducceschiReproduce MODEL.txt FRAMES OUTPUT_DIRECTORY [--gpu]");
        const auto model = ReadModel(argv[1]);
        const unsigned frames = unsigned(std::stoul(argv[2]));
        if (!frames || frames > 10000000) throw std::invalid_argument("Invalid collision duration");
        const std::filesystem::path directory = argv[3];
        std::filesystem::create_directories(directory);
        auto state = model.State;
        std::vector<Sample<double>> output(frames);
        Render<double>(model, state, output);
        WriteBinary<Sample<double>>(directory / "cpu.f64", output);
        WriteBinary<double>(directory / "state.f64", state);
        if (argc == 5) {
            auto gpu = CreateGpu();
            auto source = CreateGpuSource(gpu, FloatModel(model), 1, 128);
            std::vector<Sample<float>> metal(frames);
            for (unsigned offset = 0; offset < frames; offset += 128) {
                source.Frames = std::min(128u, frames - offset);
                BeginGpu(gpu); EncodeSource(gpu, source); SubmitGpu(gpu); WaitGpu(gpu);
                std::copy_n(BufferSpan<Sample<float>>(source.Output).begin(), source.Frames, metal.begin() + offset);
            }
            WriteBinary<Sample<float>>(directory / "gpu.f32", metal);
        }
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
