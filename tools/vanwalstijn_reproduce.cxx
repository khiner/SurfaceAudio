#include "barrier/Barrier.h"
#include "core/BinaryFile.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace surface_audio;
using namespace surface_audio::barrier;
int main(int argc, char **argv) {
    try {
        if (argc < 4 || argc > 5 || (argc == 5 && std::string_view(argv[4]) != "--gpu")) throw std::invalid_argument("Usage: vanwalstijnReproduce CONTROLS.txt FORCES.f64 OUTPUT_DIRECTORY [--gpu]");
        Controls c;
        std::ifstream controls(argv[1]);
        if (!(controls >> c.SampleRate >> c.Mass >> c.Stiffness >> c.Damping >> c.Barrier >> c.Kappa >> c.Exponent >> c.Restitution >> c.MaxVelocity >> c.Regularization >> c.Friction >> c.CapStiffness >> c.ModalCorrection >> c.Implicit)) throw std::invalid_argument("Invalid mass-barrier controls");
        controls >> std::ws;
        if (!controls.eof()) throw std::invalid_argument("Extra mass-barrier control field");
        const auto p = MakeParameters<double>(c);
        const auto input = ReadBinary<Force<double>>(argv[2]);
        if (input.empty()) throw std::invalid_argument("Empty mass-barrier input");
        const std::filesystem::path output = argv[3];
        std::filesystem::create_directories(output);
        std::vector<Sample<double>> samples(input.size());
        auto state = MakeState(p);
        Render<double>(p, state, input, samples);
        WriteBinary<Sample<double>>(output / "cpu.f64", samples);
        double residual = 0, work = 0, loss = 0, minimum_psi = 0, minimum_loss = 0;
        for (const auto &s : samples) {
            residual += s.BalanceResidual;
            work += p.Dt * s.ExternalPower;
            loss += p.Dt * (s.LinearLoss + s.ContactLoss);
            minimum_psi = std::min(minimum_psi, s.RawPsi);
            minimum_loss = std::min(minimum_loss, s.ContactLoss);
        }
        if (argc == 5) {
            constexpr unsigned block = 256;
            const std::array config{MakeParameters<float>(c)};
            auto gpu = CreateGpu();
            auto source = CreateGpuSource(gpu, config, block);
            std::vector<Sample<float>> result(input.size());
            for (size_t offset = 0; offset < input.size(); offset += block) {
                source.Frames = unsigned(std::min<size_t>(block, input.size() - offset));
                auto drive = BufferSpan<Force<float>>(source.Input);
                for (unsigned n = 0; n < source.Frames; ++n) drive[n] = {float(input[offset + n].X), float(input[offset + n].Y)};
                BeginGpu(gpu);
                EncodeSource(gpu, source);
                SubmitGpu(gpu);
                WaitGpu(gpu);
                std::copy_n(BufferSpan<Sample<float>>(source.Output).begin(), source.Frames, result.begin() + offset);
            }
            WriteBinary<Sample<float>>(output / "gpu.f32", result);
        }
        std::ofstream metadata(output / "native.json");
        metadata << std::setprecision(17) << "{\"frames\":" << input.size() << ",\"effective_stiffness\":" << p.Kappa << ",\"linear_stiffness\":" << p.Stiffness << ",\"linear_damping\":" << p.Damping << ",\"work\":" << work << ",\"dissipation\":" << loss << ",\"energy_balance_residual\":" << residual << ",\"minimum_raw_psi\":" << minimum_psi << ",\"minimum_contact_loss\":" << minimum_loss << "}\n";
        if (!metadata) throw std::runtime_error("Cannot write mass-barrier metadata");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
