#include "core/AudioFile.h"
#include "core/BinaryFile.h"
#include "core/GpuModal.h"
#include "ren/Ren.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace surface_audio;
namespace {
struct Binding {
    uint32_t Nodes[3];
    float Weights[3];
};
struct Error {
    double Energy{}, Squared{};
    void Add(float expected, float actual) {
        if (!std::isfinite(actual)) throw std::runtime_error("Nonfinite Ren output");
        Energy += double(expected) * expected;
        Squared += std::pow(double(expected) - actual, 2);
    }
    double Relative() const { return std::sqrt(Squared / std::max(Energy, 1e-30)); }
};
void Project(std::span<const Binding> bindings, std::span<const ren::Impulses> impulses, std::span<float> drive) {
    std::ranges::fill(drive, 0);
    for (size_t n = 0; n < bindings.size(); ++n)
        for (unsigned i = 0; i < 3; ++i) drive[size_t(bindings[n].Nodes[i]) * bindings.size() + n] += impulses[n].Total() * bindings[n].Weights[i];
}
}
int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::invalid_argument("Usage: renReproduce INPUT_DIRECTORY OUTPUT_DIRECTORY");
        const std::filesystem::path input = argv[1], output = argv[2];
        const auto config = ReadBinary<ren::Parameters>(input / "parameters.bin");
        if (config.size() != 1) throw std::invalid_argument("Expected one Ren parameter record");
        const auto &p = config.front();
        const auto normals = ReadBinary<ren::Normal>(input / "normals.f32");
        const auto roughness = ReadBinary<float>(input / "roughness.f32");
        const auto contacts = ReadBinary<ren::Contact>(input / "contacts.bin");
        const auto bindings = ReadBinary<Binding>(input / "bindings.bin");
        const auto modes = ReadBinary<Mode>(input / "modes.f32");
        const auto shapes = ReadBinary<float>(input / "shapes.f32");
        if (contacts.empty() || contacts.size() != bindings.size() || modes.empty() || shapes.empty() || shapes.size() % modes.size() || contacts.size() > 120 * p.SampleRate) throw std::invalid_argument("Invalid Ren render dimensions");
        const size_t vertices = shapes.size() / modes.size();
        if (vertices > 4096) throw std::invalid_argument("Ren renderer supports at most 4096 vertices");
        for (float x : shapes)
            if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite Ren mode shape");
        for (const auto &b : bindings) {
            float sum = 0;
            for (unsigned i = 0; i < 3; ++i) {
                if (b.Nodes[i] >= vertices || !std::isfinite(b.Weights[i]) || b.Weights[i] < 0 || b.Weights[i] > 1) throw std::invalid_argument("Invalid Ren contact binding");
                sum += b.Weights[i];
            }
            if (std::abs(sum - 1) > 1e-5f) throw std::invalid_argument("Ren contact weights must sum to one");
        }
        std::vector<ren::Impulses> impulses(contacts.size());
        ren::State state;
        ren::Render(p, state, normals, roughness, contacts, impulses);
        auto bank = MakeModalBank(modes, uint32_t(vertices), p.SampleRate);
        for (size_t i = 0; i < shapes.size(); ++i) bank.Coefficients[2 * shapes.size() + i] *= shapes[i];
        constexpr unsigned block = 256;
        auto gpu = CreateGpu();
        auto source = ren::CreateGpuSource(gpu, p, normals, roughness, 1, block);
        const auto modal = CreateGpuModal(gpu, bank, block);
        auto drive = CreateBuffer(gpu, vertices * block * sizeof(float));
        std::vector<float> cpu(block * vertices), audio;
        Error source_error, audio_error;
        for (size_t offset = 0; offset < contacts.size(); offset += block) {
            const auto frames = source.Frames = unsigned(std::min<size_t>(block, contacts.size() - offset));
            const auto weights = std::span(bindings).subspan(offset, frames);
            const auto reference_impulses = std::span(impulses).subspan(offset, frames);
            std::copy_n(contacts.begin() + offset, frames, BufferSpan<ren::Contact>(source.Contacts).begin());
            BeginGpu(gpu);
            ren::EncodeSource(gpu, source);
            SubmitGpu(gpu);
            WaitGpu(gpu);
            const auto result = BufferSpan<ren::Impulses>(source.Output);
            auto input_block = BufferSpan<float>(drive).first(vertices * frames);
            for (unsigned n = 0; n < frames; ++n) source_error.Add(reference_impulses[n].Total(), result[n].Total());
            Project(weights, result, input_block);
            BufferSpan<ModalBlock>(modal.Parameters)[0].Frames = frames;
            BeginGpu(gpu);
            EncodeModal(gpu, modal, drive);
            SubmitGpu(gpu);
            WaitGpu(gpu);
            Project(weights, reference_impulses, input_block);
            RenderModal(bank, frames, input_block, std::span(cpu).first(vertices * frames));
            const auto sound = BufferSpan<float>(modal.Output);
            for (unsigned n = 0; n < frames; ++n) {
                double total = 0, reference = 0;
                for (size_t vertex = 0; vertex < vertices; ++vertex) {
                    total += sound[vertex * frames + n];
                    reference += cpu[vertex * frames + n];
                }
                audio.push_back(float(total));
                audio_error.Add(float(reference), float(total));
            }
        }
        if (source_error.Relative() > .002 || audio_error.Relative() > .003) throw std::runtime_error("Ren CPU/Metal comparison failed");
        std::filesystem::create_directories(output);
        WriteWave(output / "audio.wav", uint32_t(p.SampleRate), 1, audio);
        WriteBinary<ren::Impulses>(output / "impulses.f32", impulses);
        std::ofstream metadata(output / "native.json");
        metadata << std::setprecision(12) << "{\"frames\":" << audio.size() << ",\"vertices\":" << vertices << ",\"modes\":" << modes.size() << ",\"source_cpu_gpu_relative_l2\":" << source_error.Relative() << ",\"audio_cpu_gpu_relative_l2\":" << audio_error.Relative() << ",\"device\":" << std::quoted(DeviceName(gpu)) << "}\n";
        if (!metadata) throw std::runtime_error("Cannot write Ren metadata");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
