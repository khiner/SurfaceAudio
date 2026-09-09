#include "conan/Conan.h"
#include "core/GpuMix.h"
#include "core/GpuModal.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <vector>

using namespace surface_audio;
namespace {
using Clock = std::chrono::steady_clock;
double Milliseconds(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
double Quantile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    return values[std::min(values.size() - 1, std::size_t(q * values.size()))];
}
void Run(uint32_t voices) {
    constexpr uint32_t frames = 128, blocks = 64;
    std::vector<Mode> modes;
    for (uint32_t mode = 0; mode < 32; ++mode) modes.push_back({float(311 + 131 * mode + 3.7 * mode * mode), float(.025 + .16 / (1 + .12 * mode)), float(.008 / (1 + .3 * mode))});
    auto bank = MakeModalBank(modes, voices, 48000);
    std::vector<conan::Parameters> parameters(voices, conan::MakeParameters({}, 48000));
    std::vector<conan::State> initial(voices);
    for (uint32_t voice = 0; voice < voices; ++voice) initial[voice] = conan::MakeState(42, voice);
    auto states = initial;
    std::vector<float> force(voices * frames), output(force.size()), mix(frames);
    const auto reduce = [&](std::span<const float> samples) {
        std::fill(mix.begin(), mix.end(), 0.f);
        for (uint32_t voice = 0; voice < voices; ++voice)
            for (uint32_t frame = 0; frame < frames; ++frame) mix[frame] += samples[voice * frames + frame] / voices;
    };
    auto gpu = CreateGpu();
    const auto kernel = CreateKernel(gpu, "ConanSynthesize");
    const auto modal = CreateGpuModal(gpu, bank, frames);
    const auto gpu_mix = CreateGpuMix(gpu, {voices, frames, 1, 0, 1.f / voices});
    const auto params = Upload<conan::Parameters>(gpu, parameters), state = Upload<conan::State>(gpu, initial), count = Upload(gpu, frames), excitation = CreateBuffer(gpu, force.size() * sizeof(float));
    const std::array bindings{GpuBinding{params, 0}, GpuBinding{state, 1}, GpuBinding{excitation, 2}, GpuBinding{count, 3}};
    std::vector<double> cpu_times, gpu_times;
    double checksum = 0;
    const auto cpu_block = [&] {
        const auto start = Clock::now();
        for (uint32_t voice = 0; voice < voices; ++voice) conan::Render(parameters[voice], states[voice], std::span(force).subspan(voice * frames, frames));
        RenderModal(bank, frames, force, output);
        reduce(output);
        const auto elapsed = Milliseconds(start);
        checksum += mix[frames - 1];
        return elapsed;
    };
    const auto gpu_block = [&] {
        const auto start = Clock::now();
        BeginGpu(gpu);
        DispatchGpu(gpu, kernel, bindings, {voices});
        EncodeModal(gpu, modal, excitation);
        EncodeMix(gpu, gpu_mix, modal.Output);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        std::copy_n(BufferSpan<float>(gpu_mix.Output).begin(), frames, mix.begin());
        const auto elapsed = Milliseconds(start);
        checksum += mix[frames - 1];
        return elapsed;
    };
    for (uint32_t warmup = 0; warmup < 4; ++warmup) {
        cpu_block();
        gpu_block();
    }
    // Alternate first runner to reduce thermal/order bias.
    for (uint32_t block = 0; block < blocks; ++block) {
        if (block % 2) {
            gpu_times.push_back(gpu_block());
            cpu_times.push_back(cpu_block());
        } else {
            cpu_times.push_back(cpu_block());
            gpu_times.push_back(gpu_block());
        }
    }
    std::cout << voices << ',' << frames << ',' << modes.size() << ',' << Quantile(cpu_times, .5) << ',' << Quantile(cpu_times, .99) << ',' << Quantile(gpu_times, .5) << ',' << Quantile(gpu_times, .99) << ',' << checksum << '\n';
}
} // namespace
int main() {
    try {
        std::cout << "Conan + modal + mono mix, 48 kHz; allocation/compilation excluded, GPU submit/wait/readback included\nvoices,frames,modes,cpu_p50_ms,cpu_p99_ms,gpu_p50_ms,gpu_p99_ms,checksum\n";
        for (uint32_t voices : {1u, 64u, 256u, 1024u}) Run(voices);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
