#include "core/AudioFile.h"
#include "core/Convolution.h"
#include "core/Gpu.h"
#include "core/GpuConvolution.h"
#include "core/GpuFiniteModes.h"
#include "core/GpuMix.h"
#include "core/GpuModal.h"
#include "core/Modal.h"
#include "core/Random.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <vector>

using namespace surface_audio;
namespace {
void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void RandomTest(Gpu &gpu) {
    constexpr uint32_t streams = 64, count = 257;
    auto kernel = CreateKernel(gpu, "RandomSequence");
    auto output = CreateBuffer(gpu, streams * count * sizeof(uint32_t)), dimensions = Upload(gpu, count);
    const std::array bindings{GpuBinding{output, 0}, GpuBinding{dimensions, 1}};
    BeginGpu(gpu);
    DispatchGpu(gpu, kernel, bindings, {streams});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto values = BufferSpan<uint32_t>(output);
    for (uint32_t stream = 0; stream < streams; ++stream) {
        auto state = MakeRandom(42, stream);
        for (uint32_t sample = 0; sample < count; ++sample) Require(values[stream * count + sample] == NextRandom(state), "PCG CPU/GPU bit mismatch");
    }
    auto state = MakeRandom(123);
    double sum = 0, squares = 0;
    for (unsigned i = 0; i < 100000; ++i) {
        const auto u = Uniform(state);
        Require(u > 0 && u < 1, "Uniform must be inside open interval");
        const auto z = Normal(state);
        sum += z;
        squares += z * z;
    }
    Require(std::abs(sum / 100000) < .02 && std::abs(squares / 100000 - 1) < .03, "Normal distribution moments");
}

void BatchConstantsTest(Gpu &gpu) {
    const auto kernel = CreateKernel(gpu, "RandomSequence");
    const std::array counts{17u, 257u};
    const std::array outputs{CreateBuffer(gpu, counts[0] * sizeof(uint32_t)), CreateBuffer(gpu, counts[1] * sizeof(uint32_t))};
    bool rejected = false;
    try {
        BatchUpload(gpu, counts[0]);
    } catch (const std::invalid_argument &) { rejected = true; }
    Require(rejected, "Dispatch constants require an active batch");
    BeginGpu(gpu);
    for (size_t i = 0; i < counts.size(); ++i) {
        const std::array bindings{GpuBinding{outputs[i], 0}, GpuBinding{BatchUpload(gpu, counts[i]), 1}};
        DispatchGpu(gpu, kernel, bindings, {1});
    }
    for (unsigned i = 2; i < 32; ++i) BatchUpload(gpu, i);
    rejected = false;
    try {
        BatchUpload(gpu, counts[0]);
    } catch (const std::invalid_argument &) { rejected = true; }
    Require(rejected, "Dispatch constant arena rejects overflow");
    SubmitGpu(gpu);
    WaitGpu(gpu);
    for (size_t i = 0; i < counts.size(); ++i) {
        auto state = MakeRandom(42, 0);
        for (uint32_t value : BufferSpan<uint32_t>(outputs[i])) Require(value == NextRandom(state), "Queued dispatch constants remain distinct until completion");
    }
    BeginGpu(gpu);
    Require(BatchUpload(gpu, counts[0]).Size == sizeof(uint32_t), "Dispatch constant storage is reusable after completion");
    SubmitGpu(gpu);
    WaitGpu(gpu);
}

void FirTest(Gpu &gpu) {
    const FirBlock p{131, 37};
    auto random = MakeRandom(3);
    std::vector<float> coefficients(p.TapCount * p.FrameCount), excitation(p.TapCount - 1 + p.FrameCount), cpu(p.FrameCount);
    for (auto &value : coefficients) value = Uniform(random) - .5f;
    for (auto &value : excitation) value = Uniform(random) - .5f;
    Convolve(p, coefficients, excitation, cpu);
    auto kernel = CreateKernel(gpu, "FirConvolve");
    const auto params = Upload(gpu, p), coeffs = Upload<float>(gpu, coefficients), input = Upload<float>(gpu, excitation), output = CreateBuffer(gpu, cpu.size() * sizeof(float));
    const std::array bindings{GpuBinding{params, 0}, GpuBinding{coeffs, 1}, GpuBinding{input, 2}, GpuBinding{output, 3}};
    BeginGpu(gpu);
    DispatchGroupsGpu(gpu, kernel, bindings, {p.FrameCount}, {128});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    for (uint32_t frame = 0; frame < p.FrameCount; ++frame) {
        double expected = 0;
        for (uint32_t lag = 0; lag < p.TapCount; ++lag) expected += double(coefficients[frame * p.TapCount + lag]) * excitation[p.TapCount - 1 + frame - lag];
        Require(std::abs(cpu[frame] - expected) < 2e-6, "FIR NEON/scalar convolution mismatch");
        Require(std::abs(BufferSpan<float>(output)[frame] - expected) < 2e-6, "FIR Metal/scalar convolution mismatch");
    }
}

void ModalTest(Gpu &gpu) {
    const std::array modes{Mode{370, .11f, .13f}, Mode{701, .07f, .17f}, Mode{1031, .13f, .06f}, Mode{1751, .1f, .1f}, Mode{2711, .04f, .03f}};
    constexpr uint32_t voices = 7, frames = 512;
    auto bank = MakeModalBank(modes, voices, 48000);
    const auto initial = bank;
    std::vector<float> input(voices * frames), cpu(input.size());
    for (uint32_t voice = 0; voice < voices; ++voice) input[voice * frames] = float(voice + 1) / voices;
    RenderModal(bank, frames, input, cpu);
    double max_error = 0;
    for (uint32_t voice = 0; voice < voices; ++voice)
        for (uint32_t frame = 0; frame < frames; ++frame) {
            double expected = 0;
            for (const auto &mode : modes) expected += mode.Amplitude * std::exp(-double(frame) / (48000 * mode.Decay)) * std::sin(2 * std::numbers::pi * mode.Frequency * frame / 48000);
            expected *= float(voice + 1) / voices;
            max_error = std::max(max_error, std::abs(cpu[voice * frames + frame] - expected));
        }
    Require(max_error < 4e-5, "Modal analytic impulse response mismatch");
    auto split = initial;
    std::vector<float> chunk_input(voices * 128), chunk_output(chunk_input.size());
    for (uint32_t offset = 0; offset < frames; offset += 128) {
        for (uint32_t voice = 0; voice < voices; ++voice) std::copy_n(input.data() + voice * frames + offset, 128, chunk_input.data() + voice * 128);
        RenderModal(split, 128, chunk_input, chunk_output);
        for (uint32_t voice = 0; voice < voices; ++voice)
            for (uint32_t frame = 0; frame < 128; ++frame) Require(cpu[voice * frames + offset + frame] == chunk_output[voice * 128 + frame], "Modal CPU streaming mismatch");
    }
    Require(split.State == bank.State, "Modal CPU terminal state mismatch");
    const auto modal = CreateGpuModal(gpu, initial, 128);
    const auto excitation = CreateBuffer(gpu, voices * modal.Block.Frames * sizeof(float));
    double gpu_error = 0;
    for (uint32_t offset = 0; offset < frames; offset += modal.Block.Frames) {
        for (uint32_t voice = 0; voice < voices; ++voice) std::copy_n(input.data() + voice * frames + offset, modal.Block.Frames, BufferSpan<float>(excitation).data() + voice * modal.Block.Frames);
        BeginGpu(gpu);
        EncodeModal(gpu, modal, excitation);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        for (uint32_t voice = 0; voice < voices; ++voice)
            for (uint32_t frame = 0; frame < modal.Block.Frames; ++frame) gpu_error = std::max(gpu_error, double(std::abs(BufferSpan<float>(modal.Output)[voice * modal.Block.Frames + frame] - cpu[voice * frames + offset + frame])));
    }
    Require(gpu_error < 2e-6, "Modal CPU/GPU full trace mismatch");
    const auto gpu_state = BufferSpan<float>(modal.State);
    for (std::size_t i = 0; i < bank.State.size(); ++i) Require(std::abs(gpu_state[i] - bank.State[i]) < 2e-6, "Modal CPU/GPU terminal state mismatch");
    std::cout << "Modal analytic max error " << max_error << ", GPU max error " << gpu_error << '\n';
}

void ModalMultiSimdTest(Gpu &gpu) {
    std::vector<Mode> modes;
    for (uint32_t index = 0; index < 65; ++index) modes.push_back({131.f + 77 * index, .02f + .003f * index, .01f / (1 + .1f * index)});
    constexpr uint32_t voices = 3, block = 127, frames = 16 * block;
    auto cpu_bank = MakeModalBank(modes, voices, 48000);
    const auto whole_modal = CreateGpuModal(gpu, cpu_bank, frames), blocked_modal = CreateGpuModal(gpu, cpu_bank, block);
    Require(whole_modal.Threads == 128, "65 modes require four SIMDgroups including padding");
    auto random = MakeRandom(716);
    std::vector<float> input(voices * frames), expected(input.size());
    for (float &sample : input) sample = .1f * Normal(random);
    RenderModal(cpu_bank, frames, input, expected);
    const auto whole_input = Upload<float>(gpu, input), block_input = CreateBuffer(gpu, voices * block * sizeof(float));
    BeginGpu(gpu);
    EncodeModal(gpu, whole_modal, whole_input);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto whole = BufferSpan<float>(whole_modal.Output);
    double maximum_error = 0;
    for (size_t index = 0; index < whole.size(); ++index) maximum_error = std::max(maximum_error, double(std::abs(whole[index] - expected[index])));
    Require(maximum_error < 2e-6, "65-mode fused GPU full trace matches CPU");
    for (uint32_t offset = 0; offset < frames; offset += block) {
        for (uint32_t voice = 0; voice < voices; ++voice) std::copy_n(input.data() + voice * frames + offset, block, BufferSpan<float>(block_input).data() + voice * block);
        BeginGpu(gpu);
        EncodeModal(gpu, blocked_modal, block_input);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        const auto actual = BufferSpan<float>(blocked_modal.Output);
        for (uint32_t voice = 0; voice < voices; ++voice)
            for (uint32_t frame = 0; frame < block; ++frame) Require(actual[voice * block + frame] == whole[voice * frames + offset + frame], "65-mode GPU exact block continuity");
    }
    const auto whole_state = BufferSpan<float>(whole_modal.State), blocked_state = BufferSpan<float>(blocked_modal.State);
    for (size_t index = 0; index < whole_state.size(); ++index) Require(whole_state[index] == blocked_state[index] && std::abs(whole_state[index] - cpu_bank.State[index]) < 2e-6, "65-mode GPU terminal state");
    std::cout << "Modal 65-mode GPU max error " << maximum_error << '\n';
}

void ModalForcedTest(Gpu &gpu) {
    constexpr uint32_t frames = 1024;
    const std::array modes{Mode{80, .2f, .1f}, Mode{787, .05f, -.02f}, Mode{3301, .03f, .008f}};
    auto bank = MakeModalBank(modes, 1, 48000);
    const auto modal = CreateGpuModal(gpu, bank, frames);
    auto random = MakeRandom(17);
    std::vector<float> force(frames), output(frames);
    for (auto &sample : force) sample = .02f + .03f * (Uniform(random) - .5f);
    std::vector<double> impulse(frames);
    for (uint32_t lag = 0; lag < frames; ++lag)
        for (const auto &mode : modes) impulse[lag] += mode.Amplitude * std::exp(-double(lag) / (48000 * double(mode.Decay))) * std::sin(2 * std::numbers::pi * mode.Frequency * lag / 48000);
    RenderModal(bank, frames, force, output);
    const auto excitation = Upload<float>(gpu, force);
    BeginGpu(gpu);
    EncodeModal(gpu, modal, excitation);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto gpu_output = BufferSpan<float>(modal.Output);
    double maximum_error = 0;
    for (uint32_t frame = 0; frame < frames; ++frame) {
        double expected = 0;
        for (uint32_t lag = 0; lag <= frame; ++lag) expected += impulse[lag] * force[frame - lag];
        maximum_error = std::max(maximum_error, std::abs(output[frame] - expected));
        Require(std::abs(gpu_output[frame] - expected) < 1e-5, "GPU modal forced response equals independent continuous-pole convolution");
    }
    Require(maximum_error < 1e-5, "CPU modal forced response equals independent continuous-pole convolution");
    std::cout << "Modal forced analytic max error " << maximum_error << '\n';
}

void ModalStabilityTest(Gpu &gpu) {
    const std::array low_mode{Mode{80, .2f, .1f}};
    constexpr uint32_t frames = 96000;
    auto bank = MakeModalBank(low_mode, 1, 48000);
    const auto modal = CreateGpuModal(gpu, bank, frames);
    std::vector<float> input(frames), output(frames);
    input[0] = 1;
    RenderModal(bank, frames, input, output);
    const auto excitation = Upload<float>(gpu, input);
    BeginGpu(gpu);
    EncodeModal(gpu, modal, excitation);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto gpu_output = BufferSpan<float>(modal.Output);
    double maximum_error = 0;
    for (uint32_t frame = 0; frame < frames; ++frame) {
        Require(std::abs(gpu_output[frame] - output[frame]) < 1e-6, "Low-frequency GPU modal full tail matches CPU");
        const double expected = .1 * std::exp(-double(frame) / (48000 * .2)) * std::sin(2 * std::numbers::pi * 80 * frame / 48000);
        maximum_error = std::max(maximum_error, std::abs(output[frame] - expected));
    }
    std::cout << "Modal 80 Hz full-tail analytic max error " << maximum_error << '\n';
    Require(maximum_error < .00001, "Low-frequency modal impulse matches the requested analytic full tail");
    const double radius = std::hypot(double(bank.Coefficients[0]), double(bank.Coefficients[1]));
    const double angle = std::atan2(double(bank.Coefficients[1]), double(bank.Coefficients[0]));
    const double amplitude = bank.Coefficients[2];
    double quantized_error = 0;
    for (uint32_t frame = 0; frame < frames; ++frame) {
        const double expected = amplitude * std::pow(radius, frame) * std::sin(angle * frame);
        quantized_error = std::max(quantized_error, std::abs(output[frame] - expected));
    }
    Require(quantized_error < .00003, "Modal recurrence matches analytic poles of stored float coefficients");
    Require(std::abs(output.back()) < .00001, "Low-frequency modal tail decays");
    bool rejected = false;
    try {
        MakeModalBank(std::array{Mode{.01f, 1000.f, .1f}}, 1, 96000);
    } catch (const std::invalid_argument &) { rejected = true; }
    Require(rejected, "Poles rounded outside the unit circle must be rejected");
}

void MixTest(Gpu &gpu) {
    const std::array configurations{MixBlock{7, 37, 1, 0, .125f}, MixBlock{65, 31, 7, 3, -.03125f}};
    for (const auto &configuration : configurations) {
        std::vector<float> input(size_t(configuration.Voices) * configuration.Frames * configuration.Stride, 17.f);
        std::vector<float> expected(configuration.Frames);
        for (uint32_t voice = 0; voice < configuration.Voices; ++voice)
            for (uint32_t frame = 0; frame < configuration.Frames; ++frame) {
                const float value = voice * .0625f + frame * .0078125f;
                input[(voice * configuration.Frames + frame) * configuration.Stride + configuration.Field] = value;
                expected[frame] += value * configuration.Gain;
            }
        const auto mix = CreateGpuMix(gpu, configuration);
        const auto input_buffer = Upload<float>(gpu, input);
        BeginGpu(gpu);
        EncodeMix(gpu, mix, input_buffer);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        const auto actual = BufferSpan<float>(mix.Output);
        for (uint32_t frame = 0; frame < configuration.Frames; ++frame) Require(actual[frame] == expected[frame], "GPU voice-major/strided-field mixing and gain");
    }
}

void WaveTest() {
    const auto path = std::filesystem::temp_directory_path() / "SurfaceAudioCoreTest.wav";
    const std::array samples{0.f, -2.f, 1.5f, .25f};
    WriteWave(path, 48000, 2, samples);
    const auto decoded = ReadWave(path);
    Require(decoded.SampleRate == 48000 && decoded.Channels == 2 && std::equal(decoded.Samples.begin(), decoded.Samples.end(), samples.begin(), samples.end()), "WAV float decode preserves sample rate, channels and scale");
    std::ifstream file(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
    std::filesystem::remove(path);
    Require(bytes.size() == 56 + sizeof(samples), "WAV extent");
    Require(std::memcmp(bytes.data() + 56, samples.data(), sizeof(samples)) == 0, "WAV must preserve float samples");
    for (uint16_t bits : {16, 24, 32}) {
        std::ofstream pcm(path, std::ios::binary);
        const auto write = [&](auto value) { pcm.write(reinterpret_cast<const char *>(&value), sizeof(value)); };
        pcm.write("RIFF", 4);
        write(uint32_t(36 + 4 * bits / 8));
        pcm.write("WAVEfmt ", 8);
        write(uint32_t(16));
        write(uint16_t(1));
        write(uint16_t(1));
        write(uint32_t(44100));
        write(uint32_t(44100 * bits / 8));
        write(uint16_t(bits / 8));
        write(bits);
        pcm.write("data", 4);
        write(uint32_t(4 * bits / 8));
        const std::array<uint32_t, 4> packed{0, 1u << (bits - 1), 1u << (bits - 2), (1u << (bits - 1)) - 1};
        for (uint32_t value : packed) pcm.write(reinterpret_cast<const char *>(&value), bits / 8);
        pcm.close();
        const auto data = ReadWave(path);
        Require(data.SampleRate == 44100 && data.Channels == 1 && data.Samples[0] == 0 && data.Samples[1] == -1 && data.Samples[2] == .5f && data.Samples[3] > .9999f, "Signed PCM endpoint decoding");
        std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
        bool rejected = false;
        try {
            ReadWave(path);
        } catch (const std::runtime_error &) { rejected = true; }
        Require(rejected, "Truncated PCM input rejected");
        std::filesystem::remove(path);
    }
}

void FixedFirTest(Gpu &gpu) {
    for (const auto [frames, taps] : {std::pair{10003u, 137u}, std::pair{37u, 8195u}, std::pair{1u, 1u}}) {
        auto random = MakeRandom(91);
        std::vector<float> excitation(frames), response(taps);
        for (float &value : excitation) value = Uniform(random) - .5f;
        for (float &value : response) value = Uniform(random) - .5f;
        const auto output = ConvolveFixedGpu(gpu, excitation, response);
        Require(output.size() == frames + taps - 1, "Fixed FIR includes complete tail");
        double error = 0;
        for (size_t frame = 0; frame < output.size(); ++frame) {
            double expected = 0;
            for (size_t lag = frame >= frames ? frame - frames + 1 : 0; lag < std::min<size_t>(taps, frame + 1); ++lag) expected += double(response[lag]) * excitation[frame - lag];
            error = std::max(error, std::abs(output[frame] - expected));
        }
        Require(error < 2e-6, "Fixed GPU FIR versus independent double linear convolution");
    }
}

void FiniteModesTest(Gpu &gpu) {
    constexpr uint32_t frames = 2003, taps = 137, rate = 44100;
    const std::array modes{FiniteMode{80, .2, .05, .01}, FiniteMode{1323, .017, .003, .09}, FiniteMode{7491, .003, .1, .02}};
    std::vector<float> force(frames), morph(frames);
    auto random = MakeRandom(918);
    for (uint32_t frame = 0; frame < frames; ++frame) {
        force[frame] = Uniform(random) - .5f;
        morph[frame] = float(frame % 71) / 70;
    }
    const auto actual = ConvolveFiniteModesGpu(gpu, force, morph, modes, rate, taps);
    Require(actual.size() == frames + taps - 1, "Finite modal convolution tail extent");
    double error = 0;
    for (size_t frame = 0; frame < actual.size(); ++frame) {
        double expected = 0;
        for (const auto &mode : modes) {
            const double amplitude = std::exp(std::lerp(std::log(mode.Amplitude0), std::log(mode.Amplitude1), morph[std::min(frame, size_t(frames - 1))]));
            for (size_t lag = frame >= frames ? frame - frames + 1 : 0; lag < std::min<size_t>(taps, frame + 1); ++lag) expected += force[frame - lag] * amplitude * std::exp(-double(lag) / (rate * mode.Decay)) * std::sin(2 * std::numbers::pi * mode.Frequency * lag / rate);
        }
        error = std::max(error, std::abs(actual[frame] - expected));
    }
    std::cout << "Finite modal output-location FIR max error " << error << '\n';
    Require(error < 2e-5, "Finite modal recurrence versus independent output-location finite convolution");
    constexpr uint32_t long_frames = 88217, long_taps = 22051;
    const std::array long_modes{FiniteMode{80, .7, .07, .015}, FiniteMode{1371, .1, .04, .08}};
    const std::array impulses{std::pair{0u, .5f}, std::pair{3107u, -.25f}, std::pair{70111u, .9f}};
    force.assign(long_frames, 0);
    morph.resize(long_frames);
    for (auto [index, value] : impulses) force[index] = value;
    for (uint32_t frame = 0; frame < long_frames; ++frame) morph[frame] = float(frame) / (long_frames - 1);
    const auto long_output = ConvolveFiniteModesGpu(gpu, force, morph, long_modes, rate, long_taps, -.7f);
    error = 0;
    double tail_error = 0;
    for (size_t frame = 0; frame < long_output.size(); ++frame) {
        double expected = 0;
        for (const auto &mode : long_modes) {
            const double amplitude = std::exp(std::lerp(std::log(mode.Amplitude0), std::log(mode.Amplitude1), morph[std::min(frame, size_t(long_frames - 1))]));
            for (auto [index, value] : impulses) {
                if (frame < index || frame - index >= long_taps) continue;
                const double lag = frame - index;
                expected += -.7f * value * amplitude * std::exp(-lag / (rate * mode.Decay)) * std::sin(2 * std::numbers::pi * mode.Frequency * lag / rate);
            }
        }
        error = std::max(error, std::abs(long_output[frame] - expected));
        if (frame >= impulses.back().first + long_taps) tail_error = std::max(tail_error, double(std::abs(long_output[frame])));
    }
    std::cout << "Finite modal long-tap max error " << error << ", post-support residual " << tail_error << '\n';
    Require(error < 2e-5 && tail_error < 2e-6, "Long finite modal response and cutoff cancellation");
    const auto zero = ConvolveFiniteModesGpu(gpu, force, morph, long_modes, rate, 1);
    Require(std::ranges::all_of(zero, [](float value) { return value == 0; }), "One-tap sine response is exactly zero");
}
}

int main() {
    try {
        auto gpu = CreateGpu();
        std::cout << "Core on " << DeviceName(gpu) << '\n';
        RandomTest(gpu);
        BatchConstantsTest(gpu);
        FirTest(gpu);
        FixedFirTest(gpu);
        FiniteModesTest(gpu);
        ModalTest(gpu);
        ModalMultiSimdTest(gpu);
        ModalStabilityTest(gpu);
        ModalForcedTest(gpu);
        MixTest(gpu);
        WaveTest();
        std::cout << "Core tests passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
