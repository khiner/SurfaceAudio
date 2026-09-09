#include "Modal.h"

#include <arm_neon.h>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace surface_audio {
ModalBank MakeModalBank(std::span<const Mode> modes, uint32_t voices, float sample_rate) {
    if (modes.empty() || !voices || !std::isfinite(sample_rate) || sample_rate <= 0 || modes.size() * uint64_t(voices) > UINT32_MAX / 3) throw std::invalid_argument("Invalid modal bank dimensions");
    const auto count = std::size_t(voices) * modes.size();
    ModalBank bank{.Voices = voices, .Modes = uint32_t(modes.size()), .Coefficients = std::vector<float>(3 * count), .State = std::vector<float>(2 * count)};
    for (std::size_t mode = 0; mode < modes.size(); ++mode) {
        const auto &p = modes[mode];
        if (!std::isfinite(p.Frequency) || !std::isfinite(p.Decay) || !std::isfinite(p.Amplitude) || p.Frequency <= 0 || p.Frequency >= sample_rate * 0.5f || p.Decay <= 0) throw std::invalid_argument("Invalid modal frequency, decay, or amplitude");
        const double radius = std::exp(-1.0 / (sample_rate * double(p.Decay)));
        const double angle = 2 * std::numbers::pi * p.Frequency / sample_rate;
        const double real = float(radius * std::cos(angle)), imaginary = float(radius * std::sin(angle));
        if (!(real * real + imaginary * imaginary < 1)) throw std::invalid_argument("Modal poles are not strictly stable at float precision");
        for (uint32_t voice = 0; voice < voices; ++voice) {
            const auto index = voice * modes.size() + mode;
            bank.Coefficients[index] = float(real);
            bank.Coefficients[count + index] = float(imaginary);
            bank.Coefficients[2 * count + index] = p.Amplitude;
        }
    }
    return bank;
}

void RenderModal(ModalBank &bank, uint32_t frames, std::span<const float> input, std::span<float> output) {
    const auto count = std::size_t(bank.Voices) * bank.Modes;
    if (!count || bank.Coefficients.size() != 3 * count || bank.State.size() != 2 * count || input.size() != std::size_t(bank.Voices) * frames || output.size() != input.size()) throw std::invalid_argument("Invalid modal block dimensions");
    const auto *pole_real = bank.Coefficients.data(), *pole_imaginary = pole_real + count, *amplitude = pole_imaginary + count;
    auto *real = bank.State.data(), *imaginary = real + count;
    for (uint32_t voice = 0; voice < bank.Voices; ++voice)
        for (uint32_t frame = 0; frame < frames; ++frame) {
            const auto sample = input[std::size_t(voice) * frames + frame];
            auto sum = vdupq_n_f32(0);
            uint32_t mode = 0;
            for (; mode + 4 <= bank.Modes; mode += 4) {
                const auto index = std::size_t(voice) * bank.Modes + mode;
                const auto old_real = vld1q_f32(real + index), old_imaginary = vld1q_f32(imaginary + index);
                const auto cosine = vld1q_f32(pole_real + index), sine = vld1q_f32(pole_imaginary + index);
                const auto next_real = vfmaq_n_f32(vfmaq_f32(vnegq_f32(vmulq_f32(sine, old_imaginary)), cosine, old_real), vld1q_f32(amplitude + index), sample);
                const auto next_imaginary = vfmaq_f32(vmulq_f32(cosine, old_imaginary), sine, old_real);
                vst1q_f32(real + index, next_real);
                vst1q_f32(imaginary + index, next_imaginary);
                sum = vaddq_f32(sum, next_imaginary);
            }
            float total = vaddvq_f32(sum);
            for (; mode < bank.Modes; ++mode) {
                const auto index = std::size_t(voice) * bank.Modes + mode;
                const float next_real = std::fma(amplitude[index], sample, std::fma(pole_real[index], real[index], -pole_imaginary[index] * imaginary[index]));
                const float next_imaginary = std::fma(pole_imaginary[index], real[index], pole_real[index] * imaginary[index]);
                real[index] = next_real;
                imaginary[index] = next_imaginary;
                total += next_imaginary;
            }
            output[std::size_t(voice) * frames + frame] = total;
        }
}
} // namespace surface_audio
