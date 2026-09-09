#include "Convolution.h"

#include <arm_neon.h>
#include <stdexcept>

namespace surface_audio {
void Convolve(const FirBlock &p, std::span<const float> coefficients, std::span<const float> excitation, std::span<float> output) {
    if (!p.TapCount || coefficients.size() != std::size_t(p.TapCount) * p.FrameCount || excitation.size() != std::size_t(p.TapCount) - 1 + p.FrameCount || output.size() != p.FrameCount) throw std::invalid_argument("Invalid FIR block dimensions");
    for (uint32_t frame = 0; frame < p.FrameCount; ++frame) {
        const auto *filter = coefficients.data() + std::size_t(frame) * p.TapCount;
        const auto *end = excitation.data() + p.TapCount - 1 + frame;
        float32x4_t sum = vdupq_n_f32(0.f);
        uint32_t lag = 0;
        for (; lag + 4 <= p.TapCount; lag += 4) {
            const auto reversed = vrev64q_f32(vld1q_f32(end - lag - 3));
            const auto samples = vextq_f32(reversed, reversed, 2);
            sum = vfmaq_f32(sum, vld1q_f32(filter + lag), samples);
        }
        float value = vaddvq_f32(sum);
        for (; lag < p.TapCount; ++lag) value += filter[lag] * end[-std::ptrdiff_t(lag)];
        output[frame] = value;
    }
}
} // namespace surface_audio
