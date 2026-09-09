#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace surface_audio {
struct Mode {
    float Frequency{}, Decay{}, Amplitude{};
};
struct ModalBlock {
    uint32_t Voices{}, Modes{}, Frames{};
};
struct ModalBank {
    uint32_t Voices{}, Modes{};
    // Coefficient SoA planes: real pole, imaginary pole, amplitude. State planes: real, imaginary.
    std::vector<float> Coefficients, State;
};

// Each mode has impulse response amplitude * exp(-t / decay) * sin(2*pi*frequency*t).
ModalBank MakeModalBank(std::span<const Mode> modes, uint32_t voices, float sample_rate);
// Input and output are voice-major. State persists across blocks. No allocation in rendering.
void RenderModal(ModalBank &, uint32_t frames, std::span<const float> input, std::span<float> output);
} // namespace surface_audio
