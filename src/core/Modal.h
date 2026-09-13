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
    // SoA planes store real pole, imaginary pole and amplitude coefficients, with real and imaginary state.
    std::vector<float> Coefficients, State;
};

// Each mode has impulse response amplitude * exp(-t / decay) * sin(2*pi*frequency*t).
ModalBank MakeModalBank(std::span<const Mode> modes, uint32_t voices, float sample_rate);
// Requires voice-major input/output and preserves state across blocks with allocation-free rendering.
void RenderModal(ModalBank &, uint32_t frames, std::span<const float> input, std::span<float> output);
}
