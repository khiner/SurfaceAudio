#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace surface_audio {
struct WaveData {
    uint32_t SampleRate{};
    uint16_t Channels{};
    std::vector<float> Samples;
};
// Interleaved samples retain source scale and sample rate.
// PCM16/24/32 or float32; extensible files require valid bits equal to container bits.
WaveData ReadWave(const std::filesystem::path &);
// IEEE float WAV preserves signed samples and gain. Input is interleaved when channels > 1.
void WriteWave(const std::filesystem::path &, uint32_t sample_rate, uint16_t channels, std::span<const float> samples);
} // namespace surface_audio
