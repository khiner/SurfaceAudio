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
// Writes IEEE float WAV at the supplied gain from interleaved samples.
void WriteWave(const std::filesystem::path &, uint32_t sample_rate, uint16_t channels, std::span<const float> samples);
}
