#include "AudioFile.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace surface_audio {
WaveData ReadWave(const std::filesystem::path &path) {
    static_assert(std::endian::native == std::endian::little);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open WAV: " + path.string());
    const auto size = file.tellg();
    if (size < 12 || uint64_t(size) > uint64_t(UINT32_MAX) + 8) throw std::runtime_error("Invalid WAV size");
    std::vector<char> bytes(static_cast<size_t>(size));
    file.seekg(0);
    file.read(bytes.data(), size);
    if (!file || std::memcmp(bytes.data(), "RIFF", 4) || std::memcmp(bytes.data() + 8, "WAVE", 4)) throw std::runtime_error("Invalid RIFF WAV");
    const auto read = [&]<typename T>(size_t offset) {
        if (offset > bytes.size() || bytes.size() - offset < sizeof(T)) throw std::runtime_error("Truncated WAV");
        T value;
        std::memcpy(&value, bytes.data() + offset, sizeof(T));
        return value;
    };
    const size_t end = size_t(read.operator()<uint32_t>(4)) + 8;
    if (end < 12 || end > bytes.size()) throw std::runtime_error("Truncated WAV RIFF");
    WaveData result;
    uint16_t format{}, bits{}, alignment{};
    size_t data_offset{}, data_size{};
    for (size_t offset = 12; offset + 8 <= end;) {
        const size_t length = read.operator()<uint32_t>(offset + 4), payload = offset + 8;
        if (length > end - payload) throw std::runtime_error("Truncated WAV chunk");
        if ((length & 1) && length == end - payload) throw std::runtime_error("Missing WAV chunk padding");
        if (!std::memcmp(bytes.data() + offset, "fmt ", 4)) {
            if (length < 16) throw std::runtime_error("Truncated WAV format");
            format = read.operator()<uint16_t>(payload);
            result.Channels = read.operator()<uint16_t>(payload + 2);
            result.SampleRate = read.operator()<uint32_t>(payload + 4);
            alignment = read.operator()<uint16_t>(payload + 12);
            bits = read.operator()<uint16_t>(payload + 14);
            if (format == 0xfffe) {
                constexpr unsigned char guid_tail[]{0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71};
                if (length < 40 || read.operator()<uint16_t>(payload + 16) < 22 || read.operator()<uint16_t>(payload + 18) != bits || std::memcmp(bytes.data() + payload + 26, guid_tail, sizeof(guid_tail))) throw std::runtime_error("Unsupported extensible WAV format");
                format = read.operator()<uint16_t>(payload + 24);
            }
        } else if (!std::memcmp(bytes.data() + offset, "data", 4)) {
            if (data_offset) throw std::runtime_error("Multiple WAV data chunks");
            data_offset = payload;
            data_size = length;
        }
        offset = payload + length + (length & 1);
    }
    if (!result.Channels || !result.SampleRate || !data_offset || !data_size || !bits || bits % 8 || alignment != uint32_t(result.Channels) * (bits / 8) || data_size % alignment) throw std::runtime_error("Invalid WAV dimensions");
    if (!((format == 1 && (bits == 16 || bits == 24 || bits == 32)) || (format == 3 && bits == 32))) throw std::runtime_error("Unsupported WAV sample format");
    result.Samples.resize(data_size / (bits / 8));
    for (size_t sample = 0; sample < result.Samples.size(); ++sample) {
        const size_t offset = data_offset + sample * (bits / 8);
        float value;
        if (format == 3) value = read.operator()<float>(offset);
        else if (bits == 16) value = float(read.operator()<int16_t>(offset)) / 32768;
        else if (bits == 32) value = float(double(read.operator()<int32_t>(offset)) / 2147483648.);
        else {
            const auto *p = reinterpret_cast<const unsigned char *>(bytes.data() + offset);
            const uint32_t packed = uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16);
            const int32_t signed_value = int32_t(packed ^ 0x800000) - 0x800000;
            value = float(signed_value) / 8388608;
        }
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite WAV samples");
        result.Samples[sample] = value;
    }
    return result;
}

void WriteWave(const std::filesystem::path &path, uint32_t sample_rate, uint16_t channels, std::span<const float> samples) {
    static_assert(std::endian::native == std::endian::little);
    if (!sample_rate || !channels || channels > UINT16_MAX / 4 || uint64_t(sample_rate) * channels * 4 > UINT32_MAX || samples.size() % channels || samples.size_bytes() > UINT32_MAX - 48) throw std::invalid_argument("Invalid WAV dimensions");
    for (float sample : samples)
        if (!std::isfinite(sample)) throw std::invalid_argument("Cannot write nonfinite audio");
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot create WAV: " + path.string());
    const auto write = [&](auto value) { file.write(reinterpret_cast<const char *>(&value), sizeof(value)); };
    file.write("RIFF", 4);
    write(uint32_t(samples.size_bytes() + 48));
    file.write("WAVEfmt ", 8);
    write(uint32_t(16));
    write(uint16_t(3));
    write(channels);
    write(sample_rate);
    write(uint32_t(sample_rate * channels * 4));
    write(uint16_t(channels * 4));
    write(uint16_t(32));
    file.write("fact", 4);
    write(uint32_t(4));
    write(uint32_t(samples.size() / channels));
    file.write("data", 4);
    write(uint32_t(samples.size_bytes()));
    file.write(reinterpret_cast<const char *>(samples.data()), std::streamsize(samples.size_bytes()));
    if (!file) throw std::runtime_error("WAV write failed: " + path.string());
}
} // namespace surface_audio
