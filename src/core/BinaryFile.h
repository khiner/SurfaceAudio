#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace surface_audio {
// Native-layout arrays, including empty arrays.
template<typename T>
    requires std::is_trivially_copyable_v<T>
std::vector<T> ReadBinary(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open " + path.string());
    const auto bytes = file.tellg();
    if (bytes < 0 || uint64_t(bytes) % sizeof(T)) throw std::runtime_error("Invalid binary array " + path.string());
    std::vector<T> data(size_t(bytes) / sizeof(T));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(data.data()), std::streamsize(bytes));
    if (!file) throw std::runtime_error("Cannot read " + path.string());
    return data;
}
template<typename T>
    requires std::is_trivially_copyable_v<T>
void WriteBinary(const std::filesystem::path &path, std::span<const T> data) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(data.data()), std::streamsize(data.size_bytes()));
    if (!file) throw std::runtime_error("Cannot write " + path.string());
}
} // namespace surface_audio
