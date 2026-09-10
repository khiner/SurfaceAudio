#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace surface_audio {
struct GpuState;
struct Gpu {
    std::shared_ptr<GpuState> State;
};
struct GpuBuffer {
    void *Data{};
    size_t Size{};
    uint64_t Address{};
};
struct GpuKernel {
    uint32_t Index{}, MaxThreads{};
};
struct GpuGrid {
    uint32_t X{1}, Y{1}, Z{1};
};
struct GpuBinding {
    GpuBuffer Buffer;
    uint32_t Index{};
    size_t Offset{};
};

Gpu CreateGpu(std::string_view library_path = {});
std::string DeviceName(const Gpu &);
GpuBuffer CreateBuffer(Gpu &, size_t bytes);
GpuKernel CreateKernel(Gpu &, std::string_view name);

// Use one owner thread and access context-owned buffers on the CPU only after WaitGpu.
// BeginGpu waits before recycling command memory.
// Batch dispatches execute in dependency order.
void BeginGpu(Gpu &);
// Returns immutable constants valid until the next BeginGpu, with an 8 KiB batch capacity and 256-byte upload alignment.
GpuBuffer BatchUpload(Gpu &, std::span<const std::byte>);
template<typename T> GpuBuffer BatchUpload(Gpu &gpu, const T &value) {
    static_assert(std::is_trivially_copyable_v<T>);
    return BatchUpload(gpu, std::as_bytes(std::span<const T>{&value, 1}));
}
void DispatchGpu(Gpu &, GpuKernel, std::span<const GpuBinding>, GpuGrid threads, GpuGrid group = {64, 1, 1});
void DispatchGroupsGpu(Gpu &, GpuKernel, std::span<const GpuBinding>, GpuGrid groups, GpuGrid group);
uint64_t SubmitGpu(Gpu &);
void WaitGpu(Gpu &);

template<typename T> std::span<T> BufferSpan(GpuBuffer b) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (b.Size % sizeof(T)) throw std::invalid_argument("GPU buffer element size mismatch");
    return {static_cast<T *>(b.Data), b.Size / sizeof(T)};
}
template<typename T> GpuBuffer Upload(Gpu &gpu, std::span<const T> data) {
    static_assert(std::is_trivially_copyable_v<T>);
    auto b = CreateBuffer(gpu, data.size_bytes());
    std::memcpy(b.Data, data.data(), data.size_bytes());
    return b;
}
template<typename T> GpuBuffer Upload(Gpu &gpu, const T &value) { return Upload<T>(gpu, std::span<const T>{&value, 1}); }
}
