#include "ExtendedFloatGpu.h"

namespace surface_audio {
GpuBuffer UploadExtended(Gpu &gpu, std::span<const double> values, std::span<const double> tail) {
    const auto buffer = CreateBuffer(gpu, (values.size() + tail.size()) * sizeof(ExtendedFloat));
    const auto data = BufferSpan<ExtendedFloat>(buffer);
    for (size_t i = 0; i < values.size(); ++i) data[i] = SplitFloat(values[i]);
    for (size_t i = 0; i < tail.size(); ++i) data[values.size() + i] = SplitFloat(tail[i]);
    return buffer;
}
}
