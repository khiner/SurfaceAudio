#pragma once
#include "ExtendedFloat.h"
#include "Gpu.h"

namespace surface_audio {
GpuBuffer UploadExtended(Gpu &, std::span<const double>, std::span<const double> tail = {});
}
