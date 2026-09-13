#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>

namespace surface_audio {
// Inputs and output are normalized (high, low) float pairs.
inline metal::float2 AddExtended(metal::float2 a, metal::float2 b) {
    const float sum = a.x + b.x, remainder = sum - a.x;
    const float error = (a.x - (sum - remainder)) + (b.x - remainder) + a.y + b.y;
    const float high = sum + error;
    return {high, error - (high - sum)};
}
inline metal::float2 MultiplyExtended(metal::float2 a, metal::float2 b) {
    const float product = a.x * b.x;
    const float error = metal::fma(a.x, b.x, -product) + a.x * b.y + a.y * b.x;
    const float high = product + error;
    return {high, error - (high - product)};
}
}
#else
#include <cmath>
#include <stdexcept>
namespace surface_audio {
struct ExtendedFloat {
    float High, Low;
};
inline ExtendedFloat SplitFloat(double value) {
    if (!std::isfinite(float(value))) throw std::invalid_argument("Value exceeds extended float range");
    return {float(value), float(value - double(float(value)))};
}
}
#endif
