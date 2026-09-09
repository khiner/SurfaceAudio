#pragma once
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace surface_audio {
struct Adam {
    std::vector<double> Master, First, Second;
    std::vector<std::array<double, 2>> Bounds;
    uint64_t Step;
};

Adam CreateAdam(std::span<const float> initial, std::span<const std::array<double, 2>> bounds);
// beta1=.9, beta2=.999, epsilon=1e-8. Double master update, float output; returns clipped update count.
uint64_t UpdateAdam(Adam &, std::span<const float> gradient, std::span<float> parameters, double learning_rate);
} // namespace surface_audio
