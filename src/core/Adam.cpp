#include "Adam.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace surface_audio {
Adam CreateAdam(std::span<const float> initial, std::span<const std::array<double, 2>> bounds) {
    if (initial.empty() || initial.size() != bounds.size()) throw std::invalid_argument("Invalid Adam dimensions");
    for (size_t index = 0; index < initial.size(); ++index)
        if (!std::isfinite(initial[index]) || !std::isfinite(bounds[index][0]) || !std::isfinite(bounds[index][1]) || bounds[index][0] > bounds[index][1] || initial[index] < bounds[index][0] || initial[index] > bounds[index][1]) throw std::invalid_argument("Invalid Adam initial parameters or bounds");
    return {{initial.begin(), initial.end()}, std::vector<double>(initial.size(), 0), std::vector<double>(initial.size(), 0), {bounds.begin(), bounds.end()}, 0};
}

uint64_t UpdateAdam(Adam &state, std::span<const float> gradient, std::span<float> parameters, double learning_rate) {
    if (gradient.size() != state.Master.size() || parameters.size() != state.Master.size() || !std::isfinite(learning_rate) || learning_rate <= 0 || !std::ranges::all_of(gradient, [](float value) { return std::isfinite(value); })) throw std::invalid_argument("Invalid Adam gradient, dimensions or learning rate");
    const double correction1 = 1 - std::pow(.9, double(state.Step + 1)), correction2 = 1 - std::pow(.999, double(state.Step + 1));
    uint64_t bound_updates = 0;
    for (size_t index = 0; index < parameters.size(); ++index) {
        const double derivative = gradient[index];
        state.First[index] = .9 * state.First[index] + .1 * derivative;
        state.Second[index] = .999 * state.Second[index] + .001 * derivative * derivative;
        const double proposal = state.Master[index] - learning_rate * (state.First[index] / correction1) / (std::sqrt(state.Second[index] / correction2) + 1e-8);
        state.Master[index] = std::clamp(proposal, state.Bounds[index][0], state.Bounds[index][1]);
        bound_updates += state.Master[index] != proposal;
        parameters[index] = float(state.Master[index]);
    }
    ++state.Step;
    return bound_updates;
}
} // namespace surface_audio
