// SPDX-License-Identifier: GPL-3.0-or-later
#include "SdtGpu.h"

#include <array>
#include <bit>
#include <limits>
#include <stdexcept>
#include <string>

namespace surface_audio::sdt {
GpuBodyRange AppendBody(const Body &body, std::vector<GpuMode> &modes, std::vector<GpuModeState> &states) {
    if (modes.size() != states.size() || modes.size() + body.Position.size() > std::numeric_limits<uint32_t>::max()) throw std::invalid_argument("Invalid SDT GPU mode array dimensions");
    const GpuBodyRange range{uint32_t(modes.size()), uint32_t(body.Position.size())};
    for (size_t mode = 0; mode < body.Position.size(); ++mode) {
        const GpuMode packed{float(body.Mass[mode]), float(body.Stiffness[mode]), float(body.PFromP[mode] - 1), float(body.PFromV[mode]), float(body.VFromP[mode]), float(body.VFromV[mode] - 1), float(body.B1[mode]), float(body.B0V[mode] * body.B1[mode]), float(body.ContactGain[mode]), float(body.OutputGain[mode]), float(body.ForceGain[mode])};
        for (float value : std::bit_cast<std::array<float, 11>>(packed)) {
            if (!std::isfinite(value)) throw std::domain_error("SDT mode coefficients exceed GPU precision range");
        }
        modes.push_back(packed);
        states.push_back({float(body.Position[mode]), float(body.PreviousPosition[mode]), float(body.Velocity[mode]), float(body.Force[mode])});
    }
    return range;
}

void ValidateGpuContactStates(std::span<const GpuContactState> states) {
    for (size_t index = 0; index < states.size(); ++index) {
        if (states[index].Error != GpuContactError::None) throw std::domain_error("SDT GPU contact " + std::to_string(index) + " exceeded the affine prediction domain");
    }
}
} // namespace surface_audio::sdt
