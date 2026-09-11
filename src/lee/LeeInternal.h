#pragma once
#include "Lee.h"

namespace surface_audio::lee {
struct RenderContact {
    uint32_t Frame{}, Frames{};
    std::array<std::vector<double>, 4> Signals;
};
struct RenderPlan {
    uint32_t Frames{};
    FirBank Filters;
    std::vector<RenderContact> Contacts;
};
std::vector<double> HighpassCoefficients(uint32_t rate, const Settings &);
std::vector<uint32_t> ContactEnvelope(std::span<const double>, const Settings &, std::vector<double> &envelope);
uint32_t CheckedFrameCount(double);
std::array<Band, 4> FitContact(const std::array<std::vector<double>, 4> &, const Settings &);
RenderPlan PrepareSynthesis(const Analysis &, double rate, double gain, bool reverse);
}
