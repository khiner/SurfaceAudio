// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "SdtScalar.h"
#ifndef __METAL_VERSION__
#include "Sdt.h"
#include <cstdint>
#endif

namespace surface_audio::sdt {
#ifdef __METAL_VERSION__
using GpuUint = uint;
#else
using GpuUint = uint32_t;
#endif

struct GpuDispatch {
    GpuUint Contacts{}, Frames{};
    float TimeStep{};
};

enum class GpuContactKind : GpuUint { Impact,
                                      Friction,
                                      External };

struct GpuContact {
    GpuUint Offset0{}, Count0{}, Offset1{}, Count1{};
    GpuContactKind Kind{};
    ImpactParametersT<float> Impact;
    FrictionParametersT<float> Friction;
};

enum class GpuContactError : GpuUint { None,
                                       PredictionDomain };

struct GpuContactState {
    float Energy{};
    FrictionStateT<float> Friction;
    GpuContactError Error{};
};

struct GpuMode {
    float Mass{}, Stiffness{}, PDelta{}, PFromV{}, VFromP{}, VDelta{}, PFromForce{}, VFromForce{}, ContactGain{}, OutputGain{}, ForceGain{};
};

struct GpuModeState {
    float Position{}, PreviousPosition{}, Velocity{}, Force{}, PositionError{}, VelocityError{};
};

struct GpuInput {
    float External0{}, External1{}, Noise{};
};

struct GpuOutput {
    float Force{}, Position0{}, Position1{}, Velocity0{}, Velocity1{}, Output0{}, Output1{};
};

enum class GpuSurfaceKind : GpuUint { Rolling,
                                      Scraping };

struct GpuSurfaceParameters {
    GpuSurfaceKind Kind{};
    RollingParametersT<float> Rolling;
    ScrapingParametersT<float> Scraping;
};

struct GpuSurfaceState {
    RollingStateT<float> Rolling;
    ScrapingStateT<float> Scraping;
};

#ifndef __METAL_VERSION__
struct GpuBodyRange {
    uint32_t Offset{}, Count{};
};
GpuBodyRange AppendBody(const Body &body, std::vector<GpuMode> &modes, std::vector<GpuModeState> &states);
void ValidateGpuContactStates(std::span<const GpuContactState> states);
#endif
} // namespace surface_audio::sdt
