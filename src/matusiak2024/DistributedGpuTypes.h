// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "matusiak2024/Friction.h"
namespace surface_audio::matusiak2024 {
struct BowDrive {
    float NormalForce{2.3433f}, Velocity{.3439f}, Acceleration{.8722f};
};
struct DistributedConstants {
    FrictionParameters<float> Friction;
    float Step{}, Spacing{}, TorsionSpacing{}, Density{}, Bending{}, WaveSpeed{}, TorsionSpeed{}, TorsionFeedback{}, Radius{}, PolarInertia{}, Tension{};
    float HairStiffness{}, HairDamping{}, Damping0{}, Damping1{}, TorsionDamping{}, XS{}, XT{}, XH{};
    unsigned Nodes{}, TorsionNodes{}, Contacts{}, Frames{}, Voices{}, StateStride{};
};
}
