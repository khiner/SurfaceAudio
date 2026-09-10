// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "matusiak/Friction.h"
namespace surface_audio::matusiak {
struct BowDrive {
    float NormalForce{2.3433f}, Velocity{.3439f}, Acceleration{.8722f};
};
struct DistributedConstants {
    FrictionParameters<float> Friction;
    float Step{}, Spacing{}, TorsionSpacing{}, Density{}, Bending{}, WaveSpeed{}, TorsionSpeed{}, Radius{}, PolarInertia{}, Tension{};
    float HairMass{}, HairStiffness{}, HairDamping{}, Damping0{}, Damping1{}, TorsionDamping{}, XS{}, XT{}, XH{};
    unsigned Nodes{}, TorsionNodes{}, Contacts{}, Frames{}, Voices{}, StateStride{};
};
} // namespace surface_audio::matusiak
