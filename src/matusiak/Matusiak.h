// SPDX-License-Identifier: GPL-3.0-only
// Matusiak et al. (2025), doi:10.3389/frsip.2025.1525044. See docs/Matusiak.md.
#pragma once
#include "matusiak/Friction.h"
#include <cstdint>
#include <vector>

namespace surface_audio::matusiak {
struct Parameters {
    double SampleRate{44100}, Length{.7}, Radius{.0005}, Fundamental{98}, Tension{149.7415}, Young{1.37e10};
    double Damping0{1.53714}, Damping1{.0087}, TorsionStiffness{.00030312}, PolarInertia{4.2e-10}, TorsionDamping{1.0 / 58};
    double BowWidth{.01}, BowPosition{.0786}, HairMass{.0045}, HairStiffness{48297}, HairDamping{5.7674};
    uint32_t BowPoints{5};
    FrictionParameters<double> Friction{};
};
struct Energy {
    double String{}, Torsion{}, Hair{}, Bristle{};
};
inline double TotalEnergy(Energy e) { return e.String + e.Torsion + e.Hair + e.Bristle; }
struct Sample {
    double BridgeForce{}, RelativeVelocity{}, FrictionForce{}, BristleDissipation{}, Dissipation{}, InputPower{}, EnergyError{}, Residual{};
    uint32_t Iterations{};
    Energy Stored;
};
struct StringState {
    Parameters Config;
    double Step{}, Spacing{}, TorsionSpacing{}, Density{}, Bending{}, WaveSpeed{}, TorsionSpeed{}, HairMass{}, HairStiffness{}, HairDamping{}, IntegratedLoss{}, InitialEnergy{};
    uint64_t Steps{};
    std::vector<double> U{}, PreviousU{}, W{}, PreviousW{}, Hair{}, PreviousHair{}, Z{}, Velocity{}, MidpointZ{};
    std::vector<double> Interpolation{}, TorsionInterpolation{}, Coupling{};
    std::vector<uint32_t> InterpolationOffset{}, TorsionInterpolationOffset{};
    std::vector<double> NextU{}, NextW{}, NextHair{}, FreeU{}, FreeW{}, FreeHair{}, Force{}, Work{}, Jacobian{}, Residual{};
};
StringState MakeString(Parameters = {});
Energy StoredEnergy(const StringState &);
Sample Step(StringState &, double bow_velocity, double normal_force);
} // namespace surface_audio::matusiak
