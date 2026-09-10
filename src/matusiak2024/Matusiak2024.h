// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "matusiak2024/Friction.h"
#include <cstdint>
#include <vector>

namespace surface_audio::matusiak2024 {
enum class NumericalConvention : uint32_t { Paper2024,
                                            AuthorArchive2025 };
struct Parameters {
    double SampleRate{44100}, Length{.7}, Radius{.0005}, MaterialDensity{10059}, Tension{149.6}, Young{1.37e10};
    double Damping0{1.53714}, Damping1{.0087}, TorsionStiffness{.00030312}, PolarInertia{4.2e-10}, TorsionDamping{1.0 / 58};
    double BowWidth{.01}, BowPosition{.0786}, HairStiffness{4.8297e6}, HairDamping{5.7674e3};
    uint32_t BowPoints{5};
    FrictionParameters<double> Friction{};
    NumericalConvention Convention{NumericalConvention::Paper2024};
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
    double Step{}, Spacing{}, TorsionSpacing{}, Density{}, Bending{}, WaveSpeed{}, TorsionSpeed{}, TorsionFeedback{}, HairStiffness{}, HairDamping{}, IntegratedLoss{}, InitialEnergy{};
    uint64_t Steps{};
    std::vector<double> U{}, PreviousU{}, W{}, PreviousW{}, Hair{}, PreviousHair{}, Z{}, Velocity{}, MidpointZ{};
    std::vector<double> Interpolation{}, TorsionInterpolation{}, Coupling{};
    std::vector<uint32_t> InterpolationOffset{}, TorsionInterpolationOffset{};
    std::vector<double> NextU{}, NextW{}, NextHair{}, FreeU{}, FreeW{}, FreeHair{}, Force{}, Work{}, Jacobian{}, Residual{};
};
StringState MakeString(Parameters = {});
Energy StoredEnergy(const StringState &);
Sample Step(StringState &, double bow_velocity, double normal_force);
}
