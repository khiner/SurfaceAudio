// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "SdtScalar.h"

#include <span>
#include <vector>

namespace surface_audio::sdt {
using ImpactParameters = ImpactParametersT<double>;
using FrictionParameters = FrictionParametersT<double>;
using FrictionState = FrictionStateT<double>;
using RollingParameters = RollingParametersT<double>;
using RollingState = RollingStateT<double>;
using ScrapingParameters = ScrapingParametersT<double>;
using ScrapingState = ScrapingStateT<double>;

struct ModeParameters {
    double Frequency{}, Decay{}, Mass{1}, ContactGain{1}, OutputGain{1};
};

// One contact pickup and one independent listening pickup per body. Zero frequency is an inertial mode.
struct Body {
    std::vector<double> Mass, Stiffness, B1, A1, A2, B0V, B1V, ContactGain, OutputGain, ForceGain;
    std::vector<double> PFromP, PFromV, VFromP, VFromV;
    std::vector<double> Position, PreviousPosition, Velocity, Force;
    double SampleRate{}, GainSum{};
};

struct ContactState {
    double Energy{};
    FrictionState Friction;
};

struct ContactSample {
    double Force{}, Position0{}, Position1{}, Velocity0{}, Velocity1{}, Output0{}, Output1{};
};

Body MakeBody(std::span<const ModeParameters> modes, double sample_rate, double fragment_size = 1);
void SetPosition(Body &body, double position);
void SetVelocity(Body &body, double velocity);
double Position(const Body &body);
double Velocity(const Body &body);
double Output(const Body &body);
void ApplyForce(Body &body, double force);
double PredictedEnergy(const Body &body, double contact_force);
ContactEnergyT<double> ContactEnergy(const Body &body);
void StepBody(Body &body);
ContactSample StepImpact(Body &body0, Body &body1, ContactState &state, ImpactParameters parameters, double external0 = 0, double external1 = 0);
ContactSample StepFriction(Body &body0, Body &body1, ContactState &state, FrictionParameters parameters, double noise_sample, double external0 = 0, double external1 = 0);
} // namespace surface_audio::sdt
