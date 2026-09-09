// SPDX-License-Identifier: GPL-3.0-or-later
// Numerical reference copied from SDT 0509de418e7bebc8b37866b3b4458e0acc8cf1f4.
#pragma once
#include "sdt/Sdt.h"

namespace surface_audio::sdt::reference {
void ApplyForce(Body &body, double force);
void StepBody(Body &body);
double PredictedEnergy(const Body &body, double contact_force);
double LimitForce(const Body &body0, const Body &body1, double &energy, double force, double tolerance = 0.001);
ContactSample StepImpact(Body &body0, Body &body1, ContactState &state, ImpactParameters parameters, double external0 = 0, double external1 = 0);
ContactSample StepFriction(Body &body0, Body &body1, ContactState &state, FrictionParameters parameters, double noise_sample, double external0 = 0, double external1 = 0);
} // namespace surface_audio::sdt::reference
