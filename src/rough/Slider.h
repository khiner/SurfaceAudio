#pragma once
#include "SliderDynamics.h"

namespace surface_audio::rough {
// Uses Table S1 geometry, the stated 2.5 N weight and Hertz constants; inertias follow the homogeneous 60x60x29 mm box approximation.
SliderParameters<double> InstrumentedSlider(double damping_ratio = .1);
SliderState<double> EquilibrateSlider(const SliderParameters<double> &, const SliderTrack<double> &, double force_tolerance = 1e-9);
double SensorStep(double time, double amplitude, double cutoff = .74, double quality = .656);
}
