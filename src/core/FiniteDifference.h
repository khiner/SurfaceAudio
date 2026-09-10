#pragma once

#ifndef __METAL_VERSION__
#include <cmath>
#endif

namespace surface_audio {
template<typename T> inline T SecondDifference(T left, T center, T right) { return left - T(2) * center + right; }

#ifndef __METAL_VERSION__
// Returns the minimum stable grid spacing for an explicit damped stiff string.
inline double StiffStringMinimumSpacing(double wave_speed, double stiffness_squared, double loss1, double time_step) {
    const double wave_step = wave_speed * time_step, a = wave_step * wave_step + 4 * loss1 * time_step;
    return std::sqrt(.5 * (a + std::sqrt(a * a + 16 * stiffness_squared * time_step * time_step)));
}
#endif
}
