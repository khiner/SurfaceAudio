#pragma once
#include "core/NormalContact.h"
#ifdef __METAL_VERSION__
#define SLIDER_THREAD thread
#else
#define SLIDER_THREAD
#endif

namespace surface_audio::rough {
template<typename T> struct Asperity {
    T X{}, Y{}, Height{};
};
template<typename T> struct SliderParameters {
    Asperity<T> Points[9];
    T Mass{}, InertiaX{}, InertiaY{}, Gravity{}, Stiffness{}, Damping{};
};
template<typename T> struct SliderState {
    T Position[3]{}, Velocity[3]{};
};
template<typename T> struct SliderTrack {
    T Height[9]{}, Velocity[9]{};
};
template<typename T> struct SliderResult {
    T Force[9]{}, Acceleration[3]{}, ElasticEnergy{}, Dissipation{}, DrivePower{};
};
template<typename T> SliderResult<T> EvaluateSlider(SLIDER_THREAD const SliderParameters<T> &p, SLIDER_THREAD const SliderState<T> &s, SLIDER_THREAD const SliderTrack<T> &track) {
    SliderResult<T> result{};
    result.Acceleration[0] = -p.Mass * p.Gravity;
    for (unsigned j = 0; j < 9; ++j) {
        const auto point = p.Points[j];
        // Upward point displacement and the force/torque map use the same [1, y, -x] Jacobian.
        const T penetration = track.Height[j] + point.Height - s.Position[0] - point.Y * s.Position[1] + point.X * s.Position[2];
        const T rate = track.Velocity[j] - s.Velocity[0] - point.Y * s.Velocity[1] + point.X * s.Velocity[2];
        const auto contact = EvaluateNormalContact(penetration, rate, p.Stiffness, p.Stiffness * p.Damping, T(1.5), T(1.5));
        result.Force[j] = contact.Force;
        result.Acceleration[0] += contact.Force;
        result.Acceleration[1] += point.Y * contact.Force;
        result.Acceleration[2] -= point.X * contact.Force;
        result.ElasticEnergy += contact.Energy;
        result.Dissipation += contact.Dissipation;
        result.DrivePower += contact.Force * track.Velocity[j];
    }
    result.Acceleration[0] /= p.Mass;
    result.Acceleration[1] /= p.InertiaX;
    result.Acceleration[2] /= p.InertiaY;
    return result;
}
// Velocity prediction evaluates velocity-dependent damping at the end of the Verlet position step.
template<typename T> SliderResult<T> StepSlider(SLIDER_THREAD const SliderParameters<T> &p, SLIDER_THREAD SliderState<T> &s, SLIDER_THREAD const SliderTrack<T> &before, SLIDER_THREAD const SliderTrack<T> &after, T dt) {
    const auto initial = EvaluateSlider(p, s, before);
    SliderState<T> predicted{};
    for (unsigned k = 0; k < 3; ++k) {
        predicted.Position[k] = s.Position[k] + dt * s.Velocity[k] + dt * dt / 2 * initial.Acceleration[k];
        predicted.Velocity[k] = s.Velocity[k] + dt * initial.Acceleration[k];
    }
    const auto final = EvaluateSlider(p, predicted, after);
    for (unsigned k = 0; k < 3; ++k) {
        s.Position[k] = predicted.Position[k];
        s.Velocity[k] += dt / 2 * (initial.Acceleration[k] + final.Acceleration[k]);
    }
    return EvaluateSlider(p, s, after);
}
template<typename T> T SliderEnergy(SLIDER_THREAD const SliderParameters<T> &p, SLIDER_THREAD const SliderState<T> &s, SLIDER_THREAD const SliderTrack<T> &track) {
    return EvaluateSlider(p, s, track).ElasticEnergy + p.Mass * p.Gravity * s.Position[0] +
        (p.Mass * s.Velocity[0] * s.Velocity[0] + p.InertiaX * s.Velocity[1] * s.Velocity[1] + p.InertiaY * s.Velocity[2] * s.Velocity[2]) / 2;
}
}
#undef SLIDER_THREAD
