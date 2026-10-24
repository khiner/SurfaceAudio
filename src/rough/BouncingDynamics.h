#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define BOUNCE_THREAD thread
#define BOUNCE_CONSTANT constant
#else
#include <cmath>
#define BOUNCE_THREAD
#define BOUNCE_CONSTANT
#endif

namespace surface_audio::rough {
template<typename T> struct PlateMotion { T Height{}, Velocity{}, Acceleration{}; };
template<typename Filter, typename State>
auto AdvancePlate(BOUNCE_CONSTANT const Filter &p, BOUNCE_THREAD State &state, decltype(Filter::Cutoff) normal) {
    State next;
    for (unsigned k = 0; k < 3; ++k)
        next[k] = p.Transition[4 * k] * state[0] + p.Transition[4 * k + 1] * state[1] + p.Transition[4 * k + 2] * state[2] + p.Transition[4 * k + 3] * normal;
    state = next;
    const auto velocity = p.Cutoff * (state[0] - state[2]);
    return PlateMotion<decltype(Filter::Cutoff)>{state[2], velocity, p.Cutoff * (state[1] - velocity)};
}
template<typename T> struct BouncingState {
    T Height{}, Velocity{}, FlightTime{}, TakeoffVelocity{}, TakeoffHeight{};
    bool Stuck{true};
};
template<typename T> struct BouncingEvent {
    T Velocity{}, Duration{}, Impulse{};
    bool Completed{};
};
// Requires a positive timestep, gravity and sticking threshold; plate heights are interpolated linearly.
// Free initial states require take-off height and velocity; the timestep must resolve successive impacts.
// Relative sticking uses abs(v_after - w); otherwise the threshold applies to signed v_after.
// Completed events report the preceding flight's take-off velocity, including departures after sticking.
template<typename T> BouncingEvent<T> AdvanceBouncing(BOUNCE_THREAD BouncingState<T> &s, PlateMotion<T> a, PlateMotion<T> b,
                                                      T dt, T gravity, T restitution, T stick_velocity, bool relative_sticking = true) {
#ifdef __METAL_VERSION__
    using metal::sqrt;
    using metal::abs;
#else
    using std::sqrt;
    using std::abs;
#endif
    const T slope = (b.Height - a.Height) / dt;
    if (s.Stuck) {
        if (a.Acceleration >= -gravity) {
            s.Height = b.Height;
            s.Velocity = b.Velocity;
            return {};
        }
        s.Height = a.Height;
        s.Velocity = a.Velocity;
        s.TakeoffHeight = s.Height;
        s.TakeoffVelocity = s.Velocity;
        s.FlightTime = 0;
        s.Stuck = false;
    }
    const T gap = s.Height - (b.Height - slope * dt), relative = s.Velocity - slope;
    const T discriminant = relative * relative + 2 * gravity * (gap > 0 ? gap : T(0));
    const T root = sqrt(discriminant);
    const T impact = relative >= 0 ? (relative + root) / gravity : gap > 0 ? 2 * gap / (root - relative) : T(0);
    if (impact > dt) {
        s.Height += dt * (s.Velocity - gravity * dt / 2);
        s.Velocity -= gravity * dt;
        s.FlightTime += dt;
        return {};
    }
    const T height = b.Height - slope * (dt - impact), incoming = s.Velocity - gravity * impact;
    const T plate_velocity = a.Velocity + (b.Velocity - a.Velocity) * (1 - (dt - impact) / dt);
    const T outgoing = -restitution * incoming + (1 + restitution) * plate_velocity;
    const BouncingEvent<T> event{s.TakeoffVelocity, s.FlightTime + impact, outgoing - incoming, true};
    s.Stuck = (relative_sticking ? abs(outgoing - plate_velocity) : outgoing) < stick_velocity;
    s.TakeoffHeight = height;
    s.TakeoffVelocity = outgoing;
    s.FlightTime = dt - impact;
    s.Height = s.Stuck ? b.Height : height + s.FlightTime * (outgoing - gravity * s.FlightTime / 2);
    s.Velocity = s.Stuck ? b.Velocity : outgoing - gravity * s.FlightTime;
    return event;
}
}

#undef BOUNCE_THREAD
#undef BOUNCE_CONSTANT
