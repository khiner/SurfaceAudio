#pragma once

namespace surface_audio::rough {
struct SurfaceSpectrumConstants {
    unsigned X, Y;
    float Spacing, Low, High, Exponent, Minimum, Scale;
    unsigned long Seed;
};
struct SeparationConstants { unsigned TrackX, SliderX, Y, Start; };
struct BounceConstants {
    float Transition[12];
    float Dt, Gravity, Restitution, StickVelocity, VelocityScale, Cutoff;
    unsigned Streams, Flights, Discard, Steps, TraceSamples, RelativeSticking;
    unsigned long Start, Seed;
};
// Descending compares successive take-offs; TakeoffPairs excludes departures without a predecessor.
struct BounceStatistics {
    unsigned long Steps, Sticking;
    unsigned Flights, Descending, NegativeVelocity, TakeoffPairs;
    float VelocitySum, FlightTimeSum, HeightSquare, VelocitySquare, AccelerationSquare;
    unsigned Histogram[200];
};
static_assert(sizeof(SurfaceSpectrumConstants) == 40 && sizeof(SeparationConstants) == 16);
static_assert(sizeof(BounceConstants) == 112 && sizeof(BounceStatistics) == 856);
}
