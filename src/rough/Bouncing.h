#pragma once
#include "BouncingDynamics.h"
#include "ContactGpuTypes.h"
#include "core/Gpu.h"
#include <array>
#include <vector>

namespace surface_audio::rough {
struct CorrelatedPlate {
    std::array<double, 12> Transition;
    double Cutoff, VelocityDeviation, AccelerationDeviation, Bandwidth;
};
struct BounceSimulation {
    double Restitution{.8}, Acceleration{5}, CorrelationTime{1.11}, TimeStep{.005}, StickFactor{5};
    unsigned Streams{256}, Flights{4000}, Discard{400}, TraceSamples{0};
    uint64_t MaxSteps{100000000}, Seed{2016};
    bool RelativeSticking{true};
};
struct BounceSimulationResult {
    std::vector<BounceStatistics> Statistics;
    std::vector<std::array<float, 4>> Trace;
    double Seconds;
};
// Unit natural angular frequency and unit height variance; cutoff is the extra first-order low-pass frequency.
CorrelatedPlate MakeCorrelatedPlate(double correlation_time, double dt, double cutoff = 10);
// Trace channels are plate height/velocity and ball height/velocity; Statistics reports incomplete flight counts at MaxSteps.
// Runs independent realizations on Metal; each histogram spans normalized take-off velocities [-5,15).
BounceSimulationResult SimulateBouncing(Gpu &, BounceSimulation);
}
