#pragma once
#include "Contact.h"
#include "ContactEvents.h"
#include "core/Gpu.h"

namespace surface_audio::rough {
struct GpuContact {
    GpuBuffer Shapes, Heights, Coefficients, State, Displacement, Maps, Forces, Active, Receiver, Events, EventStatistics;
    GpuKernel Kernel;
    unsigned BottomNodes{}, TopNodes{}, BottomModes{}, TopModes{}, EventStride{1}, DisplacementRefresh{32};
    double TimeStep{}, Spacing{}, Penalty{};
};
struct GpuTrace {
    std::vector<float> Velocity, Force, MeanSquareVelocity;
    std::vector<double> Displacement, Previous;
    std::array<ContactEventStatistics, 2> Events;
    bool operator==(const GpuTrace &) const = default;
};
struct GpuContactRun {
    const GpuContact *Model;
    unsigned Steps;
    double Offset, Separation, Speed;
};
// Uses extended-precision state and arithmetic; surfaces must share their grid spacing and have at most 256 total modes.
// A refresh interval of one evaluates every nodal displacement each step; larger intervals conservatively exclude separated contacts.
// Event stride selects unfiltered contact snapshots; it leaves the integration and output rates unchanged.
GpuContact MakeGpuContact(Gpu &, const Model &, const State &, std::span<const double> receiver, unsigned event_stride = 1, unsigned displacement_refresh = 32);
GpuTrace RenderGpu(Gpu &, const GpuContact &, unsigned steps, double offset, double separation, double speed);
// Runs independent trajectories concurrently; each run requires distinct mutable model buffers.
std::vector<GpuTrace> RenderGpu(Gpu &, std::span<const GpuContactRun>);
}
