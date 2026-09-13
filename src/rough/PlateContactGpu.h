#pragma once
#include "PlateContact.h"
#include "core/Gpu.h"

namespace surface_audio::rough {
struct PlateSurfaceGpu {
    GpuBuffer XShapes, YShapes, Height;
    std::vector<std::array<double, 2>> ShapeBounds;
    double HeightBound{};
};
struct PlateContactGpu {
    PlateSurfaceGpu Bottom, Top;
    GpuBuffer State, Rows, Nodes, Count;
    GpuKernel Detect, Filter;
    unsigned Modes{}, Capacity{};
    std::vector<unsigned> Order, Candidates;
    std::array<double, 5> CandidateBounds{};
    bool ReuseCandidates{}, CandidateValid{}, DenseCandidates{}, CandidateTwoPass{};
};
// Requires immutable surface grids and modes after upload; supports at most 256 total modes.
PlateContactGpu MakePlateContactGpu(Gpu &, const PlateContactModel &, unsigned capacity = 4096, bool reuse_candidates = true);
// Disable reuse_candidates for the full GPU reference; dense candidate sets use full GPU detection.
// Returns penetrating contacts in material-node order; grows storage and repeats detection as needed.
void PreparePlateContactGpu(Gpu &, PlateContactGpu &, const PlateContactModel &, PlateContactState &, double offset_x, double offset_y, double separation);
}
