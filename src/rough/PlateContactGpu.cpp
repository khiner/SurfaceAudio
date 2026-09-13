#include "PlateContactGpu.h"
#include "core/ExtendedFloatGpu.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace surface_audio::rough {
namespace {
struct Constants {
    unsigned SlaveX, SlaveY, MasterX, MasterY, SlaveModes, MasterModes, SlaveBase, MasterBase;
    unsigned BeginX, BeginY, Width, Height, Top, Capacity;
    ExtendedFloat OffsetX, OffsetY, RatioX, RatioY, Area, Separation, DisplacementBound, MotionX, MotionY;
};
static_assert(sizeof(Constants) == 128);
PlateSurfaceGpu UploadSurface(Gpu &gpu, const PlateSurface &s) {
    const auto shapes = [&](std::span<const double> values) { return values.empty() ? CreateBuffer(gpu, sizeof(ExtendedFloat)) : UploadExtended(gpu, values); };
    std::vector<std::array<double, 2>> bounds(s.Modes.size());
    for (unsigned k = 0; k < s.Modes.size(); ++k) {
        const auto x = std::ranges::minmax(std::span(s.XShapes).subspan(k * s.Grid.XNodes, s.Grid.XNodes));
        const auto y = std::ranges::minmax(std::span(s.YShapes).subspan(k * s.Grid.YNodes, s.Grid.YNodes));
        const auto range = std::minmax({x.min * y.min, x.min * y.max, x.max * y.min, x.max * y.max});
        bounds[k] = {range.first, range.second};
    }
    const auto heights = std::ranges::minmax(s.Height);
    return {shapes(s.XShapes), shapes(s.YShapes), UploadExtended(gpu, s.Height), std::move(bounds), std::max(std::abs(heights.min), std::abs(heights.max))};
}
void DispatchContacts(Gpu &gpu, const PlateContactGpu &device, const PlateContactModel &m, bool candidates, double offset_x, double offset_y, double separation, double displacement_bound, std::array<double, 3> margin = {}) {
    const auto [mx, my, roundoff] = margin;
    for (unsigned pass = 0; pass < (m.TwoPass ? 2u : 1u); ++pass) {
        const bool top = pass == 0;
        const auto &slave = top ? m.Top : m.Bottom, &master = top ? m.Bottom : m.Top;
        const auto &source = top ? device.Top : device.Bottom, &target = top ? device.Bottom : device.Top;
        const auto &a = slave.Grid, &b = master.Grid;
        const double dx = top ? offset_x : -offset_x, dy = top ? offset_y : -offset_y;
        const auto left = PlateOverlap(a, b, dx - mx, dy - my), right = candidates ? PlateOverlap(a, b, dx + mx, dy + my) : left;
        const std::array window{std::min(left[0], right[0]), std::max(left[1], right[1]), std::min(left[2], right[2]), std::max(left[3], right[3])};
        const unsigned width = window[1] - window[0], height = window[3] - window[2];
        if (!width || !height) continue;
        const Constants c{a.XNodes, a.YNodes, b.XNodes, b.YNodes, candidates ? 0 : unsigned(slave.Modes.size()), candidates ? 0 : unsigned(master.Modes.size()),
            !candidates && top ? unsigned(m.Bottom.Modes.size()) : 0, !candidates && !top ? unsigned(m.Bottom.Modes.size()) : 0,
            window[0], window[2], width, height, unsigned(top), device.Capacity,
            SplitFloat((a.OriginX + dx - b.OriginX) / b.StepX), SplitFloat((a.OriginY + dy - b.OriginY) / b.StepY), SplitFloat(a.StepX / b.StepX), SplitFloat(a.StepY / b.StepY),
            candidates ? ExtendedFloat{} : SplitFloat(a.StepX * a.StepY), SplitFloat(separation), SplitFloat(displacement_bound),
            candidates ? SplitFloat((mx + roundoff) / b.StepX) : ExtendedFloat{}, candidates ? SplitFloat((my + roundoff) / b.StepY) : ExtendedFloat{}};
        const auto constants = BatchUpload(gpu, c);
        const std::array bindings{GpuBinding{constants, 0}, GpuBinding{source.Height, 5}, GpuBinding{target.Height, 6}, GpuBinding{device.Count, 9}, GpuBinding{device.Nodes, 10},
            GpuBinding{source.XShapes, 1}, GpuBinding{source.YShapes, 2}, GpuBinding{target.XShapes, 3}, GpuBinding{target.YShapes, 4}, GpuBinding{device.State, 7}, GpuBinding{device.Rows, 8}};
        DispatchGpu(gpu, candidates ? device.Filter : device.Detect, std::span(bindings).first(candidates ? 5 : 11), {width, height, 1}, {16, 16, 1});
    }
}
bool PrepareCached(Gpu &gpu, PlateContactGpu &device, const PlateContactModel &m, PlateContactState &s, double offset_x, double offset_y, double separation, double displacement_bound) {
    const double coordinate_magnitude = std::abs(offset_x) + std::abs(offset_y) + m.Top.Grid.OriginX + m.Top.Grid.OriginY + m.Bottom.Grid.OriginX + m.Bottom.Grid.OriginY +
        m.Top.Grid.XNodes * m.Top.Grid.StepX + m.Top.Grid.YNodes * m.Top.Grid.StepY + m.Bottom.Grid.XNodes * m.Bottom.Grid.StepX + m.Bottom.Grid.YNodes * m.Bottom.Grid.StepY;
    if (0x1p-46 * coordinate_magnitude > std::min({m.Top.Grid.StepX, m.Top.Grid.StepY, m.Bottom.Grid.StepX, m.Bottom.Grid.StepY})) return false;
    const double closing_bound = displacement_bound - separation;
    if (!std::isfinite(float(displacement_bound)) || !std::isfinite(float(separation)) || !std::isfinite(float(closing_bound))) return false;
    const auto &bounds = device.CandidateBounds;
    if (!device.CandidateValid || device.CandidateTwoPass != m.TwoPass || offset_x < bounds[0] || offset_x > bounds[1] || offset_y < bounds[2] || offset_y > bounds[3] || closing_bound > bounds[4]) {
        const double mx = .125 * std::min(m.Bottom.Grid.StepX, m.Top.Grid.StepX), my = .125 * std::min(m.Bottom.Grid.StepY, m.Top.Grid.StepY);
        const double mz = .01 * (device.Bottom.HeightBound + device.Top.HeightBound) + 1e-12;
        device.CandidateBounds = {offset_x - mx, offset_x + mx, offset_y - my, offset_y + my, closing_bound + mz};
        WaitGpu(gpu);
        BufferSpan<unsigned>(device.Count)[0] = 0;
        BeginGpu(gpu);
        DispatchContacts(gpu, device, m, true, offset_x, offset_y, 0, device.CandidateBounds[4], {mx, my, 0x1p-42 * coordinate_magnitude});
        SubmitGpu(gpu);
        WaitGpu(gpu);
        const unsigned count = BufferSpan<unsigned>(device.Count)[0];
        device.DenseCandidates = count > device.Capacity;
        device.Candidates.clear();
        if (!device.DenseCandidates) {
            const auto nodes = BufferSpan<unsigned>(device.Nodes).first(count);
            device.Candidates.assign(nodes.begin(), nodes.end());
            std::ranges::sort(device.Candidates);
        }
        device.CandidateValid = true;
        device.CandidateTwoPass = m.TwoPass;
    }
    if (device.DenseCandidates) return false;
    PreparePlateContactNodes(m, s, device.Candidates, offset_x, offset_y, separation, displacement_bound);
    return true;
}

}
PlateContactGpu MakePlateContactGpu(Gpu &gpu, const PlateContactModel &m, unsigned capacity, bool reuse_candidates) {
    const unsigned modes = unsigned(m.Bottom.Modes.size() + m.Top.Modes.size());
    if (!modes || modes > 256 || !capacity || m.Bottom.Height.size() + m.Top.Height.size() > UINT32_MAX) throw std::invalid_argument("Invalid GPU plate dimensions");
    for (const auto *s : {&m.Bottom, &m.Top})
        if (s->Grid.XNodes > (1u << 24) || s->Grid.YNodes > (1u << 24) || s->XShapes.size() > UINT32_MAX || s->YShapes.size() > UINT32_MAX)
            throw std::invalid_argument("GPU plate grid exceeds exact integer coordinate range");
    return {UploadSurface(gpu, m.Bottom), UploadSurface(gpu, m.Top), CreateBuffer(gpu, modes * sizeof(ExtendedFloat)), CreateBuffer(gpu, size_t(capacity) * (modes + 2) * sizeof(ExtendedFloat)), CreateBuffer(gpu, size_t(capacity) * sizeof(unsigned)), CreateBuffer(gpu, sizeof(unsigned)), CreateKernel(gpu, "PlateContactDetect"), reuse_candidates ? CreateKernel(gpu, "PlateContactCandidates") : GpuKernel{}, modes, capacity, {}, {}, {}, reuse_candidates, false, false, false};
}
void PreparePlateContactGpu(Gpu &gpu, PlateContactGpu &device, const PlateContactModel &m, PlateContactState &s, double offset_x, double offset_y, double separation) {
    if (device.Modes != m.Bottom.Modes.size() + m.Top.Modes.size() || s.Displacement.size() != device.Modes || !std::isfinite(offset_x) || !std::isfinite(offset_y) || !std::isfinite(separation))
        throw std::invalid_argument("Invalid GPU plate state or geometry");
    double displacement_bound{}, magnitude = device.Bottom.HeightBound + device.Top.HeightBound + std::abs(separation);
    for (unsigned k = 0; k < device.Modes; ++k) {
        if (!std::isfinite(float(s.Displacement[k]))) throw std::invalid_argument("Plate displacement exceeds GPU range");
        const bool top = k >= device.Bottom.ShapeBounds.size();
        const auto range = top ? device.Top.ShapeBounds[k - device.Bottom.ShapeBounds.size()] : device.Bottom.ShapeBounds[k];
        const double a = s.Displacement[k] * range[0], b = s.Displacement[k] * range[1];
        displacement_bound += top ? -std::min(a, b) : std::max(a, b);
        magnitude += std::max(std::abs(a), std::abs(b));
    }
    displacement_bound += 1e-11 * magnitude + 1e-30;
    if (device.ReuseCandidates && PrepareCached(gpu, device, m, s, offset_x, offset_y, separation, displacement_bound)) return;
    WaitGpu(gpu);
    const auto state = BufferSpan<ExtendedFloat>(device.State);
    for (unsigned k = 0; k < device.Modes; ++k) state[k] = SplitFloat(s.Displacement[k]);
    unsigned count{};
    for (;;) {
        BufferSpan<unsigned>(device.Count)[0] = 0;
        BeginGpu(gpu);
        DispatchContacts(gpu, device, m, false, offset_x, offset_y, separation, displacement_bound);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        count = BufferSpan<unsigned>(device.Count)[0];
        if (count <= device.Capacity) break;
        device.Capacity = count;
        device.Rows = CreateBuffer(gpu, size_t(count) * (device.Modes + 2) * sizeof(ExtendedFloat));
        device.Nodes = CreateBuffer(gpu, size_t(count) * sizeof(unsigned));
    }
    s.Jacobian.resize(size_t(count) * device.Modes);
    s.Penetration.resize(count);
    s.Area.resize(count);
    device.Order.resize(count);
    std::iota(device.Order.begin(), device.Order.end(), 0u);
    const auto nodes = BufferSpan<unsigned>(device.Nodes);
    std::ranges::sort(device.Order, {}, [&](unsigned row) { return nodes[row]; });
    const auto rows = BufferSpan<ExtendedFloat>(device.Rows);
    const auto value = [&](size_t index) { return double(rows[index].High) + rows[index].Low; };
    for (unsigned row = 0; row < count; ++row) {
        const size_t base = size_t(device.Order[row]) * (device.Modes + 2);
        s.Penetration[row] = value(base);
        s.Area[row] = value(base + 1);
        for (unsigned k = 0; k < device.Modes; ++k) s.Jacobian[size_t(row) * device.Modes + k] = value(base + 2 + k);
    }
}
}
