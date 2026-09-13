#include "ContactGpu.h"
#include "core/ExtendedFloatGpu.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace surface_audio::rough {
namespace {
struct Constants {
    unsigned BottomNodes, TopNodes, BottomModes, TopModes, Steps, OutputOffset, ShortEventSamples, EventStride, DisplacementRefresh, Padding;
    ExtendedFloat Offset, Increment, Separation, Stiffness, InverseTimeStep, InverseLength, TimeStep;
};
struct Job {
    Constants Parameters;
    uint64_t Shapes, Heights, Coefficients, State, Displacement, Maps, Forces, Receiver, Output, Active, Events, Statistics;
};
static_assert(sizeof(Constants) == 96 && sizeof(Job) == 192);
GpuBuffer ZeroBuffer(Gpu &gpu, size_t bytes) {
    const auto result = CreateBuffer(gpu, bytes);
    std::memset(result.Data, 0, bytes);
    return result;
}
ContactEventStatistics ReadEvents(const GpuContact &m, unsigned body) {
    const auto all = BufferSpan<unsigned>(m.EventStatistics), counters = all.subspan(body * EventStorageSize, EventStorageSize);
    ContactEventStatistics result{.Completed = counters[EventCompleted], .LeftCensored = counters[EventLeftCensored], .RightCensored = 0, .ZeroWork = counters[EventZeroWork], .Frames = all[EventFrames], .Samples = all[EventSamples], .ForceBelowPaperWeights = {counters[EventBelowWeight], counters[EventBelowTenWeights], counters[EventBelowHundredWeights]}, .ShorterThan100Microseconds = counters[EventShort], .Work = 0, .Force = {}, .Duration = {}, .PositiveWork = {}, .NegativeWork = {}};
    unsigned histogram{};
    for (auto *values : {&result.Force, &result.Duration, &result.PositiveWork, &result.NegativeWork}) {
        std::copy_n(counters.begin() + EventHistogramStart + histogram * EventHistogramSize, EventHistogramSize, values->begin());
        ++histogram;
    }
    for (const auto &node : BufferSpan<NodeContactEvent<ExtendedFloat>>(m.Events).subspan(body ? m.BottomNodes : 0, body ? m.TopNodes : m.BottomNodes)) {
        result.RightCensored += node.Frames != 0;
        result.LeftCensored += node.Frames != 0 && node.LeftCensored;
        const double work = double(node.TotalWork.High) + node.TotalWork.Low;
        if (!std::isfinite(work)) throw std::runtime_error("Nodal contact work became nonfinite");
        result.Work += work;
    }
    return result;
}
GpuTrace ReadTrace(const GpuContact &m, GpuBuffer output, unsigned steps) {
    const auto samples = BufferSpan<float>(output);
    const auto state = BufferSpan<ExtendedFloat>(m.State);
    const unsigned modes = m.BottomModes + m.TopModes;
    GpuTrace result{std::vector<float>(steps), std::vector<float>(steps), std::vector<float>(steps), std::vector<double>(modes), std::vector<double>(modes), {ReadEvents(m, 0), ReadEvents(m, 1)}};
    for (unsigned n = 0; n < steps; ++n) {
        if (!std::isfinite(samples[3 * n]) || !std::isfinite(samples[3 * n + 1]) || !std::isfinite(samples[3 * n + 2])) throw std::runtime_error("GPU contact integration became nonfinite");
        result.Velocity[n] = samples[3 * n];
        result.Force[n] = samples[3 * n + 1];
        result.MeanSquareVelocity[n] = samples[3 * n + 2];
    }
    for (unsigned k = 0; k < modes; ++k) {
        result.Displacement[k] = double(state[k].High) + state[k].Low;
        result.Previous[k] = double(state[modes + k].High) + state[modes + k].Low;
        if (!std::isfinite(result.Displacement[k]) || !std::isfinite(result.Previous[k])) throw std::runtime_error("GPU modal state became nonfinite");
    }
    return result;
}
}
GpuContact MakeGpuContact(Gpu &gpu, const Model &m, const State &s, std::span<const double> receiver, unsigned event_stride, unsigned displacement_refresh) {
    const size_t nodes = m.Bottom.Height.size() + m.Top.Height.size(), modes = m.Bottom.Modes.size() + m.Top.Modes.size();
    if (!event_stride || !displacement_refresh || !modes || modes > 256 || m.Bottom.Height.size() > (1u << 24) || m.Top.Height.size() > (1u << 24) || nodes * modes > UINT32_MAX || receiver.size() != modes || s.Displacement.size() != modes || s.Previous.size() != modes ||
        std::abs(m.Bottom.Step / m.Top.Step - 1) > 1e-12) throw std::invalid_argument("Invalid GPU rough-contact dimensions or grid spacing");
    std::vector<double> coefficients;
    coefficients.reserve(5 * modes);
    for (const auto *surface : {&m.Bottom, &m.Top})
        for (unsigned k = 0; k < surface->Modes.size(); ++k) {
            const auto mode = surface->Modes[k];
            const auto basis = std::span{surface->Shapes}.subspan(k * surface->Height.size(), surface->Height.size());
            const double maximum = std::abs(*std::ranges::max_element(basis, {}, [](double value) { return std::abs(value); }));
            coefficients.insert(coefficients.end(), {mode.Stiffness, mode.Retention, mode.Compliance, surface->Gravity[k], maximum});
        }
    return {UploadExtended(gpu, m.Bottom.Shapes, m.Top.Shapes), UploadExtended(gpu, m.Bottom.Height, m.Top.Height), UploadExtended(gpu, coefficients), UploadExtended(gpu, s.Displacement, s.Previous), CreateBuffer(gpu, nodes * sizeof(ExtendedFloat)), CreateBuffer(gpu, nodes * 64), CreateBuffer(gpu, nodes * sizeof(ExtendedFloat)), CreateBuffer(gpu, std::bit_ceil(nodes) * sizeof(unsigned)), UploadExtended(gpu, receiver), ZeroBuffer(gpu, nodes * sizeof(NodeContactEvent<ExtendedFloat>)), ZeroBuffer(gpu, 2 * EventStorageSize * sizeof(unsigned)), CreateKernel(gpu, "RoughContactPenalty"), unsigned(m.Bottom.Height.size()), unsigned(m.Top.Height.size()), unsigned(m.Bottom.Modes.size()), unsigned(m.Top.Modes.size()), event_stride, displacement_refresh, m.TimeStep, m.Bottom.Step, m.Penalty};
}
GpuTrace RenderGpu(Gpu &gpu, const GpuContact &m, unsigned steps, double offset, double separation, double speed) {
    const GpuContactRun run{&m, steps, offset, separation, speed};
    return std::move(RenderGpu(gpu, std::span{&run, 1})[0]);
}
std::vector<GpuTrace> RenderGpu(Gpu &gpu, std::span<const GpuContactRun> runs) {
    if (runs.empty() || runs.size() > UINT32_MAX) throw std::invalid_argument("Invalid GPU contact batch size");
    unsigned steps{}, lanes = 1024;
    std::vector<GpuBuffer> outputs;
    outputs.reserve(runs.size());
    for (const auto &run : runs) {
        if (!run.Model || !run.Model->DisplacementRefresh || !run.Steps || run.Steps > UINT32_MAX / 3 || !std::isfinite(run.Offset) || !std::isfinite(run.Separation) || !std::isfinite(run.Speed)) throw std::invalid_argument("Invalid GPU contact motion or output size");
        for (const auto &other : runs.first(size_t(&run - runs.data())))
            if (run.Model->State.Address == other.Model->State.Address) throw std::invalid_argument("GPU contact batch contains a shared mutable model");
        steps = std::max(steps, run.Steps);
        lanes = std::min(lanes, run.Model->Kernel.MaxThreads / 32 * 32);
        outputs.push_back(CreateBuffer(gpu, size_t(run.Steps) * 3 * sizeof(float)));
    }
    const auto job_buffer = CreateBuffer(gpu, runs.size() * sizeof(Job));
    const auto jobs = BufferSpan<Job>(job_buffer);
    constexpr unsigned block_steps = 256;
    for (unsigned start = 0; start < steps;) {
        const unsigned count = std::min(block_steps, steps - start);
        for (size_t i = 0; i < runs.size(); ++i) {
            const auto &run = runs[i];
            const auto &m = *run.Model;
            const double event_step = m.TimeStep * m.EventStride, short_samples = 1e-4 / event_step;
            const unsigned short_event_samples = short_samples > UINT32_MAX ? 0 : unsigned(std::max(1., std::ceil(short_samples - 1e-12 * std::max(1., short_samples))));
            const Constants c{m.BottomNodes, m.TopNodes, m.BottomModes, m.TopModes, start < run.Steps ? std::min(count, run.Steps - start) : 0, start, short_event_samples, m.EventStride, m.DisplacementRefresh, 0, SplitFloat((run.Offset + run.Speed * m.TimeStep * start) / m.Spacing), SplitFloat(run.Speed * m.TimeStep / m.Spacing), SplitFloat(run.Separation), SplitFloat(m.Penalty * m.Spacing), SplitFloat(1 / (2 * m.TimeStep)), SplitFloat(1 / (m.Spacing * (m.BottomNodes - 1))), SplitFloat(event_step)};
            jobs[i] = {c, m.Shapes.Address, m.Heights.Address, m.Coefficients.Address, m.State.Address, m.Displacement.Address, m.Maps.Address, m.Forces.Address, m.Receiver.Address, outputs[i].Address, m.Active.Address, m.Events.Address, m.EventStatistics.Address};
        }
        BeginGpu(gpu);
        const GpuBinding binding{job_buffer, 0};
        DispatchGroupsGpu(gpu, runs[0].Model->Kernel, std::span{&binding, 1}, {unsigned(runs.size()), 1, 1}, {lanes, 1, 1});
        SubmitGpu(gpu);
        WaitGpu(gpu);
        for (const auto &run : runs) {
            const auto counters = BufferSpan<unsigned>(run.Model->EventStatistics);
            if (counters[EventOverflow] || counters[EventStorageSize + unsigned(EventOverflow)]) throw std::overflow_error("Contact event counter overflow");
        }
        start += count;
    }
    std::vector<GpuTrace> result;
    result.reserve(runs.size());
    for (size_t i = 0; i < runs.size(); ++i) result.push_back(ReadTrace(*runs[i].Model, outputs[i], runs[i].Steps));
    return result;
}
}
