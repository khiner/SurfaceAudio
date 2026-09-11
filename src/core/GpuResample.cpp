#include "core/GpuResample.h"
#include <algorithm>
#include <array>
#include <limits>

namespace surface_audio {
FirResampleGpu CreateFirResampleGpu(Gpu &gpu, std::span<const float> input, std::span<const float> taps, std::span<const FirResampleJob> jobs) {
    if (jobs.empty() || jobs.size() > UINT32_MAX) throw std::invalid_argument("FIR resampling requires bounded nonempty jobs");
    uint32_t output_frames{}, maximum_frames{};
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    ranges.reserve(jobs.size());
    for (const auto &job : jobs) {
        const uint64_t end = uint64_t(job.OutputOffset) + job.OutputFrames;
        const uint64_t last_offset = uint64_t(job.OutputFrames ? job.OutputFrames - 1 : 0) * job.Downsample;
        if (!job.InputFrames || !job.OutputFrames || !job.Taps || !job.Upsample || !job.Downsample || end > UINT32_MAX ||
            uint64_t(job.InputOffset) + job.InputFrames > std::min<uint64_t>(input.size(), UINT32_MAX) ||
            uint64_t(job.TapOffset) + job.Taps > std::min<uint64_t>(taps.size(), UINT32_MAX) || job.Taps > INT32_MAX ||
            int64_t(job.Delay) - job.Taps < INT32_MIN || last_offset > INT32_MAX ||
            int64_t(last_offset) + job.Delay > INT32_MAX || job.Upsample > INT32_MAX || job.Downsample > INT32_MAX)
            throw std::invalid_argument("Invalid FIR resampling dimensions");
        output_frames = std::max(output_frames, uint32_t(end));
        maximum_frames = std::max(maximum_frames, job.OutputFrames);
        ranges.emplace_back(job.OutputOffset, uint32_t(end));
    }
    std::ranges::sort(ranges);
    for (size_t i = 1; i < ranges.size(); ++i)
        if (ranges[i].first < ranges[i - 1].second) throw std::invalid_argument("FIR resampling outputs overlap");
    const auto output = CreateBuffer(gpu, size_t(output_frames) * sizeof(float));
    std::ranges::fill(BufferSpan<float>(output), 0.f);
    return {.Count = uint32_t(jobs.size()), .MaximumFrames = maximum_frames, .Input = Upload(gpu, input), .Taps = Upload(gpu, taps), .Jobs = Upload(gpu, jobs), .Output = output, .Kernel = CreateKernel(gpu, "ResampleFir")};
}
void EncodeFirResample(Gpu &gpu, const FirResampleGpu &plan) {
    const auto count = BatchUpload(gpu, plan.Count);
    const std::array bindings{GpuBinding{plan.Jobs, 0}, GpuBinding{plan.Input, 1}, GpuBinding{plan.Taps, 2}, GpuBinding{plan.Output, 3}, GpuBinding{count, 4}};
    DispatchGpu(gpu, plan.Kernel, bindings, {plan.MaximumFrames, plan.Count, 1});
}
std::vector<float> ResampleFirGpu(Gpu &gpu, std::span<const float> input, std::span<const float> taps, std::span<const FirResampleJob> jobs) {
    const auto plan = CreateFirResampleGpu(gpu, input, taps, jobs);
    BeginGpu(gpu);
    EncodeFirResample(gpu, plan);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto samples = BufferSpan<float>(plan.Output);
    return {samples.begin(), samples.end()};
}
}
