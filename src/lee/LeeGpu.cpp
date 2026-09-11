#include "LeeInternal.h"
#include "core/GpuResample.h"
#include <array>
#include <cmath>
#include <stdexcept>

namespace surface_audio::lee {
Analysis AnalyzeGpu(Gpu &gpu, std::span<const float> input, uint32_t rate, Settings options) {
    const auto frames = CheckedFrameCount(input.size());
    const auto high = HighpassCoefficients(rate, options);
    for (float x : input)
        if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite Lee recording");
    if (input.empty()) return {rate, frames, options, {}, {}, {}};
    const std::vector<float> high_taps(high.begin(), high.end());
    const FirResampleJob detector{0, uint32_t(input.size()), 0, uint32_t(input.size()), 0, uint32_t(high.size()), 1, 1, int32_t(high.size() / 2)};
    const auto highpassed = ResampleFirGpu(gpu, input, high_taps, std::span{&detector, 1});
    const std::vector<double> precise(highpassed.begin(), highpassed.end());
    std::vector<double> envelope;
    auto onsets = ContactEnvelope(precise, options, envelope);
    if (onsets.empty()) return {rate, frames, options, std::move(onsets), std::move(envelope), {}};
    const auto bank = MakeFilterBank();
    std::vector<float> taps;
    std::array<uint32_t, 4> tap_offsets;
    for (size_t b = 0; b < 4; ++b) {
        tap_offsets[b] = taps.size();
        taps.insert(taps.end(), bank.Analysis[b].begin(), bank.Analysis[b].end());
    }
    std::vector<FirResampleJob> jobs;
    uint32_t output_frames = 0;
    for (size_t i = 0; i < onsets.size(); ++i) {
        const uint32_t first = onsets[i], last = i + 1 < onsets.size() ? onsets[i + 1] : frames;
        for (size_t b = 0; b < 4; ++b) {
            const uint32_t band_frames = CheckedFrameCount((last - first + bank.Analysis[b].size() - 2) / Decimation[b] + 1);
            jobs.push_back({first, last - first, output_frames, band_frames, tap_offsets[b], uint32_t(bank.Analysis[b].size()), 1, Decimation[b], 0});
            output_frames = CheckedFrameCount(double(output_frames) + band_frames);
        }
    }
    const auto filtered = ResampleFirGpu(gpu, input, taps, jobs);
    std::vector<Contact> contacts;
    contacts.reserve(onsets.size());
    for (size_t i = 0; i < onsets.size(); ++i) {
        std::array<std::vector<double>, 4> bands;
        for (size_t b = 0; b < 4; ++b) {
            const auto &job = jobs[i * 4 + b];
            bands[b].assign(filtered.begin() + job.OutputOffset, filtered.begin() + job.OutputOffset + job.OutputFrames);
        }
        const uint32_t first = onsets[i], last = i + 1 < onsets.size() ? onsets[i + 1] : frames;
        contacts.push_back({first, last - first, FitContact(bands, options)});
    }
    return {rate, frames, options, std::move(onsets), std::move(envelope), std::move(contacts)};
}
std::vector<float> SynthesizeGpu(Gpu &gpu, const Analysis &analysis, double rate, double gain, bool reverse) {
    const auto plan = PrepareSynthesis(analysis, rate, gain, reverse);
    if (plan.Contacts.empty()) return std::vector<float>(plan.Frames);
    std::vector<float> input, taps;
    std::array<uint32_t, 4> tap_offsets;
    for (size_t b = 0; b < 4; ++b) {
        tap_offsets[b] = taps.size();
        taps.insert(taps.end(), plan.Filters.Synthesis[b].begin(), plan.Filters.Synthesis[b].end());
    }
    std::vector<FirResampleJob> jobs;
    uint32_t output_frames = 0;
    for (const auto &contact : plan.Contacts) {
        for (size_t b = 0; b < 4; ++b) {
            const auto input_offset = CheckedFrameCount(input.size()), input_frames = CheckedFrameCount(contact.Signals[b].size());
            CheckedFrameCount(double(input_offset) + input_frames);
            jobs.push_back({input_offset, input_frames, output_frames, contact.Frames, tap_offsets[b], uint32_t(plan.Filters.Synthesis[b].size()), Decimation[b], 1, int32_t((plan.Filters.Synthesis[b].size() - 1) / 2)});
            input.insert(input.end(), contact.Signals[b].begin(), contact.Signals[b].end());
            output_frames = CheckedFrameCount(double(output_frames) + contact.Frames);
        }
    }
    const auto fir = CreateFirResampleGpu(gpu, input, taps, jobs);
    struct OverlapContact {
        uint32_t Frame, Frames, Offset;
    };
    std::vector<OverlapContact> contacts;
    for (size_t i = 0; i < plan.Contacts.size(); ++i)
        contacts.push_back({plan.Contacts[i].Frame, plan.Contacts[i].Frames, jobs[i * 4].OutputOffset});
    const auto contact_buffer = Upload<OverlapContact>(gpu, contacts);
    const auto output = CreateBuffer(gpu, size_t(plan.Frames) * sizeof(float));
    const auto shape = Upload(gpu, std::array{uint32_t(contacts.size()), plan.Frames});
    const auto overlap = CreateKernel(gpu, "LeeOverlap");
    BeginGpu(gpu);
    EncodeFirResample(gpu, fir);
    const std::array bindings{GpuBinding{contact_buffer, 0}, GpuBinding{fir.Output, 1}, GpuBinding{output, 2}, GpuBinding{shape, 3}};
    DispatchGpu(gpu, overlap, bindings, {plan.Frames});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto samples = BufferSpan<float>(output);
    return {samples.begin(), samples.end()};
}
}
