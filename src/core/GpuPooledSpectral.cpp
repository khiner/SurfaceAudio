#include "GpuPooledSpectral.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace surface_audio {
namespace {
struct PoolBlock {
    uint32_t Pools, Size, Bins, Frames;
    float Weight, FloorSquared;
};
void Energy(Gpu &gpu, const GpuPooledSpectral &pool, GpuBuffer spectrum) {
    const std::array bindings{GpuBinding{pool.Parameters, 0}, GpuBinding{pool.Regions, 1}, GpuBinding{spectrum, 2}, GpuBinding{pool.Energy, 3}};
    DispatchGroupsGpu(gpu, pool.ReduceEnergy, bindings, {pool.Pools}, {256});
}
} // namespace

GpuPooledSpectral CreatePooledSpectral(Gpu &gpu, const SpectralLossGpu &spectral, std::span<const SpectralPoolBand> bands, SpectralPoolOptions options) {
    if (!std::isfinite(options.Weight) || options.Weight < 0) throw std::invalid_argument("Invalid pooled spectral weight");
    if (options.Weight == 0) return {.Options = options};
    if (spectral.Options.Scale != SpectralMagnitudeScale::Linear || options.Resolution >= spectral.Resolutions.size() || !options.EventSamples || options.EventSamples > spectral.Samples || !options.TimePoolSamples || bands.empty() || bands.size() > 16 || !std::isfinite(options.RelativeFloor) || options.RelativeFloor <= 0 || options.RelativeFloor > 1) throw std::invalid_argument("Invalid pooled spectral options or event extent");
    const auto &r = spectral.Resolutions[options.Resolution];
    const uint32_t bins = r.Size / 2 + 1;
    const auto frame_at = [&](uint32_t sample) { return uint32_t((uint64_t(sample) + r.Hop - 1) / r.Hop); };
    std::vector<SpectralPoolRegion> regions;
    std::vector<uint32_t> membership(size_t(r.Frames) * bins, UINT32_MAX);
    for (size_t band = 0; band < bands.size(); ++band) {
        const auto [low, high] = bands[band];
        if (!std::isfinite(low) || !std::isfinite(high) || low < 0 || low >= high || high > double(spectral.SampleRate) / 2) throw std::invalid_argument("Invalid pooled spectral band bounds");
        const uint32_t first_bin = uint32_t(std::ceil(double(low) * r.Size / spectral.SampleRate)), end_bin = uint32_t(std::ceil(double(high) * r.Size / spectral.SampleRate));
        if (first_bin >= end_bin || end_bin > bins) throw std::invalid_argument("Pooled spectral band has no FFT bins");
        for (uint64_t begin = 0; begin < options.EventSamples; begin += options.TimePoolSamples) {
            const uint32_t first_frame = frame_at(uint32_t(begin)), end_frame = frame_at(uint32_t(std::min<uint64_t>(begin + options.TimePoolSamples, options.EventSamples)));
            if (first_frame == end_frame) continue;
            if (regions.size() >= 65536) throw std::invalid_argument("Too many pooled spectral regions");
            const uint32_t index = uint32_t(regions.size());
            regions.push_back({first_bin, end_bin, first_frame, end_frame, uint32_t(band)});
            for (uint32_t frame = first_frame; frame < end_frame; ++frame)
                for (uint32_t bin = first_bin; bin < end_bin; ++bin) {
                    auto &owner = membership[size_t(frame) * bins + bin];
                    if (owner != UINT32_MAX) throw std::invalid_argument("Pooled spectral bands overlap");
                    owner = index;
                }
        }
    }
    const uint32_t pools = uint32_t(regions.size());
    const PoolBlock block{pools, r.Size, bins, r.Frames, options.Weight, spectral.Options.MagnitudeFloor * spectral.Options.MagnitudeFloor};
    const GpuPooledSpectral result{
        .Options = options, .Pools = pools, .Bands = uint32_t(bands.size()), .Parameters = Upload(gpu, block), .Regions = Upload<SpectralPoolRegion>(gpu, regions), .Membership = Upload<uint32_t>(gpu, membership), .TargetScale = CreateBuffer(gpu, pools * 2 * sizeof(float)), .Energy = CreateBuffer(gpu, pools * sizeof(float)), .Coefficients = CreateBuffer(gpu, pools * sizeof(float)), .Adjoint = CreateBuffer(gpu, size_t(r.Frames) * r.Size * sizeof(float)), .Loss = CreateBuffer(gpu, (size_t(pools) + 1) * sizeof(float)), .ReduceEnergy = CreateKernel(gpu, "SpectralPoolEnergy"), .Compare = CreateKernel(gpu, "SpectralPoolCompare"), .Differentiate = CreateKernel(gpu, "SpectralPoolAdjoint"), .Add = CreateKernel(gpu, "SpectralPoolAdd")
    };
    BeginGpu(gpu);
    Energy(gpu, result, r.Target);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto energy = BufferSpan<float>(result.Energy);
    std::vector<double> band_energy(bands.size()), band_count(bands.size());
    for (uint32_t pool = 0; pool < pools; ++pool) {
        if (!std::isfinite(energy[pool]) || energy[pool] < 0) throw std::invalid_argument("Nonfinite pooled target energy");
        const auto &region = regions[pool];
        const double count = double(region.EndFrame - region.FirstFrame) * (region.EndBin - region.FirstBin);
        band_energy[region.Band] += energy[pool] * count;
        band_count[region.Band] += count;
    }
    std::vector<double> band_rms(bands.size());
    for (size_t band = 0; band < bands.size(); ++band) band_rms[band] = std::sqrt(band_energy[band] / band_count[band] + block.FloorSquared);
    const double floor = std::max(double(options.RelativeFloor) * *std::max_element(band_rms.begin(), band_rms.end()), double(spectral.Options.MagnitudeFloor));
    const auto target_scale = BufferSpan<std::array<float, 2>>(result.TargetScale);
    for (uint32_t pool = 0; pool < pools; ++pool) {
        const double denominator = std::max(band_rms[regions[pool].Band], floor);
        target_scale[pool] = {energy[pool], float(denominator * denominator)};
        if (!std::isfinite(target_scale[pool][1]) || target_scale[pool][1] <= 0) throw std::invalid_argument("Pooled target normalization exceeds float range");
    }
    return result;
}

void EncodePooledSpectral(Gpu &gpu, const SpectralLossGpu &spectral, const GpuPooledSpectral &pool) {
    if (pool.Options.Weight == 0) return;
    const auto &r = spectral.Resolutions[pool.Options.Resolution];
    Energy(gpu, pool, r.Spectrum);
    const std::array compare{GpuBinding{pool.Parameters, 0}, GpuBinding{pool.Regions, 1}, GpuBinding{pool.Energy, 2}, GpuBinding{pool.TargetScale, 3}, GpuBinding{pool.Coefficients, 4}, GpuBinding{pool.Loss, 5}};
    DispatchGpu(gpu, pool.Compare, compare, {pool.Pools});
    const std::array adjoint{GpuBinding{r.Parameters, 0}, GpuBinding{pool.Membership, 1}, GpuBinding{pool.Coefficients, 2}, GpuBinding{r.Spectrum, 3}, GpuBinding{pool.Adjoint, 4}};
    DispatchGroupsGpu(gpu, pool.Differentiate, adjoint, {r.Frames}, {256});
    const std::array add{GpuBinding{r.Parameters, 0}, GpuBinding{pool.Parameters, 1}, GpuBinding{pool.Adjoint, 2}, GpuBinding{pool.Loss, 3}, GpuBinding{spectral.Gradient, 4}, GpuBinding{spectral.Loss, 5}};
    DispatchGpu(gpu, pool.Add, add, {spectral.Samples});
}
} // namespace surface_audio
