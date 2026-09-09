#include "GpuDecimate.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>

namespace surface_audio {
namespace {
struct DecimateBlock {
    uint32_t InputFrames, OutputFrames, Channels, Factor, Radius;
};
double BesselI0(double x) {
    const double quarter_square{x * x / 4};
    double sum{1}, term{1};
    for (uint32_t order = 1; order < 128; ++order) {
        term *= quarter_square / (double(order) * order);
        sum += term;
        if (term <= sum * std::numeric_limits<double>::epsilon()) return sum;
    }
    throw std::runtime_error("Kaiser Bessel series did not converge");
}
void ValidateFactor(uint32_t factor) {
    if (!factor || factor > 128) throw std::invalid_argument("Invalid decimation factor");
}
} // namespace

std::vector<double> KaiserDecimateCoefficients(uint32_t factor) {
    ValidateFactor(factor);
    if (factor == 1) return {1};
    const uint32_t radius{64 * factor};
    const double cutoff{.475 / factor}, bessel{BesselI0(10)};
    std::vector<double> coefficients(2 * size_t(radius) + 1);
    double normalization{0};
    for (uint32_t tap = 0; tap < coefficients.size(); ++tap) {
        const double offset{double(tap) - radius}, position{offset / radius};
        const double sinc{offset == 0 ? 2 * cutoff : std::sin(2 * std::numbers::pi * cutoff * offset) / (std::numbers::pi * offset)};
        const double window{BesselI0(10 * std::sqrt(std::max(0., 1 - position * position))) / bessel};
        coefficients[tap] = sinc * window;
        normalization += coefficients[tap];
    }
    for (double &value : coefficients) value /= normalization;
    return coefficients;
}

GpuDecimatePlan CreateDecimatePlan(Gpu &gpu, uint32_t input_frames, uint32_t channels, uint32_t factor) {
    ValidateFactor(factor);
    if (!input_frames || !channels || uint64_t(input_frames) * channels > UINT32_MAX) throw std::invalid_argument("Invalid planar decimation dimensions");
    const uint32_t output_frames{(input_frames - 1) / factor + 1};
    const auto coefficients = KaiserDecimateCoefficients(factor);
    const std::vector<float> coefficients_float(coefficients.begin(), coefficients.end());
    return {input_frames, output_frames, channels, factor, Upload(gpu, DecimateBlock{input_frames, output_frames, channels, factor, factor == 1 ? 0 : 64 * factor}), Upload<float>(gpu, coefficients_float), CreateKernel(gpu, factor == 1 ? "DecimateCopy" : "DecimateKaiserSinc")};
}

void DispatchDecimateGpu(Gpu &gpu, const GpuDecimatePlan &plan, GpuBuffer input, GpuBuffer output) {
    if (!input.Data || !output.Data || !input.Address || !output.Address || input.Size != size_t(plan.InputFrames) * plan.Channels * sizeof(float) || output.Size != size_t(plan.OutputFrames) * plan.Channels * sizeof(float) || !plan.Parameters.Data || !plan.Parameters.Address || !plan.Coefficients.Data || !plan.Coefficients.Address) throw std::invalid_argument("Decimation buffer extent, null address or plan mismatch");
    const bool overlap = input.Address <= output.Address ? output.Address - input.Address < input.Size : input.Address - output.Address < output.Size;
    if (overlap) throw std::invalid_argument("Decimation input and output GPU address ranges overlap");
    const std::array bindings{GpuBinding{plan.Parameters, 0}, GpuBinding{plan.Coefficients, 1}, GpuBinding{input, 2}, GpuBinding{output, 3}};
    if (plan.Factor == 1) DispatchGpu(gpu, plan.Kernel, bindings, {plan.OutputFrames, plan.Channels}, {64});
    else DispatchGroupsGpu(gpu, plan.Kernel, bindings, {plan.OutputFrames, plan.Channels}, {128});
}
} // namespace surface_audio
