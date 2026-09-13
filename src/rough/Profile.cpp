#include "Profile.h"
#include "core/GpuFft.h"
#include <array>
#include <bit>
#include <cmath>
#include <numbers>

namespace surface_audio::rough {
namespace {
struct ProfileConstants {
    uint64_t Seed;
    unsigned Nodes, Size;
    float Spacing, Correlation, Rms;
};
struct WhiteConstants {
    uint64_t Seed;
    unsigned Nodes;
    float Rms;
};
struct FilterConstants {
    unsigned Nodes, Period, Stride, Start, Count;
};
}
std::vector<double> GaussianProfile(Gpu &gpu, unsigned nodes, double spacing, double rms, double correlation, uint64_t seed) {
    if (nodes < 2 || nodes > (1u << 24) || !std::isfinite(spacing) || spacing <= 0 || !std::isfinite(rms) || rms < 0 ||
        !std::isfinite(correlation) || correlation < spacing || !std::isfinite(float(spacing)) || float(spacing) <= 0 ||
        !std::isfinite(float(correlation)) || !std::isfinite(float(rms))) throw std::invalid_argument("Invalid Gaussian profile parameters");
    const bool circular = std::has_single_bit(nodes);
    const unsigned size = circular ? nodes : std::bit_ceil(2 * nodes - 1);
    if (size > (1u << 24)) throw std::invalid_argument("Exact profile convolution exceeds GPU FFT capacity");
    const double scale = std::sqrt(2 / std::sqrt(std::numbers::pi) * (nodes - 1) * spacing / nodes / correlation);
    const ProfileConstants c{seed, nodes, size, float(spacing), float(correlation), float(rms)};
    const auto input = CreateBuffer(gpu, size_t(size) * 4 * sizeof(float)), product = CreateBuffer(gpu, size_t(size) * 2 * sizeof(float));
    const auto forward = CreateFftGpu(gpu, size, 2), inverse = CreateFftGpu(gpu, size);
    const auto generate = CreateKernel(gpu, "RoughGaussianInput"), multiply = CreateKernel(gpu, "FftMultiply");
    BeginGpu(gpu);
    const auto constants = BatchUpload(gpu, c);
    const std::array inputs{GpuBinding{constants, 0}, GpuBinding{input, 1}};
    DispatchGpu(gpu, generate, inputs, {size, 1, 1}, {256, 1, 1});
    EncodeFftGpu(gpu, forward, input);
    const auto dimensions = BatchUpload(gpu, size);
    const std::array products{GpuBinding{dimensions, 0}, GpuBinding{forward.Output, 1}, GpuBinding{product, 2}};
    DispatchGpu(gpu, multiply, products, {size, 1, 1}, {256, 1, 1});
    EncodeFftGpu(gpu, inverse, product, true);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto values = BufferSpan<std::array<float, 2>>(inverse.Output);
    std::vector<double> result(nodes);
    for (unsigned n = 0; n < nodes; ++n) {
        // Fold the zero-padded linear convolution modulo the requested period.
        result[n] = (double(values[n][0]) + (!circular && n + nodes < 2 * nodes - 1 ? double(values[n + nodes][0]) : 0.)) * scale;
        if (!std::isfinite(result[n])) throw std::runtime_error("Gaussian profile became nonfinite");
    }
    return result;
}
std::vector<double> GaussianSurface(Gpu &gpu, unsigned nx, unsigned ny, double dx, double dy, double rms, double cx, double cy, uint64_t seed) {
    if (nx < 2 || ny < 2 || nx > INT32_MAX || ny > INT32_MAX || uint64_t(nx) * ny > UINT32_MAX || !std::isfinite(rms) || rms < 0 || !std::isfinite(float(rms)))
        throw std::invalid_argument("Invalid Gaussian surface dimensions or amplitude");
    for (const auto axis : {std::array{dx, cx}, std::array{dy, cy}})
        if (!std::isfinite(axis[0]) || axis[0] <= 0 || !std::isfinite(axis[1]) || axis[1] < axis[0]) throw std::invalid_argument("Invalid Gaussian surface spacing or correlation");
    const unsigned nodes = nx * ny;
    const auto white = CreateBuffer(gpu, size_t(nodes) * sizeof(float)), scratch = CreateBuffer(gpu, size_t(nodes) * sizeof(float)), output = CreateBuffer(gpu, size_t(nodes) * sizeof(float));
    const auto generate = CreateKernel(gpu, "RoughWhiteNoise"), filter = CreateKernel(gpu, "RoughGaussianAxis");
    BeginGpu(gpu);
    const auto constants = BatchUpload(gpu, WhiteConstants{seed, nodes, float(rms)});
    const std::array bindings{GpuBinding{constants, 0}, GpuBinding{white, 1}};
    DispatchGpu(gpu, generate, bindings, {nodes, 1, 1}, {256, 1, 1});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    for (unsigned axis = 0; axis < 2; ++axis) {
        const unsigned period = axis ? ny : nx, stride = axis ? nx : 1;
        const double spacing = axis ? dy : dx, correlation = axis ? cy : cx, center = .5 * (period - 1);
        // exp(-2 * 8^2) is below the FP32 subnormal range.
        const double radius = 8 * correlation / spacing;
        const unsigned first = unsigned(std::max(0., std::ceil(center - radius))), last = unsigned(std::min(double(period - 1), std::floor(center + radius)));
        const double scale = std::sqrt(2 / std::sqrt(std::numbers::pi) * spacing / correlation);
        std::vector<float> coefficients(last - first + 1);
        for (unsigned k = first; k <= last; ++k) coefficients[k - first] = float(scale * std::exp(-2 * std::pow((k - center) * spacing / correlation, 2)));
        const auto taps = Upload<float>(gpu, coefficients);
        BeginGpu(gpu);
        const auto settings = BatchUpload(gpu, FilterConstants{nodes, period, stride, first, unsigned(coefficients.size())});
        const std::array pass{GpuBinding{settings, 0}, GpuBinding{axis ? scratch : white, 1}, GpuBinding{taps, 2}, GpuBinding{axis ? output : scratch, 3}};
        DispatchGpu(gpu, filter, pass, {nodes, 1, 1}, {256, 1, 1});
        SubmitGpu(gpu);
        WaitGpu(gpu);
    }
    const auto values = BufferSpan<float>(output);
    std::vector<double> result(nodes);
    for (unsigned n = 0; n < nodes; ++n) {
        result[n] = values[n];
        if (!std::isfinite(result[n])) throw std::runtime_error("Gaussian surface became nonfinite");
    }
    return result;
}
}
