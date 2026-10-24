#include "GeometricContact.h"
#include "ContactGpuTypes.h"
#include "core/GpuFft.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>

namespace surface_audio::rough {
namespace {
constexpr double Pi = std::numbers::pi;
double Integral(double low, double high, double exponent) {
    return std::abs(exponent) < 1e-12 ? std::log(high / low) : std::pow(low, exponent) * std::expm1(exponent * std::log(high / low)) / exponent;
}
void Validate(SurfaceSpectrum s) {
    if (!std::isfinite(s.Rms) || s.Rms <= 0 || !std::isfinite(s.Low) || s.Low <= 0 || !std::isfinite(s.High) || s.High < s.Low ||
        !std::isfinite(s.Minimum) || s.Minimum < 0 || s.Minimum >= s.Low || !std::isfinite(s.Exponent) || s.Exponent <= 0)
        throw std::invalid_argument("Invalid self-affine surface spectrum");
}
void Patches(double n) {
    if (!std::isfinite(n) || n < 1) throw std::invalid_argument("Extreme-value patch count must be at least one");
}
}
std::array<double, 3> SurfaceMoments(SurfaceSpectrum s) {
    Validate(s);
    std::array<double, 3> result;
    for (unsigned i = 0; i < 3; ++i)
        result[i] = 2 * Pi * ((std::pow(s.Low, i + 2) - std::pow(s.Minimum, i + 2)) / (i + 2) +
                             std::pow(s.Low, s.Exponent) * Integral(s.Low, s.High, i + 2 - s.Exponent));
    const double scale = s.Rms * s.Rms / result[0];
    for (auto &moment : result) moment *= scale;
    return result;
}
std::vector<float> SelfAffineSurface(Gpu &gpu, unsigned nx, unsigned ny, double spacing, SurfaceSpectrum s, uint64_t seed) {
    Validate(s);
    if (!(std::has_single_bit(nx) || (nx % 5 == 0 && std::has_single_bit(nx / 5))) || !std::has_single_bit(ny) || nx < 4 || ny < 4 || uint64_t(nx) * ny > (1u << 28) ||
        !std::isfinite(spacing) || spacing <= 0 || s.High >= .5 / spacing) throw std::invalid_argument("Invalid self-affine surface grid");
    const double integral = 2 * Pi * ((s.Low * s.Low - s.Minimum * s.Minimum) / 2 + std::pow(s.Low, s.Exponent) * Integral(s.Low, s.High, 2 - s.Exponent));
    const SurfaceSpectrumConstants c{nx, ny, float(spacing), float(s.Low), float(s.High), float(s.Exponent), float(s.Minimum),
                              float(s.Rms * std::sqrt(double(nx) * ny / (spacing * spacing * integral))), seed};
    if (!(c.Spacing > 0) || !(c.Low > 0) || !std::isfinite(c.High) || !std::isfinite(c.Exponent) || !(c.Scale > 0) || !std::isfinite(c.Scale))
        throw std::invalid_argument("Surface spectrum exceeds GPU precision");
    const auto spectrum = CreateBuffer(gpu, size_t(nx) * ny * 8), transposed = CreateBuffer(gpu, spectrum.Size);
    const bool radix_five = !std::has_single_bit(nx);
    const auto rows = CreateFftGpu(gpu, radix_five ? nx / 5 : nx, radix_five ? 5 * ny : ny), columns = CreateFftGpu(gpu, ny, nx);
    const auto generate = CreateKernel(gpu, "SelfAffineSpectrum"), transpose = CreateKernel(gpu, "SurfaceTranspose");
    const auto pack = CreateKernel(gpu, "SurfaceRadixFiveInput");
    BeginGpu(gpu);
    const auto constants = BatchUpload(gpu, c);
    const std::array input{GpuBinding{constants, 0}, GpuBinding{spectrum, 1}};
    DispatchGpu(gpu, generate, input, {nx, ny, 1}, {16, 16, 1});
    if (radix_five) {
        const std::array packing{GpuBinding{constants, 0}, GpuBinding{spectrum, 1}, GpuBinding{transposed, 2}};
        DispatchGpu(gpu, pack, packing, {nx, ny, 1}, {16, 16, 1});
    }
    EncodeFftGpu(gpu, rows, radix_five ? transposed : spectrum, true);
    const std::array exchange{GpuBinding{constants, 0}, GpuBinding{rows.Output, 1}, GpuBinding{transposed, 2}};
    DispatchGpu(gpu, transpose, exchange, {nx, ny, 1}, {16, 16, 1});
    EncodeFftGpu(gpu, columns, transposed, true);
    SubmitGpu(gpu);
    WaitGpu(gpu);
    const auto values = BufferSpan<std::array<float, 2>>(columns.Output);
    std::vector<float> result(size_t(nx) * ny);
    for (unsigned y = 0; y < ny; ++y)
        for (unsigned x = 0; x < nx; ++x) result[size_t(y) * nx + x] = values[size_t(x) * ny + y][0];
    return result;
}
std::vector<float> SlidingSeparation(Gpu &gpu, SurfaceView track, SurfaceView slider) {
    for (const auto s : {track, slider}) {
        if (!s.XNodes || !s.YNodes || s.Height.size() != uint64_t(s.XNodes) * s.YNodes || s.Height.size() > UINT32_MAX)
            throw std::invalid_argument("Invalid geometric-contact surface dimensions");
        for (float height : s.Height)
            if (!std::isfinite(height)) throw std::invalid_argument("Nonfinite geometric-contact height");
    }
    if (track.XNodes < slider.XNodes || track.YNodes != slider.YNodes) throw std::invalid_argument("Slider requires complete track overlap");
    const unsigned count = track.XNodes - slider.XNodes + 1;
    const auto a = Upload<float>(gpu, track.Height), b = Upload<float>(gpu, slider.Height), result = CreateBuffer(gpu, size_t(count) * 4);
    const auto kernel = CreateKernel(gpu, "GeometricSeparation");
    for (unsigned start = 0; start < count; start += 256) {
        BeginGpu(gpu);
        const auto c = BatchUpload(gpu, SeparationConstants{track.XNodes, slider.XNodes, slider.YNodes, start});
        const std::array bindings{GpuBinding{c, 0}, GpuBinding{a, 1}, GpuBinding{b, 2}, GpuBinding{result, 3}};
        DispatchGroupsGpu(gpu, kernel, bindings, {std::min(256u, count - start), 1, 1}, {256, 1, 1});
        SubmitGpu(gpu);
        WaitGpu(gpu);
    }
    const auto values = BufferSpan<float>(result);
    return {values.begin(), values.end()};
}
SeparationStatistics ExtremeSeparation(double n) {
    Patches(n);
    constexpr unsigned samples = 8192;
    const double low = -10, high = std::sqrt(2 * std::log(n)) + 10, step = (high - low) / samples;
    std::array<double, 4> moments{};
    for (unsigned i = 0; i <= samples; ++i) {
        const double x = low + i * step, weight = (i == 0 || i == samples ? 1 : i % 2 ? 4 : 2) * step / 3;
        const double cdf = .5 * std::erfc(-x / std::sqrt(2.));
        double value = weight * (cdf == 0 ? 0 : std::exp(std::log(n) - .5 * x * x - .5 * std::log(2 * Pi) + (n - 1) * std::log(cdf)));
        for (auto &moment : moments) { moment += value; value *= x; }
    }
    const double mean = moments[1] / moments[0], variance = moments[2] / moments[0] - mean * mean;
    return {mean, std::sqrt(variance), (moments[3] / moments[0] - 3 * mean * variance - mean * mean * mean) / std::pow(variance, 1.5)};
}
SeparationStatistics CorrelatedSeparation(double n, double bandwidth) {
    Patches(n);
    if (!std::isfinite(bandwidth) || bandwidth <= 0 || bandwidth > 1) throw std::invalid_argument("Invalid surface bandwidth");
    const double ku = bandwidth < .5 ? 1.5 * -std::expm1(-1.8 * bandwidth) : .94, ka = bandwidth < .5 ? 7 * bandwidth : 4.05;
    if (ku * n <= 1 || ka * n <= 1) throw std::invalid_argument("Correlated extreme-value approximation requires more patches");
    const double scale = 1 / std::sqrt(2 * std::log(ka * n));
    return {std::sqrt(2 * std::log(ku * n)) + std::numbers::egamma * scale, Pi / std::sqrt(6.) * scale, 1.1395470994046487};
}
std::array<double, 3> SeparationSpectrum(double size, double ratio, double bandwidth) {
    if (!std::isfinite(size) || size <= 0 || !std::isfinite(ratio) || ratio <= 0 || ratio > 1 ||
        !std::isfinite(bandwidth) || bandwidth < 0 || bandwidth > 1) throw std::invalid_argument("Invalid separation spectrum parameters");
    return {(5.18e-5 / std::pow(ratio, 1.54) + .0584) * std::pow(size, .0892), .38 * bandwidth * bandwidth + .58,
            -.178 * bandwidth + 1.50 / std::pow(size, .0685)};
}
}
