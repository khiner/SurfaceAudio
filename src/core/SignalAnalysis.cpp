#define ACCELERATE_NEW_LAPACK
#include "core/SignalAnalysis.h"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace surface_audio {
namespace {
using Complex = std::complex<double>;
auto *Lapack(Complex *p) { return reinterpret_cast<__LAPACK_double_complex *>(p); }
bool Finite(Complex z) { return std::isfinite(z.real()) && std::isfinite(z.imag()); }
void CheckLapack(int info) {
    if (info) throw std::runtime_error("Signal analysis factorization failed: " + std::to_string(info));
}
std::vector<Complex> LeastSquares(std::vector<Complex> a, std::vector<Complex> b, int rows, int columns, int right_sides) {
    const char trans = 'N';
    int info{}, workspace = -1;
    Complex query;
    zgels_(&trans, &rows, &columns, &right_sides, Lapack(a.data()), &rows, Lapack(b.data()), &rows, Lapack(&query), &workspace, &info);
    CheckLapack(info);
    workspace = int(query.real());
    std::vector<Complex> work(workspace);
    zgels_(&trans, &rows, &columns, &right_sides, Lapack(a.data()), &rows, Lapack(b.data()), &rows, Lapack(work.data()), &workspace, &info);
    CheckLapack(info);
    std::vector<Complex> result(size_t(columns) * right_sides);
    for (int c = 0; c < right_sides; ++c) std::copy_n(b.data() + size_t(c) * rows, columns, result.data() + size_t(c) * columns);
    if (!std::ranges::all_of(result, Finite)) throw std::runtime_error("Nonfinite signal analysis solution");
    return result;
}
}

void FourierTransform(std::span<Complex> signal, bool inverse) {
    if (!std::has_single_bit(signal.size()) || signal.size() > (1u << 27)) throw std::invalid_argument("FFT requires a bounded power-of-two length");
    if (!std::ranges::all_of(signal, Finite)) throw std::invalid_argument("FFT requires finite samples");
    if (signal.size() == 1) return;
    const auto bits = std::countr_zero(signal.size());
    const std::unique_ptr<std::remove_pointer_t<FFTSetupD>, decltype(&vDSP_destroy_fftsetupD)> setup(vDSP_create_fftsetupD(bits, kFFTRadix2), vDSP_destroy_fftsetupD);
    if (!setup) throw std::runtime_error("Cannot allocate FFT setup");
    std::vector<double> real(signal.size()), imaginary(signal.size());
    DSPDoubleSplitComplex split{real.data(), imaginary.data()};
    vDSP_ctozD(reinterpret_cast<const DSPDoubleComplex *>(signal.data()), 2, &split, 1, signal.size());
    vDSP_fft_zipD(setup.get(), &split, 1, bits, inverse ? FFT_INVERSE : FFT_FORWARD);
    if (inverse) {
        const double scale = 1. / signal.size();
        vDSP_vsmulD(real.data(), 1, &scale, real.data(), 1, signal.size());
        vDSP_vsmulD(imaginary.data(), 1, &scale, imaginary.data(), 1, signal.size());
    }
    vDSP_ztocD(&split, 1, reinterpret_cast<DSPDoubleComplex *>(signal.data()), 2, signal.size());
}

std::vector<Complex> FourierTransform(std::span<const double> signal, uint32_t size) {
    if (!std::has_single_bit(size) || size > (1u << 27) || signal.size() > size) throw std::invalid_argument("Invalid FFT length");
    std::vector<Complex> result(size);
    std::ranges::copy(signal, result.begin());
    FourierTransform(result);
    return result;
}

LinearPrediction FitLinearPrediction(std::span<const double> signal, uint32_t order) {
    if (order >= signal.size() || !std::ranges::all_of(signal, [](double x) { return std::isfinite(x); })) throw std::invalid_argument("Invalid linear prediction samples or order");
    std::vector<double> correlation(size_t(order) + 1), denominator(size_t(order) + 1), previous(size_t(order) + 1);
    for (uint32_t lag = 0; lag <= order; ++lag) vDSP_dotprD(signal.data(), 1, signal.data() + lag, 1, &correlation[lag], signal.size() - lag);
    if (!std::ranges::all_of(correlation, [](double x) { return std::isfinite(x); })) throw std::overflow_error("Linear prediction correlation overflows");
    denominator[0] = 1;
    double error = correlation[0];
    for (uint32_t degree = 1; degree <= order && error > correlation[0] * std::numeric_limits<double>::epsilon(); ++degree) {
        double residual = correlation[degree];
        for (uint32_t j = 1; j < degree; ++j) residual += denominator[j] * correlation[degree - j];
        const double reflection = -residual / error;
        if (!std::isfinite(reflection) || std::abs(reflection) >= 1) throw std::runtime_error("Linear prediction lost positive definiteness");
        std::copy_n(denominator.begin(), degree, previous.begin());
        for (uint32_t j = 1; j < degree; ++j) denominator[j] = previous[j] + reflection * previous[degree - j];
        denominator[degree] = reflection;
        error *= 1 - reflection * reflection;
    }
    return {std::move(denominator), error};
}

void FitDampedGains(std::span<const Complex> signal, std::span<DampedSinusoid> modes) {
    if (modes.empty() || signal.size() < modes.size() || signal.size() > INT32_MAX || !std::ranges::all_of(signal, Finite)) throw std::invalid_argument("Invalid damped sinusoid fit dimensions or samples");
    std::vector<Complex> matrix(signal.size() * modes.size());
    for (size_t mode = 0; mode < modes.size(); ++mode) {
        if (!Finite(modes[mode].Pole)) throw std::invalid_argument("Nonfinite damped sinusoid pole");
        Complex value = 1;
        for (size_t row = 0; row < signal.size(); ++row, value *= modes[mode].Pole) matrix[mode * signal.size() + row] = value;
    }
    if (!std::ranges::all_of(matrix, Finite)) throw std::invalid_argument("Damped sinusoid basis overflows");
    const auto gain = LeastSquares(std::move(matrix), {signal.begin(), signal.end()}, int(signal.size()), int(modes.size()), 1);
    for (size_t mode = 0; mode < modes.size(); ++mode) modes[mode].Gain = gain[mode];
}

std::vector<DampedSinusoid> EstimateEsprit(std::span<const Complex> signal, uint32_t poles, uint32_t requested_rows) {
    const uint32_t row_count = requested_rows ? requested_rows : uint32_t(signal.size() / 2);
    if (signal.size() > INT32_MAX || !poles || row_count <= poles || row_count > signal.size() || signal.size() - row_count + 1 < poles || !std::ranges::all_of(signal, Finite)) throw std::invalid_argument("Invalid ESPRIT dimensions or samples");
    const int rows = int(row_count), columns = int(signal.size() - row_count + 1), rank = std::min(rows, columns), one = 1;
    std::vector<Complex> hankel(size_t(rows) * columns), u(size_t(rows) * rank);
    std::vector<double> singular(rank), real_work(size_t(5) * rank);
    for (int column = 0; column < columns; ++column)
        for (int row = 0; row < rows; ++row) hankel[size_t(column) * rows + row] = signal[row + column];
    const char thin = 'S', none = 'N';
    int info{}, workspace = -1;
    Complex query, unused;
    zgesvd_(&thin, &none, &rows, &columns, Lapack(hankel.data()), &rows, singular.data(), Lapack(u.data()), &rows, Lapack(&unused), &one, Lapack(&query), &workspace, real_work.data(), &info);
    CheckLapack(info);
    workspace = int(query.real());
    std::vector<Complex> work(workspace);
    zgesvd_(&thin, &none, &rows, &columns, Lapack(hankel.data()), &rows, singular.data(), Lapack(u.data()), &rows, Lapack(&unused), &one, Lapack(work.data()), &workspace, real_work.data(), &info);
    CheckLapack(info);
    if (!(singular[0] > 0)) throw std::invalid_argument("ESPRIT requires nonzero signal energy");
    std::vector<Complex> upper(size_t(rows - 1) * poles), lower(upper.size());
    for (uint32_t column = 0; column < poles; ++column) {
        std::copy_n(u.data() + size_t(column) * rows, rows - 1, upper.data() + size_t(column) * (rows - 1));
        std::copy_n(u.data() + size_t(column) * rows + 1, rows - 1, lower.data() + size_t(column) * (rows - 1));
    }
    auto shift = LeastSquares(std::move(upper), std::move(lower), rows - 1, int(poles), int(poles));
    const int count = int(poles);
    std::vector<Complex> eigenvalues(poles);
    real_work.resize(size_t(2) * poles);
    workspace = -1;
    zgeev_(&none, &none, &count, Lapack(shift.data()), &count, Lapack(eigenvalues.data()), Lapack(&unused), &one, Lapack(&unused), &one, Lapack(&query), &workspace, real_work.data(), &info);
    CheckLapack(info);
    workspace = int(query.real());
    work.resize(workspace);
    zgeev_(&none, &none, &count, Lapack(shift.data()), &count, Lapack(eigenvalues.data()), Lapack(&unused), &one, Lapack(&unused), &one, Lapack(work.data()), &workspace, real_work.data(), &info);
    CheckLapack(info);
    std::vector<DampedSinusoid> result;
    result.reserve(poles);
    for (const auto pole : eigenvalues) result.push_back({pole, {}});
    FitDampedGains(signal, result);
    return result;
}
}
