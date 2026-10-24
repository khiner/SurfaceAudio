#include "Bouncing.h"
#include <algorithm>
#include <chrono>
#include <cmath>

namespace surface_audio::rough {
CorrelatedPlate MakeCorrelatedPlate(double correlation, double dt, double cutoff) {
    if (!std::isfinite(correlation) || correlation <= 0 || !std::isfinite(dt) || dt <= 0 ||
        !std::isfinite(cutoff) || cutoff <= 0 || dt * std::max({1., 2 / correlation, cutoff}) > .2)
        throw std::invalid_argument("Invalid correlated plate filter parameters");
    const double damping = 1 / correlation, d = cutoff * cutoff + 2 * damping * cutoff + 1;
    const double height_variance = cutoff * (cutoff + 2 * damping) / d;
    const double noise = std::sqrt(4 * damping / (dt * height_variance));
    const auto derivative = [=](std::array<double, 3> q, double input) {
        return std::array{q[1], -q[0] - 2 * damping * q[1] + input, cutoff * (q[0] - q[2])};
    };
    const auto add = [](auto q, auto rate, double step) {
        for (unsigned k = 0; k < 3; ++k) q[k] += step * rate[k];
        return q;
    };
    std::array<double, 12> transition;
    for (unsigned column = 0; column < 4; ++column) {
        const std::array q{double(column == 0), double(column == 1), double(column == 2)};
        const double input = column == 3 ? noise : 0;
        const auto a = derivative(q, input), b = derivative(add(q, a, dt / 2), input);
        const auto c = derivative(add(q, b, dt / 2), input), e = derivative(add(q, c, dt), input);
        for (unsigned k = 0; k < 3; ++k) transition[4 * k + column] = q[k] + dt / 6 * (a[k] + 2 * b[k] + 2 * c[k] + e[k]);
    }
    const double velocity = std::sqrt(cutoff / (cutoff + 2 * damping));
    const double acceleration = std::sqrt(cutoff * (2 * damping * cutoff + 1) / (cutoff + 2 * damping));
    return {transition, cutoff, velocity, acceleration, velocity * velocity / acceleration};
}
BounceSimulationResult SimulateBouncing(Gpu &gpu, BounceSimulation s) {
    if (!std::isfinite(s.Restitution) || s.Restitution < 0 || s.Restitution >= 1 || !std::isfinite(s.Acceleration) || s.Acceleration <= 0 ||
        !std::isfinite(s.StickFactor) || s.StickFactor < 1 || !s.Streams || s.Streams > 65536 || !s.Flights || s.Discard >= s.Flights ||
        !s.MaxSteps || s.TraceSamples > s.MaxSteps) throw std::invalid_argument("Invalid bouncing simulation parameters");
    const auto filter = MakeCorrelatedPlate(s.CorrelationTime, s.TimeStep);
    BounceConstants c{.Transition = {}, .Dt = float(s.TimeStep), .Gravity = float(filter.VelocityDeviation * filter.VelocityDeviation / s.Acceleration),
                .Restitution = float(s.Restitution), .StickVelocity = float(s.StickFactor * s.TimeStep * filter.VelocityDeviation * filter.VelocityDeviation / s.Acceleration),
                .VelocityScale = float(1 / filter.VelocityDeviation), .Cutoff = float(filter.Cutoff), .Streams = s.Streams,
                .Flights = s.Flights, .Discard = s.Discard, .Steps = 0, .TraceSamples = s.TraceSamples, .RelativeSticking = s.RelativeSticking, .Start = 0, .Seed = s.Seed};
    std::ranges::copy(filter.Transition, c.Transition);
    if (!(c.Dt > 0) || !(c.Gravity > 0) || !std::isfinite(c.Gravity) || !std::isfinite(c.StickVelocity) ||
        !std::ranges::all_of(c.Transition, [](float x) { return std::isfinite(x); }))
        throw std::invalid_argument("Bouncing parameters exceed GPU precision");
    const auto statistics = CreateBuffer(gpu, size_t(s.Streams) * sizeof(BounceStatistics));
    const auto state = CreateBuffer(gpu, size_t(s.Streams) * 128);
    const auto trace = CreateBuffer(gpu, std::max(size_t(16), size_t(s.TraceSamples) * 16));
    const auto kernel = CreateKernel(gpu, "CorrelatedBouncing");
    const auto values = BufferSpan<BounceStatistics>(statistics);
    const auto begin = std::chrono::steady_clock::now();
    for (uint64_t start = 0; start < s.MaxSteps;) {
        c.Start = start;
        c.Steps = unsigned(std::min(uint64_t(32768), s.MaxSteps - start));
        BeginGpu(gpu);
        const auto constants = BatchUpload(gpu, c);
        const std::array bindings{GpuBinding{constants, 0}, GpuBinding{state, 1}, GpuBinding{statistics, 2}, GpuBinding{trace, 3}};
        DispatchGpu(gpu, kernel, bindings, {s.Streams, 1, 1}, {64, 1, 1});
        SubmitGpu(gpu);
        WaitGpu(gpu);
        start += c.Steps;
        if (start >= s.TraceSamples && std::ranges::all_of(values, [&](const auto &v) { return v.Flights >= s.Flights - s.Discard; })) break;
    }
    const auto audio = BufferSpan<std::array<float, 4>>(trace).first(s.TraceSamples);
    return {{values.begin(), values.end()}, {audio.begin(), audio.end()}, std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count()};
}
}
