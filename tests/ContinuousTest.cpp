#include "continuous/Continuous.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace surface_audio;
using namespace surface_audio::continuous;
namespace {
void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
void Navigation() {
    constexpr double turn = 2 * std::numbers::pi;
    for (unsigned anchor = 0; anchor < 3; ++anchor) {
        const auto weights = ActionWeights(anchor * turn / 3, 1);
        for (unsigned index = 0; index < 3; ++index) Require(std::abs(weights[index] - double(index == anchor)) < 1e-14, "Action anchor");
    }
    for (int index = -50; index <= 50; ++index) {
        const auto w = ActionWeights(index * .17, .7), center = ActionWeights(index * .17, 0);
        Require(std::abs(std::accumulate(w.begin(), w.end(), 0.) - 1) < 1e-14 && *std::min_element(w.begin(), w.end()) >= 0, "Convex disk weights");
        for (double x : center) Require(std::abs(x - 1. / 3) < 1e-14, "Disk center");
    }
    const auto edge0 = ActionWeights(1e-10, 1), edge1 = ActionWeights(turn - 1e-10, 1);
    for (unsigned index = 0; index < 3; ++index) Require(std::abs(edge0[index] - edge1[index]) < 1e-9, "Navigation wrap continuity");
}
void Distribution() {
    constexpr unsigned count = 150000;
    for (unsigned anchor = 0; anchor < 3; ++anchor) {
        const auto p = MakeParameters({.Angle = anchor * 2 * std::numbers::pi / 3, .CutoffHz = 0});
        auto s = MakeState(781);
        double sum = 0, square = 0, interval = 0, interval_square = 0;
        for (unsigned i = 0; i < count; ++i) {
            const auto e = NextEvent(p, s);
            sum += e.Amplitude;
            square += double(e.Amplitude) * e.Amplitude;
            interval += e.Interval;
            interval_square += double(e.Interval) * e.Interval;
        }
        const double mean = sum / count, variance = square / count - mean * mean;
        if (anchor < 2) Require(std::abs(mean) < .015 && std::abs(variance - 1) < .025, "Signed Gaussian friction amplitude");
        if (anchor == 0) Require(std::abs(interval / count - 1. / p.SampleRate) < 1e-11, "One rubbing impact per sample");
        if (anchor == 1) {
            const double dt_mean = interval / count, dt_var = interval_square / count - dt_mean * dt_mean;
            Require(std::abs(dt_mean - .01) < .0002 && std::abs(dt_var - .0001) < .000008, "Exponential scratch intervals");
        }
        if (anchor == 2) {
            const auto &a = p.Amplitude;
            const double expected = a.Sigma * a.Sigma * (1 + a.B1 * a.B1 - 2 * a.A1 * a.B1) / (1 - a.A1 * a.A1);
            Require(std::abs(mean - a.Mean) < .02 && std::abs(variance / expected - 1) < .07, "Rolling ARMA mean and variance");
        }
    }
    const auto mixed = MakeParameters({.Radius = 0, .CutoffHz = 0});
    const auto rolling = conan::MakeParameters({}, 44100);
    for (unsigned i = 1; i < conan::QuantileCount - 1; ++i) {
        const double x = mixed.Amplitude.Quantiles[i];
        const double probability = (2 * .5 * std::erfc(-x / std::sqrt(2.)) + .5 * std::erfc(-x / (rolling.Amplitude.Sigma * std::sqrt(2.)))) / 3;
        Require(std::abs(probability - conan::QuantileProbability(i)) < 1e-6, "Mix PDFs before inverse CDF; no quantile crossfade");
    }
}
void Pulses() {
    Parameters p{.Amplitude = {.Mean = -.7f}, .Interval = {.Mean = .01f}, .SampleRate = 44100, .DurationScale = .002f};
    auto s = MakeState(1);
    std::vector<float> output(400);
    Render(p, s, output);
    const double center = std::ceil(.5f * p.MaximumDuration * p.SampleRate), duration = double(p.DurationScale * p.SampleRate);
    for (unsigned sample = 0; sample < output.size(); ++sample) {
        const double time = sample - center;
        const double expected = std::abs(time) <= duration / 2 ? -.7 * .5 * (1 + std::cos(2 * std::numbers::pi * time / duration)) : 0;
        Require(std::abs(output[sample] - expected) < 3e-7, "Independent signed raised cosine with fixed lookahead");
    }
    auto rub = MakeParameters({.CutoffHz = 0});
    auto state = MakeState(82), split = state;
    std::vector<float> whole(2048), blocked(whole.size());
    Render(rub, state, whole);
    for (size_t begin = 0; begin < blocked.size();) {
        const auto size = std::min<size_t>(73, blocked.size() - begin);
        Render(rub, split, std::span(blocked).subspan(begin, size));
        begin += size;
    }
    Require(whole == blocked && std::memcmp(&state, &split, sizeof(State)) == 0, "CPU block partition invariance");
    auto random = MakeRandom(82, 1);
    const unsigned latency = unsigned(std::ceil(.5f * rub.MaximumDuration * rub.SampleRate));
    for (unsigned sample = latency; sample < whole.size(); ++sample) Require(std::abs(whole[sample] - Normal(random)) < 1e-7, "Rubbing is delayed white Gaussian noise, no rectification");
}
void LowpassAndValidation() {
    const auto p = MakeParameters({.Velocity = .5f, .CutoffHz = 1000});
    auto constant = p;
    constant.Amplitude = {.Mean = 1};
    auto state = MakeState(18);
    std::vector<float> samples(512);
    Render(constant, state, samples);
    const unsigned latency = unsigned(std::ceil(.5f * p.MaximumDuration * p.SampleRate));
    for (unsigned frame = latency; frame < samples.size(); ++frame) {
        const double expected = 1 - std::pow(double(p.LowpassPole), frame - latency + 1);
        Require(std::abs(samples[frame] - expected) < 1e-6, "Independent lowpass step response");
    }
    unsigned rejected = 0;
    for (const Controls c : {Controls{.Radius = 1.1}, Controls{.ScratchDensity = 0}, Controls{.FrictionSigma = -1}, Controls{.Velocity = -1}, Controls{.Size = 0}}) {
        try {
            MakeParameters(c);
        } catch (const std::invalid_argument &) { ++rejected; }
    }
    Require(rejected == 5, "Invalid continuous controls rejected");
}
void GpuStreaming() {
    auto gpu = CreateGpu();
    constexpr unsigned voices = 4, frames = 127;
    std::array<Parameters, voices> p;
    std::array<State, voices> states;
    for (unsigned voice = 0; voice < voices; ++voice) {
        p[voice] = MakeParameters({.Angle = voice * 2 * std::numbers::pi / 3, .Radius = voice == 3 ? .4 : 1, .Asymmetry = .5f});
        states[voice] = MakeState(18 + voice);
    }
    const auto source = CreateGpuSource(gpu, p, states, frames);
    double error = 0, energy = 0, maximum = 0;
    for (unsigned block = 0; block < 40; ++block) {
        if (block == 20) {
            for (auto &parameter : p) parameter = MakeParameters({.Angle = .8, .Radius = .8, .Asymmetry = .2f});
            std::ranges::copy(p, BufferSpan<Parameters>(source.Parameters).begin());
        }
        BeginGpu(gpu);
        EncodeSource(gpu, source);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        const auto output = BufferSpan<float>(source.Output);
        for (unsigned voice = 0; voice < voices; ++voice) {
            std::array<float, frames> cpu;
            Render(p[voice], states[voice], cpu);
            for (unsigned frame = 0; frame < frames; ++frame) {
                const double difference = output[voice * frames + frame] - cpu[frame];
                error += difference * difference;
                energy += double(cpu[frame]) * cpu[frame];
                maximum = std::max(maximum, std::abs(difference));
            }
            Require(BufferSpan<State>(source.States)[voice].Events == states[voice].Events, "GPU event count through controls transition");
        }
    }
    const double relative = std::sqrt(error / energy);
    std::cout << "Continuous streaming relative_l2=" << relative << " max=" << maximum << '\n';
    Require(std::isfinite(relative) && relative < .003 && maximum < .025, "CPU GPU continuous streaming parity");
}
}
int main() {
    try {
        Navigation();
        Distribution();
        Pulses();
        LowpassAndValidation();
        GpuStreaming();
        std::cout << "Continuous independent equations, statistics, navigation and streaming passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
