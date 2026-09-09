#include "agarwal/SpatialConvolution.h"
#include "core/Random.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string_view>

using namespace surface_audio;
using namespace surface_audio::agarwal;

static void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

struct Fixture {
    uint32_t Taps;
    std::vector<double> Frequency0, Frequency1, ObjectFrequency, Amplitude0, Amplitude1, ObjectAmplitude;
    ImpulseResponses Responses() const { return {44100, .37, Taps, {Frequency0, Amplitude0}, {Frequency1, Amplitude1}, {ObjectFrequency, ObjectAmplitude}}; }
};

static Fixture MakeFixture(uint32_t taps, uint32_t surface_modes, uint32_t object_modes) {
    Fixture fixture{taps, {}, {}, {}, {}, {}, {}};
    for (uint32_t mode = 0; mode < surface_modes; ++mode) {
        fixture.Frequency0.push_back(91 + 147 * mode);
        fixture.Frequency1.push_back(207 + 183 * mode);
        for (uint32_t lag = 0; lag < taps; ++lag) {
            const double decay = std::exp(-double(lag) / (80 + 17 * mode));
            fixture.Amplitude0.push_back(.023 * decay * (1 + .2 * std::sin(.27 * lag + mode)));
            fixture.Amplitude1.push_back(.013 * decay * (1 + .3 * std::cos(.19 * lag - mode)));
        }
    }
    for (uint32_t mode = 0; mode < object_modes; ++mode) {
        fixture.ObjectFrequency.push_back(613 + 319 * mode);
        for (uint32_t lag = 0; lag < taps; ++lag) fixture.ObjectAmplitude.push_back(.07 * std::exp(-double(lag) / (90 + 13 * mode)) * (1 + .15 * std::sin(.33 * lag)));
    }
    return fixture;
}

static std::vector<double> Oracle(std::span<const float> force, std::span<const float> morph, const ImpulseResponses &responses, double gain) {
    std::vector<double> result(force.size() + responses.TapCount - 1);
    for (size_t frame = 0; frame < result.size(); ++frame) {
        const double location = morph[std::min(frame, morph.size() - 1)];
        const size_t first = frame >= force.size() ? frame - force.size() + 1 : 0, end = std::min<size_t>(responses.TapCount, frame + 1);
        for (size_t lag = first; lag < end; ++lag) {
            const double time = double(lag) / responses.SampleRate;
            double impulse = 0;
            for (size_t mode = 0; mode < responses.Surface0.Frequencies.size(); ++mode) {
                const size_t index = mode * responses.TapCount + lag;
                const double frequency = std::pow(responses.Surface0.Frequencies[mode], 1 - location) * std::pow(responses.Surface1.Frequencies[mode], location);
                const double amplitude = std::pow(responses.Surface0.Amplitudes[index], 1 - location) * std::pow(responses.Surface1.Amplitudes[index], location);
                impulse += amplitude * std::sin(2 * std::numbers::pi * frequency * time);
            }
            for (size_t mode = 0; mode < responses.Object.Frequencies.size(); ++mode) impulse += responses.ObjectGain * responses.Object.Amplitudes[mode * responses.TapCount + lag] * std::sin(2 * std::numbers::pi * responses.Object.Frequencies[mode] * time);
            result[frame] += gain * force[frame - lag] * impulse;
        }
    }
    return result;
}

static void CheckOracle(std::span<const float> actual, std::span<const double> expected, double maximum_error = 3e-6, double relative_error = 3.162277660168379e-5) {
    Require(actual.size() == expected.size(), "Spatial convolution includes the complete finite tail");
    double error = 0, energy = 0, maximum = 0;
    for (size_t frame = 0; frame < actual.size(); ++frame) {
        const double difference = actual[frame] - expected[frame];
        error += difference * difference;
        energy += expected[frame] * expected[frame];
        maximum = std::max(maximum, std::abs(difference));
    }
    std::cout << "Spatial direct oracle: max=" << maximum << " relative_RMS=" << std::sqrt(error / energy) << '\n';
    Require(maximum < maximum_error && error < energy * relative_error * relative_error, "Full spatial output-location convolution matches independent double direct sum");
}

static void TestSpatial(Gpu &gpu) {
    const auto fixture = MakeFixture(113, 3, 2);
    auto responses = fixture.Responses();
    std::vector<float> force(219), morph(force.size());
    auto random = MakeRandom(8123);
    for (size_t frame = 0; frame < force.size(); ++frame) {
        force[frame] = Uniform(random) - .5f;
        const unsigned phase = frame % 64;
        morph[frame] = phase <= 32 ? phase / 32.f : (64 - phase) / 32.f;
    }
    constexpr float gain = -.71f;
    const auto expected = Oracle(force, morph, responses, gain);
    const auto first = ConvolveSpatialGpu(gpu, force, morph, responses, 1, gain);
    CheckOracle(first, expected);
    for (uint32_t block : {17, 128, 1024}) Require(ConvolveSpatialGpu(gpu, force, morph, responses, block, gain) == first, "Spatial convolution is bit-identical across block partitions");
    const auto saved = responses.Surface1.Frequencies;
    responses.Surface1.Frequencies = responses.Surface0.Frequencies;
    const auto fixed_frequencies = Oracle(force, morph, responses, gain);
    double frequency_difference = 0;
    for (size_t frame = 0; frame < first.size(); ++frame) frequency_difference += std::abs(expected[frame] - fixed_frequencies[frame]);
    Require(frequency_difference > 1, "Oracle case distinguishes changing frequencies from the fixed-pole special case");
    responses.Surface1.Frequencies = saved;
    responses.ObjectGain = 0;
    const auto without_object = Oracle(force, morph, responses, gain);
    double object_difference = 0;
    for (size_t frame = 0; frame < first.size(); ++frame) object_difference += std::abs(expected[frame] - without_object[frame]);
    Require(object_difference > 1, "Oracle exercises the separately added fixed object response");
    const std::array impulse{.83f}, location{.43f};
    CheckOracle(ConvolveSpatialGpu(gpu, impulse, location, fixture.Responses(), 13), Oracle(impulse, location, fixture.Responses(), 1));
    auto long_fixture = MakeFixture(11025, 1, 1);
    long_fixture.Frequency0[0] = 9100;
    long_fixture.Frequency1[0] = 11500;
    long_fixture.ObjectFrequency[0] = 12400;
    for (uint32_t lag = 0; lag < long_fixture.Taps; ++lag) {
        const double decay = std::exp(-double(lag) / 8000);
        long_fixture.Amplitude0[lag] = .02 * decay * (1 + .2 * std::sin(.0027 * lag));
        long_fixture.Amplitude1[lag] = .03 * decay * (1 + .3 * std::cos(.0019 * lag));
        long_fixture.ObjectAmplitude[lag] = .01 * decay * (1 + .1 * std::sin(.0033 * lag));
    }
    // At 12.4 kHz and a quarter-second lag, float phase arguments reach 19,476
    // radians. Keep a separate 0.2% relative bound for this long-lag precision check.
    CheckOracle(ConvolveSpatialGpu(gpu, impulse, location, long_fixture.Responses(), 256), Oracle(impulse, location, long_fixture.Responses(), 1), 1e-4, .002);
    const auto single = MakeFixture(1, 2, 1);
    const auto zero = ConvolveSpatialGpu(gpu, force, morph, single.Responses(), 37);
    Require(std::ranges::all_of(zero, [](float sample) { return sample == 0; }), "A one-tap sine response has exactly zero output");
    bool rejected = false;
    morph[19] = 1.01f;
    try {
        ConvolveSpatialGpu(gpu, force, morph, fixture.Responses());
    } catch (const std::invalid_argument &) { rejected = true; }
    Require(rejected, "Spatial convolution rejects invalid output locations");
}

static void TestEnvelopeCollapse(Gpu &gpu) {
    const auto original = MakeFixture(97, 3, 2);
    auto first = original, second = original, collapsed = original;
    for (uint32_t mode = 0; mode < 3; ++mode)
        for (uint32_t lag = 0; lag < original.Taps; ++lag) {
            const size_t index = size_t(mode) * original.Taps + lag;
            const double a = .3 + .2 * mode, b = (1.1 - .1 * mode) * std::exp(-double(lag) / (11 + 9 * mode));
            first.Amplitude0[index] *= a;
            first.Amplitude1[index] *= a;
            second.Amplitude0[index] *= b;
            second.Amplitude1[index] *= b;
            collapsed.Amplitude0[index] *= a + b;
            collapsed.Amplitude1[index] *= a + b;
        }
    for (uint32_t mode = 0; mode < 2; ++mode)
        for (uint32_t lag = 0; lag < original.Taps; ++lag) {
            const size_t index = size_t(mode) * original.Taps + lag;
            const double a = .7 + .1 * mode, b = .4 * std::exp(-double(lag) / (17 + 8 * mode));
            first.ObjectAmplitude[index] *= a;
            second.ObjectAmplitude[index] *= b;
            collapsed.ObjectAmplitude[index] *= a + b;
        }
    std::vector<float> force(137), morph(force.size());
    auto random = MakeRandom(9182);
    for (size_t frame = 0; frame < force.size(); ++frame) {
        force[frame] = Uniform(random) - .5f;
        const unsigned phase = frame % 48;
        morph[frame] = phase <= 24 ? phase / 24.f : (48 - phase) / 24.f;
    }
    const auto first_oracle = Oracle(force, morph, first.Responses(), 1), second_oracle = Oracle(force, morph, second.Responses(), 1);
    const auto collapsed_oracle = Oracle(force, morph, collapsed.Responses(), 1);
    std::vector<double> sum(first_oracle.size());
    for (size_t frame = 0; frame < sum.size(); ++frame) {
        sum[frame] = first_oracle[frame] + second_oracle[frame];
        Require(std::abs(sum[frame] - collapsed_oracle[frame]) < 1e-12, "Common lag factors collapse exactly under logarithmic spatial interpolation");
    }
    const auto rendered = ConvolveSpatialGpu(gpu, force, morph, collapsed.Responses(), 19);
    CheckOracle(rendered, sum);
    const auto first_gpu = ConvolveSpatialGpu(gpu, force, morph, first.Responses(), 13), second_gpu = ConvolveSpatialGpu(gpu, force, morph, second.Responses(), 31);
    std::vector<double> gpu_sum(rendered.size());
    for (size_t frame = 0; frame < gpu_sum.size(); ++frame) gpu_sum[frame] = double(first_gpu[frame]) + second_gpu[frame];
    CheckOracle(rendered, gpu_sum);
    // Independently varying each branch's endpoint ratio breaks the identity.
    const double separate = std::sqrt(1. * 4.) + std::sqrt(4. * 1.), merged = std::sqrt((1. + 4.) * (4. + 1.));
    Require(std::abs(separate - merged) > .9, "Independent branch endpoint weights cannot be collapsed as one mode");
}

static void Benchmark(Gpu &gpu) {
    constexpr uint32_t taps = 11025, frames = 65700, modes = 50, block = 256;
    auto fixture = MakeFixture(taps, modes, 0);
    // Long positive envelopes avoid denormals while retaining nonexponential lag dependence.
    for (uint32_t mode = 0; mode < modes; ++mode)
        for (uint32_t lag = 0; lag < taps; ++lag) {
            fixture.Amplitude0[size_t(mode) * taps + lag] = .002 * std::exp(-double(lag) / 8000) * (1 + .2 * std::sin(.0027 * lag + mode));
            fixture.Amplitude1[size_t(mode) * taps + lag] = .003 * std::exp(-double(lag) / 7000) * (1 + .3 * std::cos(.0019 * lag - mode));
        }
    std::vector<float> force(frames), morph(frames);
    auto random = MakeRandom(891);
    for (uint32_t frame = 0; frame < frames; ++frame) {
        force[frame] = Uniform(random) - .5f;
        morph[frame] = .5f + .5f * std::sin(.0003f * frame);
    }
    const auto begin = std::chrono::steady_clock::now();
    const auto output = ConvolveSpatialGpu(gpu, force, morph, fixture.Responses(), block);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    double energy = 0;
    for (float value : output) energy += double(value) * value;
    std::cout << "Spatial benchmark device=\"" << DeviceName(gpu) << "\" surface_mode_pairs=" << modes << " object_modes=0 taps=" << taps << " input_frames=" << frames << " output_frames=" << output.size() << " block_frames=" << block << " response_tile_bytes=" << size_t(block) * taps * sizeof(float) << " wall_seconds=" << seconds << " audio_seconds=" << double(frames) / 44100 << " output_RMS=" << std::sqrt(energy / output.size()) << '\n';
}

int main(int argc, char **argv) {
    try {
        auto gpu = CreateGpu();
        if (argc == 2 && std::string_view(argv[1]) == "--benchmark") Benchmark(gpu);
        else {
            TestSpatial(gpu);
            TestEnvelopeCollapse(gpu);
            std::cout << "Agarwal spatial convolution passed\n";
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
