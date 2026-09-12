#include "agarwal/Impact.h"
#include "agarwal/ObjectResponse.h"
#include "core/AudioFile.h"
#include "core/BinaryFile.h"
#include "core/GpuConvolution.h"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
using namespace surface_audio;
using namespace surface_audio::agarwal;
namespace {
template<size_t D> void WriteGaussian(std::ostream &out, const Gaussian<D> &g) {
    out << "{\"count\":" << g.Count << ",\"mean\":[";
    for (size_t i = 0; i < D; ++i) out << (i ? "," : "") << g.Mean[i];
    out << "],\"factor\":[";
    for (size_t i = 0; i < D; ++i) {
        out << (i ? ",[" : "[");
        for (size_t j = 0; j < D; ++j) out << (j ? "," : "") << g.Factor[i][j];
        out << ']';
    }
    out << "]}";
}
void Distribution(const std::filesystem::path &input, const std::filesystem::path &output, uint32_t count, uint64_t seed) {
    if (!count || count > 1000000) throw std::invalid_argument("Invalid cohort size");
    const auto records = ReadBinary<ObjectResponseParameters>(input);
    const auto distribution = FitObjectResponseDistribution(records);
    auto random = MakeRandom(seed);
    ObjectResponseSampleStats stats{};
    std::vector<ObjectResponseParameters> samples;
    samples.reserve(count);
    for (uint32_t i = 0; i < count; ++i) samples.push_back(SampleObjectResponse(distribution, random, &stats));
    std::filesystem::create_directories(output);
    WriteBinary<ObjectResponseParameters>(output / "samples.f32", samples);
    std::ofstream metadata(output / "distribution.json");
    metadata << std::setprecision(17) << "{\"seed\":" << seed << ",\"mode_rejections\":" << stats.ModeRejections << ",\"noise_rejections\":" << stats.NoiseRejections << ",\"modes\":";
    WriteGaussian(metadata, distribution.Modes);
    metadata << ",\"noise\":[";
    for (uint32_t i = 0; i < 20; ++i) {
        if (i) metadata << ',';
        WriteGaussian(metadata, distribution.Noise[i]);
    }
    metadata << "]}\n";
    if (!metadata) throw std::runtime_error("Cannot write response distribution");
}
}
int main(int argc, char **argv) {
    try {
        const std::string_view command = argc > 1 ? argv[1] : "";
        if (argc == 6 && command == "distribution") {
            Distribution(argv[2], argv[3], uint32_t(std::stoul(argv[4])), std::stoull(argv[5]));
            return 0;
        }
        if (argc == 6 && command == "forces") {
            const std::filesystem::path input(argv[2]);
            const auto scraping = ReadBinary<float>(input / "scrape.f32"), elastic = ReadBinary<float>(input / "elastic.f32"), damping = ReadBinary<float>(input / "damping.f32");
            auto gpu = CreateGpu();
            const auto force = MixContactForcesGpu(gpu, scraping, elastic, damping, std::stod(argv[4]), std::stod(argv[5]));
            WriteWave(argv[3], 44100, 1, force);
            return 0;
        }
        const bool contact = command == "contact";
        if (argc != 11 + contact || (!contact && command != "impact")) throw std::invalid_argument("Usage: agarwalObjectReproduce impact response.wav output mass velocity kA kB scale limit combination | contact excitation.wav response.wav output mass velocity kA kB scale limit combination | distribution records.f32 output count seed | forces directory output.wav k lambda");
        const int offset = contact;
        const auto response = ReadWave(argv[2 + offset]);
        if (response.Channels != 1 || response.SampleRate != 44100) throw std::invalid_argument("Response requires mono 44100 Hz audio");
        const auto excitation = contact ? ReadWave(argv[2]) : WaveData{};
        if (contact && (excitation.Channels != 1 || excitation.SampleRate != response.SampleRate)) throw std::invalid_argument("Invalid contact excitation");
        const std::filesystem::path output(argv[3 + offset]);
        const ImpactSettings controls{.Mass = std::stod(argv[4 + offset]), .Velocity = std::stod(argv[5 + offset]), .StiffnessA = std::stod(argv[6 + offset]), .StiffnessB = std::stod(argv[7 + offset]), .Scale = std::stod(argv[8 + offset]), .ForceLimit = std::stod(argv[9 + offset]), .Combination = StiffnessCombination(std::stoul(argv[10 + offset]))};
        const double duration = ImpactDuration(controls);
        if (duration * response.SampleRate >= UINT32_MAX) throw std::invalid_argument("Impact duration too long");
        auto gpu = CreateGpu();
        const auto start = std::chrono::steady_clock::now();
        const auto force = contact ? RenderMicroImpactGpu(gpu, controls, response.SampleRate) : RenderImpactForcesGpu(gpu, std::span{&controls, 1}, uint32_t(std::ceil(duration * response.SampleRate)), response.SampleRate);
        auto impact = ConvolveFixedGpu(gpu, force, response.Samples);
        const auto audio = contact ? ConvolveFixedGpu(gpu, excitation.Samples, impact) : std::move(impact);
        const double milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::filesystem::create_directories(output);
        WriteWave(output / "synthesis.wav", response.SampleRate, 1, audio);
        WriteWave(output / "force.wav", response.SampleRate, 1, force);
        std::ofstream metadata(output / "render.json");
        metadata << std::setprecision(17) << "{\"duration_seconds\":" << duration << ",\"render_ms\":" << milliseconds << ",\"frames\":" << audio.size() << ",\"sampling\":\"sample interval averages\",\"device\":\"" << DeviceName(gpu) << "\"}\n";
        if (!metadata) throw std::runtime_error("Cannot write impact metadata");
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
