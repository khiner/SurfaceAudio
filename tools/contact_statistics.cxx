#include "rough/GeometricContact.h"
#include "rough/Bouncing.h"
#include "core/BinaryFile.h"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>

using namespace surface_audio;
using namespace surface_audio::rough;
namespace {
namespace fs = std::filesystem;
unsigned Number(const char *value) {
    const auto result = std::stoul(value);
    if (result > UINT32_MAX) throw std::invalid_argument("Integer argument exceeds range");
    return unsigned(result);
}
void Ponthus(const fs::path &output, unsigned nodes, unsigned draws, double exponent, double ratio) {
    if (!draws || draws > 1000 || nodes > 4096) throw std::invalid_argument("Invalid surface ensemble dimensions");
    constexpr double spacing = 1. / 12;
    const SurfaceSpectrum spectrum{1, ratio, 1, exponent, 1 / (nodes * spacing)};
    const auto moments = SurfaceMoments(spectrum);
    const double wavelength = .5 * std::sqrt(moments[0] / moments[2]), bandwidth = std::sqrt(1 - moments[1] * moments[1] / (moments[0] * moments[2]));
    const double size = nodes * spacing / wavelength, patches = 4 / std::numbers::pi * size * size;
    const auto independent = ExtremeSeparation(patches), correlated = CorrelatedSeparation(patches, bandwidth);
    std::vector<float> records;
    const auto start = std::chrono::steady_clock::now();
    for (unsigned draw = 0; draw < draws; ++draw) {
        auto gpu = CreateGpu();
        const auto track = SelfAffineSurface(gpu, 5 * nodes, nodes, spacing, spectrum, 2019 + 2 * draw);
        const auto slider = SelfAffineSurface(gpu, nodes, nodes, spacing, spectrum, 2020 + 2 * draw);
        const auto separation = SlidingSeparation(gpu, {track, 5 * nodes, nodes}, {slider, nodes, nodes});
        records.insert(records.end(), separation.begin(), separation.end());
        if (!draw) {
            WriteBinary<float>(output / "track.f32", track);
            WriteBinary<float>(output / "slider.f32", slider);
        }
    }
    WriteBinary<float>(output / "separation.f32", records);
    const auto empirical = SeparationSpectrum(size, ratio, bandwidth);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::ofstream out(output / "Run.json");
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << std::setprecision(15) << "{\"method\":\"ponthus\",\"seed\":2019,\"nodes\":" << nodes << ",\"draws\":" << draws << ",\"spacing\":" << spacing
        << ",\"exponent\":" << exponent << ",\"low_high_ratio\":" << ratio << ",\"wavelength\":" << wavelength << ",\"bandwidth\":" << bandwidth
        << ",\"size_over_wavelength\":" << size << ",\"patches\":" << patches << ",\"evt_mean\":" << independent.Mean
        << ",\"evt_deviation\":" << independent.Deviation << ",\"evt_skewness\":" << independent.Skewness
        << ",\"preumont_mean\":" << correlated.Mean << ",\"preumont_deviation\":" << correlated.Deviation
        << ",\"empirical_crossover\":" << empirical[0] << ",\"empirical_bandwidth\":" << empirical[1]
        << ",\"empirical_wavelength\":" << empirical[2] << ",\"seconds\":" << seconds << "}\n";
    std::cout << "Ponthus: " << draws << " surfaces, " << nodes << " nodes per side, " << seconds << " seconds\n";
}
void Zouabi(const fs::path &output, BounceSimulation settings) {
    auto gpu = CreateGpu();
    const auto result = SimulateBouncing(gpu, settings);
    WriteBinary<BounceStatistics>(output / "statistics.bin", result.Statistics);
    WriteBinary<std::array<float, 4>>(output / "trajectory.f32", result.Trace);
    const auto filter = MakeCorrelatedPlate(settings.CorrelationTime, settings.TimeStep);
    std::ofstream out(output / "Run.json");
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << std::setprecision(15) << "{\"method\":\"zouabi\",\"seed\":" << settings.Seed << ",\"natural_frequency\":1,\"cutoff\":10,\"restitution\":" << settings.Restitution << ",\"lambda\":" << settings.Acceleration
        << ",\"correlation_time\":" << settings.CorrelationTime << ",\"dt\":" << settings.TimeStep << ",\"streams\":" << settings.Streams
        << ",\"flights_per_stream\":" << settings.Flights << ",\"discard\":" << settings.Discard << ",\"max_steps\":" << settings.MaxSteps
        << ",\"relative_sticking\":" << (settings.RelativeSticking ? "true" : "false") << ",\"trace_samples\":" << settings.TraceSamples << ",\"velocity_deviation\":" << filter.VelocityDeviation
        << ",\"acceleration_deviation\":" << filter.AccelerationDeviation << ",\"bandwidth\":" << filter.Bandwidth << ",\"seconds\":" << result.Seconds << ",\"plate_transition\":[";
    for (unsigned i = 0; i < filter.Transition.size(); ++i) out << (i ? "," : "") << float(filter.Transition[i]);
    out << "]}\n";
    std::cout << "Zouabi: " << settings.Streams << " streams, " << result.Seconds << " seconds\n";
}
}
int main(int argc, char **argv) {
    try {
        if (argc < 3) throw std::invalid_argument("Usage: contactStatistics ponthus OUTPUT [nodes draws exponent ratio] | zouabi OUTPUT [lambda correlation streams flights trace_samples dt relative_sticking max_steps]");
        const std::string method = argv[1];
        const fs::path output = argv[2];
        fs::create_directories(output);
        if (method == "ponthus") Ponthus(output, argc > 3 ? Number(argv[3]) : 512, argc > 4 ? Number(argv[4]) : 15,
                                         argc > 5 ? std::stod(argv[5]) : 4, argc > 6 ? std::stod(argv[6]) : .46);
        else if (method == "zouabi") {
            const unsigned flights = argc > 6 ? Number(argv[6]) : 1100;
            const BounceSimulation s{.Acceleration = argc > 3 ? std::stod(argv[3]) : 5, .CorrelationTime = argc > 4 ? std::stod(argv[4]) : 1.11,
                                     .TimeStep = argc > 8 ? std::stod(argv[8]) : .005, .Streams = argc > 5 ? Number(argv[5]) : 1024, .Flights = flights, .Discard = flights / 10,
                                     .TraceSamples = argc > 7 ? Number(argv[7]) : 524288, .MaxSteps = argc > 10 ? std::stoull(argv[10]) : 100000000, .RelativeSticking = argc <= 9 || Number(argv[9]) != 0};
            Zouabi(output, s);
        } else throw std::invalid_argument("Unknown contact statistics method");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
