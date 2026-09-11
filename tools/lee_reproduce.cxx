#include "core/AudioFile.h"
#include "lee/Lee.h"
#include "lee/LeeInternal.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace surface_audio;
namespace {
using Clock = std::chrono::steady_clock;
double Seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
void Parameters(const lee::Analysis &analysis, const std::filesystem::path &path) {
    std::ofstream file(path);
    file << std::setprecision(17) << "{\"sample_rate\":" << analysis.SampleRate << ",\"frames\":" << analysis.Frames << ",\"contacts\":[";
    bool first = true;
    for (const auto &contact : analysis.Contacts) {
        if (!first) file << ',';
        first = false;
        file << "{\"frame\":" << contact.Frame << ",\"frames\":" << contact.Frames << ",\"bands\":[";
        for (size_t b = 0; b < 4; ++b) {
            if (b) file << ',';
            const auto &band = contact.Bands[b];
            file << "{\"gain\":" << band.Gain << ",\"denominator\":[";
            for (size_t i = 0; i < band.Denominator.size(); ++i) {
                if (i) file << ',';
                file << band.Denominator[i];
            }
            file << "],\"notches\":[";
            for (size_t i = 0; i < band.Notches.size(); ++i) {
                if (i) file << ',';
                file << '[' << band.Notches[i].Frequency << ',' << band.Notches[i].Bandwidth << ']';
            }
            file << "]}";
        }
        file << "]}";
    }
    file << "]}\n";
}
}
int main(int argc, char **argv) {
    try {
        if (argc < 3 || argc > 9) throw std::invalid_argument(
            "Usage: leeReproduce input.wav output [threshold=.03] [envelope=16] [tail=.1] [max_notches=12] [highpass=10000] [taps=129]");
        const auto source = ReadWave(argv[1]);
        std::vector<float> mono(source.Samples.size() / source.Channels);
        for (size_t n = 0; n < mono.size(); ++n)
            for (uint32_t c = 0; c < source.Channels; ++c) mono[n] += source.Samples[n * source.Channels + c] / source.Channels;
        lee::Settings options;
        if (argc > 3) options.EnvelopeThreshold = std::stod(argv[3]);
        if (argc > 4) options.EnvelopeFrames = uint32_t(std::stoul(argv[4]));
        if (argc > 5) options.TailSeconds = std::stod(argv[5]);
        if (argc > 6) options.MaximumNotches = uint32_t(std::stoul(argv[6]));
        if (argc > 7) options.HighpassHz = std::stod(argv[7]);
        if (argc > 8) options.HighpassTaps = uint32_t(std::stoul(argv[8]));
        const std::filesystem::path output(argv[2]);
        std::filesystem::create_directories(output);
        auto start = Clock::now();
        const auto analysis = lee::Analyze(mono, source.SampleRate, options);
        const double analysis_seconds = Seconds(start);
        start = Clock::now();
        const auto cpu = lee::Synthesize(analysis);
        const double cpu_seconds = Seconds(start);
        start = Clock::now();
        const auto plan = lee::PrepareSynthesis(analysis, 1, 1, false);
        const double prepare_seconds = Seconds(start);
        auto gpu = CreateGpu();
        start = Clock::now();
        const auto accelerated = lee::AnalyzeGpu(gpu, mono, source.SampleRate, options);
        const double gpu_analysis_seconds = Seconds(start);
        if (analysis.Onsets != accelerated.Onsets) throw std::runtime_error("GPU contact detector disagreement");
        const auto complete = lee::SynthesizeGpu(gpu, accelerated);
        start = Clock::now();
        const auto actual = lee::SynthesizeGpu(gpu, analysis);
        const double gpu_seconds = Seconds(start);
        double error = 0, complete_error = 0, energy = 0, peak_error = 0;
        for (size_t n = 0; n < cpu.size(); ++n) {
            error += std::pow(double(cpu[n]) - actual[n], 2);
            complete_error += std::pow(double(cpu[n]) - complete[n], 2);
            energy += double(cpu[n]) * cpu[n];
            peak_error = std::max(peak_error, std::abs(double(cpu[n]) - actual[n]));
        }
        const double relative = std::sqrt(error / std::max(energy, 1e-30));
        if (!std::isfinite(relative) || relative > 2e-6) throw std::runtime_error("Lee GPU agreement failed");
        const double pipeline_relative = std::sqrt(complete_error / std::max(energy, 1e-30));
        if (!std::isfinite(pipeline_relative) || pipeline_relative > .001) throw std::runtime_error("Lee complete GPU pipeline disagreement");
        std::array<double, 3> cpu_trials, gpu_trials;
        for (size_t i = 0; i < 3; ++i) {
            start = Clock::now();
            const auto repeated_cpu = lee::Synthesize(analysis);
            cpu_trials[i] = Seconds(start);
            start = Clock::now();
            const auto repeated_gpu = lee::SynthesizeGpu(gpu, analysis);
            gpu_trials[i] = Seconds(start);
            if (repeated_cpu != cpu || repeated_gpu != actual) throw std::runtime_error("Lee repeatability failure");
        }
        auto sorted_cpu = cpu_trials, sorted_gpu = gpu_trials;
        std::ranges::sort(sorted_cpu);
        std::ranges::sort(sorted_gpu);
        Parameters(accelerated, output / "gpu_parameters.json");
        WriteWave(output / "gpu_pipeline.wav", source.SampleRate, 1, complete);
        WriteWave(output / "input_mono.wav", source.SampleRate, 1, mono);
        WriteWave(output / "native.wav", source.SampleRate, 1, cpu);
        WriteWave(output / "gpu.wav", source.SampleRate, 1, actual);
        WriteWave(output / "half_speed.wav", source.SampleRate, 1, lee::Synthesize(analysis, .5));
        WriteWave(output / "reverse_trajectory.wav", source.SampleRate, 1, lee::Synthesize(analysis, 1, 1, true));
        Parameters(analysis, output / "parameters.json");
        std::ofstream envelope(output / "envelope.f64", std::ios::binary);
        envelope.write(reinterpret_cast<const char *>(analysis.Envelope.data()), analysis.Envelope.size() * sizeof(double));
        const auto bank = lee::MakeFilterBank();
        std::ofstream filters(output / "filters.json");
        filters << std::setprecision(17) << '[';
        for (size_t b = 0; b < 4; ++b) {
            if (b) filters << ',';
            filters << '[';
            for (size_t n = 0; n < bank.Analysis[b].size(); ++n) {
                if (n) filters << ',';
                filters << bank.Analysis[b][n];
            }
            filters << ']';
        }
        filters << "]\n";
        std::ofstream metrics(output / "timings.json");
        metrics << std::setprecision(12) << "{\"device\":\"" << DeviceName(gpu) << "\",\"contacts\":" << analysis.Contacts.size()
                << ",\"analysis_seconds\":" << analysis_seconds << ",\"cpu_synthesis_seconds\":" << cpu_seconds
                << ",\"gpu_synthesis_seconds\":" << gpu_seconds << ",\"gpu_relative_error\":" << relative
                << ",\"gpu_peak_error\":" << peak_error
                << ",\"gpu_analysis_seconds\":" << gpu_analysis_seconds << ",\"cpu_prepare_seconds\":" << prepare_seconds
                << ",\"gpu_pipeline_relative_error\":" << pipeline_relative
                << ",\"cpu_synthesis_median_seconds\":" << sorted_cpu[1] << ",\"gpu_synthesis_median_seconds\":" << sorted_gpu[1]
                << ",\"cpu_trials_seconds\":[" << cpu_trials[0] << ',' << cpu_trials[1] << ',' << cpu_trials[2]
                << "],\"gpu_trials_seconds\":[" << gpu_trials[0] << ',' << gpu_trials[1] << ',' << gpu_trials[2] << "]}\n";
        std::cout << analysis.Contacts.size() << " contacts, CPU " << cpu_seconds << " s, GPU " << gpu_seconds << " s, error " << relative << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
