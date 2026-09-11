#pragma once

#include "core/Gpu.h"
#include <array>
#include <vector>

namespace surface_audio::lee {
inline constexpr std::array<uint32_t, 4> Decimation{8, 8, 4, 2};
struct Settings {
    double HighpassHz{10000}, EnvelopeThreshold{.03}, NotchProminenceDb{3}, TailSeconds{.1};
    uint32_t EnvelopeFrames{16}, HighpassTaps{129}, MaximumNotches{12};
    std::array<uint32_t, 4> LpcOrder{25, 10, 5, 5};
};
struct Notch {
    double Frequency{}, Bandwidth{};
};
struct Band {
    std::vector<double> Denominator;
    std::vector<Notch> Notches;
    double Gain{};
};
struct Contact {
    uint32_t Frame{}, Frames{};
    std::array<Band, 4> Bands;
};
struct Analysis {
    uint32_t SampleRate{}, Frames{};
    Settings Options;
    std::vector<uint32_t> Onsets;
    std::vector<double> Envelope;
    std::vector<Contact> Contacts;
};
struct FirBank {
    std::array<std::vector<double>, 4> Analysis, Synthesis;
};
FirBank MakeFilterBank();
std::array<std::vector<double>, 4> SplitBands(std::span<const double>, const FirBank &);
std::vector<double> MergeBands(const std::array<std::vector<double>, 4> &, const FirBank &, uint32_t frames, bool analysis_delay);
std::vector<uint32_t> DetectContacts(std::span<const float>, uint32_t sample_rate, const Settings &, std::vector<double> &envelope);
std::vector<Notch> EstimateNotches(std::span<const double>, double prominence_db, uint32_t maximum);
std::vector<double> FilterNotches(std::span<const double>, std::span<const Notch>, bool inverse);
Analysis AnalyzeGpu(Gpu &, std::span<const float>, uint32_t sample_rate, Settings = {});
Analysis Analyze(std::span<const float>, uint32_t sample_rate, Settings = {});
// Rate scales onset times, and trajectory reverses the sequence of contact filters without reversing their causal responses.
std::vector<float> Synthesize(const Analysis &, double rate = 1, double gain = 1, bool reverse_trajectory = false);
std::vector<float> SynthesizeGpu(Gpu &, const Analysis &, double rate = 1, double gain = 1, bool reverse_trajectory = false);
}
