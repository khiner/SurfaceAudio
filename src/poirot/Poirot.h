#pragma once
#include "core/Gpu.h"
#include <span>
#include <vector>

namespace surface_audio::poirot {
struct StringParameters {
    double SampleRate{44100}, Length{.5}, WaveSpeed{404.02}, Stiffness{1.297}, Loss0{.05}, Loss1{.002};
    double Density{7800}, Area{7.85e-7}, ExcitationPosition{.15}, ExcitationForce{200}, ExcitationDuration{.001};
    double ObstaclePosition{.5}, ObstacleHeight{}, ContactStiffness{5e10}, ContactExponent{1.4}, Activation{.5}, ReadoutPosition{.8};
};
struct SignalParameters {
    float SampleRate{44100}, Activation{.5f}, Lambda{1.f / 800}, Height{.005f}, Position{.5f};
    float SplitThreshold{340}, SplitSlope{.0006f}, PowerScale{1}, ReturnGain{1};
    // Both options reproduce behavior measured in author audio, as documented in docs/Poirot.md.
    bool ShapeWeightedSplit{}, ResetPhaseAtActivation{};
};
struct Mode {
    float Frequency{}, Damping{}, Amplitude{}, Phase{};
};
struct SignalState {
    SignalParameters Parameters;
    uint64_t Frame{};
    std::vector<Mode> Modes{};
    std::vector<float> Power{}, UpperPhase{}, LowerPhase{}, Threshold{}, Weight{}, Shape{}, Loss{}, Transfer{};
};
std::vector<Mode> StringModes(const StringParameters &);
SignalState MakeSignal(const SignalParameters &, std::span<const Mode>);
float TransferPower(const SignalParameters &, std::span<const float> power, std::span<const float> threshold, std::span<const float> weight, std::span<float> transfer);
void RenderSignal(SignalState &, std::span<float> output);
struct GpuSignal {
    GpuBuffer Parameters, Modes, State, Output;
    GpuKernel Kernel;
    uint32_t Voices{}, ModeCount{}, Capacity{}, Threads{};
    uint64_t Frame{};
};
GpuSignal CreateGpuSignal(Gpu &, std::span<const SignalState>, uint32_t frame_capacity);
void EncodeSignal(Gpu &, GpuSignal &, uint32_t frames);
struct StringState {
    StringParameters Parameters;
    uint64_t Frame{};
    uint32_t Segments{};
    double Spacing{}, LastContactForce{}, MaximumPenetration{};
    std::vector<double> Current{}, Previous{}, Next{};
};
StringState MakeString(const StringParameters &);
double ContactPotential(double penetration, double stiffness, double exponent);
double ContactGradient(double a, double b, double stiffness, double exponent);
void RenderString(StringState &, std::span<float> output, std::span<float> contact = {});
}
