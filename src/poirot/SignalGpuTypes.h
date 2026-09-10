#pragma once

namespace surface_audio::poirot {
struct SignalGpuParameters {
    unsigned Voices, Modes, Frames, Frame;
};
struct SignalGpuControls {
    float SampleRate;
    unsigned Activation;
    float Lambda, SplitThreshold, SplitSlope, PowerScale, ReturnGain;
    unsigned ShapeWeightedSplit, ResetPhaseAtActivation;
};
struct SignalGpuMode {
    float Frequency, Loss, Threshold, Weight, Shape;
};
} // namespace surface_audio::poirot
