#pragma once
#ifndef __METAL_VERSION__
#include <array>
#endif

namespace surface_audio::rough {
enum : unsigned { EventHistogramBins = 1024,
                  EventHistogramSize = EventHistogramBins + 2,
                  EventHistogramStart = 16,
                  EventStorageSize = EventHistogramStart + 4 * EventHistogramSize };
enum : int { EventForceLogMinimum = -12,
             EventForceLogMaximum = 4,
             EventDurationLogMinimum = -8,
             EventDurationLogMaximum = -1,
             EventWorkLogMinimum = -20,
             EventWorkLogMaximum = 2 };
enum EventCounter : unsigned {
    EventCompleted,
    EventLeftCensored,
    EventZeroWork,
    EventBelowWeight,
    EventBelowTenWeights,
    EventBelowHundredWeights,
    EventShort,
    EventFrames,
    EventOverflow,
    EventBottomBegin,
    EventBottomEnd,
    EventTopBegin,
    EventTopEnd,
    EventSamples
};
template<typename T> struct NodeContactEvent {
    T Peak, Work, TotalWork;
    unsigned Frames, LeftCensored;
};
#ifndef __METAL_VERSION__
struct ContactEventStatistics {
    unsigned Completed{}, LeftCensored{}, RightCensored{}, ZeroWork{}, Frames{}, Samples{};
    std::array<unsigned, 3> ForceBelowPaperWeights{};
    unsigned ShorterThan100Microseconds{};
    double Work{};
    std::array<unsigned, EventHistogramSize> Force{}, Duration{}, PositiveWork{}, NegativeWork{};
    bool operator==(const ContactEventStatistics &) const = default;
};
#endif
}
