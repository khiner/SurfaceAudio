#include <metal_stdlib>
using namespace metal;
struct InterpolationParameters {
    uint Nodes, Frames;
};
kernel void InterpolateSignals(constant InterpolationParameters &p [[buffer(0)]], device const float *signals [[buffer(1)]], device const float *nodes [[buffer(2)]], device const float *positions [[buffer(3)]], device float *output [[buffer(4)]], uint frame [[thread_position_in_grid]]) {
    if (frame >= p.Frames) return;
    const float position = positions[frame];
    if (p.Nodes == 1 || position <= nodes[0]) output[frame] = signals[frame];
    else if (position >= nodes[p.Nodes - 1]) output[frame] = signals[(p.Nodes - 1) * p.Frames + frame];
    else {
        uint low = 0, high = p.Nodes - 1;
        while (high - low > 1) {
            const uint mid = low + (high - low) / 2;
            if (nodes[mid] <= position) low = mid;
            else high = mid;
        }
        const float weight = (position - nodes[low]) / (nodes[high] - nodes[low]);
        output[frame] = mix(signals[low * p.Frames + frame], signals[high * p.Frames + frame], weight);
    }
}
