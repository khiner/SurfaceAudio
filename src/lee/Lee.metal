#include <metal_stdlib>
using namespace metal;

struct LeeOverlapContact {
    uint Frame, Frames, Offset;
};
kernel void LeeOverlap(device const LeeOverlapContact *contacts [[buffer(0)]], device const float *bands [[buffer(1)]], device float *output [[buffer(2)]], constant uint2 &shape [[buffer(3)]], uint frame [[thread_position_in_grid]]) {
    if (frame >= shape.y) return;
    float sum = 0, correction = 0;
    for (uint k = 0; k < shape.x; ++k) {
        const auto contact = contacts[k];
        if (frame < contact.Frame || frame - contact.Frame >= contact.Frames) continue;
        const uint local = frame - contact.Frame;
        for (uint band = 0; band < 4; ++band) {
            const float value = bands[contact.Offset + band * contact.Frames + local] - correction;
            const float next = sum + value;
            correction = (next - sum) - value;
            sum = next;
        }
    }
    output[frame] = sum;
}
