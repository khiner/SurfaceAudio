#include <metal_stdlib>
using namespace metal;
struct FirResampleJob {
    uint InputOffset, InputFrames, OutputOffset, OutputFrames, TapOffset, Taps, Upsample, Downsample;
    int Delay;
};
kernel void ResampleFir(device const FirResampleJob *jobs [[buffer(0)]], device const float *input [[buffer(1)]], device const float *taps [[buffer(2)]], device float *output [[buffer(3)]], constant uint &count [[buffer(4)]], uint2 index [[thread_position_in_grid]]) {
    if (index.y >= count) return;
    const FirResampleJob job = jobs[index.y];
    if (index.x >= job.OutputFrames) return;
    const int position = int(index.x * job.Downsample) + job.Delay;
    const int first = max(0, position - int(job.Taps) + 1);
    const uint begin = uint(first / int(job.Upsample) + (first % int(job.Upsample) != 0));
    const uint end = position < 0 ? 0 : min(job.InputFrames, uint(position / int(job.Upsample)) + 1);
    float value = 0;
    for (uint sample = begin; sample < end; ++sample) value += input[job.InputOffset + sample] * taps[job.TapOffset + uint(position - int(sample * job.Upsample))];
    output[job.OutputOffset + index.x] = value;
}
