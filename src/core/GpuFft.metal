#include <metal_stdlib>
using namespace metal;
struct FftBlock {
    uint Size, Count, LogSize, Width;
    float Sign, Scale;
};
inline float2 FftProduct(float2 a, float2 b) { return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
kernel void FftMultiply(constant uint &size [[buffer(0)]], device const float2 *input [[buffer(1)]], device float2 *output [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index < size) output[index] = FftProduct(input[index], input[size + index]);
}
kernel void FftLocal(constant FftBlock &p [[buffer(0)]], device const float2 *input [[buffer(1)]], device float2 *output [[buffer(2)]], device const float2 *twiddles [[buffer(3)]], uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float2 data[4096];
    const uint size = min(p.Size, 4096u), chunks = p.Size / size, record = group / chunks, offset = (group % chunks) * size;
    for (uint index = lane; index < size; index += 256) {
        const uint reversed = p.LogSize ? reverse_bits(offset + index) >> (32 - p.LogSize) : 0;
        data[index] = input[record * p.Size + reversed];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 2; width <= size; width *= 2) {
        const uint midpoint = width / 2;
        for (uint butterfly = lane; butterfly < size / 2; butterfly += 256) {
            const uint k = butterfly % midpoint, first = (butterfly / midpoint) * width + k;
            const float2 w = twiddles[k * (p.Size / width)] * float2(1, p.Sign);
            const float2 a = data[first], b = FftProduct(data[first + midpoint], w);
            data[first] = a + b;
            data[first + midpoint] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint index = lane; index < size; index += 256) output[record * p.Size + offset + index] = data[index] * p.Scale;
}
kernel void FftStage(constant FftBlock &p [[buffer(0)]], device const float2 *input [[buffer(1)]], device float2 *output [[buffer(2)]], device const float2 *twiddles [[buffer(3)]], uint2 index [[thread_position_in_grid]]) {
    if (index.x >= p.Size / 2 || index.y >= p.Count) return;
    const uint midpoint = p.Width / 2, k = index.x % midpoint, first = index.y * p.Size + (index.x / midpoint) * p.Width + k;
    const float2 w = twiddles[k * (p.Size / p.Width)] * float2(1, p.Sign);
    const float2 a = input[first], b = FftProduct(input[first + midpoint], w);
    output[first] = (a + b) * p.Scale;
    output[first + midpoint] = (a - b) * p.Scale;
}
