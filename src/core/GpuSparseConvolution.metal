#include <metal_stdlib>
using namespace metal;
struct SparseConvolutionBlock {
    uint Size, Frames, Observations;
    float Step, Penalty, Momentum;
};
kernel void SparseConvolutionPrepare(constant SparseConvolutionBlock &p [[buffer(0)]], device const float2 *transformed [[buffer(1)]], device float4 *model [[buffer(2)]], device float2 *rhs [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index >= p.Size) return;
    float power = 0;
    float2 cross = 0;
    for (uint channel = 0; channel < p.Observations; ++channel) {
        const float2 source = transformed[2 * channel * p.Size + index], impulse = transformed[(2 * channel + 1) * p.Size + index];
        power += dot(impulse, impulse);
        cross += float2(dot(source, impulse), source.y * impulse.x - source.x * impulse.y);
    }
    model[index] = float4(power, cross, 0);
    rhs[index] = cross;
}
kernel void SparseConvolutionMaximum(constant uint2 &p [[buffer(0)]], device const float *input [[buffer(1)]], device float *partial [[buffer(2)]], uint index [[thread_position_in_grid]], uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float maxima[8];
    const float value = simd_max(index < p.x ? input[index * p.y] : 0.f);
    if (lane % 32 == 0) maxima[lane / 32] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 32) {
        const float result = simd_max(lane < 8 ? maxima[lane] : 0.f);
        if (lane == 0) partial[group] = result;
    }
}
kernel void SparseConvolutionGradient(constant SparseConvolutionBlock &p [[buffer(0)]], device const float4 *model [[buffer(1)]], device const float2 *input [[buffer(2)]], device float2 *output [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index < p.Size) output[index] = model[index].x * input[index] - model[index].yz;
}
kernel void SparseConvolutionStep(constant SparseConvolutionBlock &p [[buffer(0)]], device const float2 *gradient [[buffer(1)]], device float *coefficients [[buffer(2)]], device float2 *extrapolated [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index >= p.Size) return;
    const float previous = coefficients[index];
    const float next = index < p.Frames ? max(0.f, extrapolated[index].x - p.Step * (gradient[index].x + p.Penalty)) : 0.f;
    coefficients[index] = next;
    extrapolated[index] = float2(next + p.Momentum * (next - previous), 0);
}
kernel void SparseConvolutionKkt(constant SparseConvolutionBlock &p [[buffer(0)]], device const float2 *gradient [[buffer(1)]], device const float *coefficients [[buffer(2)]], device float *violation [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index >= p.Size) return;
    const float g = gradient[index].x + p.Penalty;
    violation[index] = index < p.Frames ? (coefficients[index] > 0 ? abs(g) : max(0.f, -g)) : 0.f;
}
