#include <metal_stdlib>
using namespace metal;

struct LagrangeBlock { uint Size, Modes; float Floor, Padding; };
struct LagrangeComb { uint Frames; float Length, DelayScale, First, Second; };
float LagrangeDelayed(device const float *signal, uint frames, float index) {
    const int lower = int(floor(index));
    const float first = lower >= 0 && uint(lower) < frames ? signal[lower] : 0;
    const float second = lower + 1 >= 0 && uint(lower + 1) < frames ? signal[lower + 1] : 0;
    return mix(first, second, index - floor(index));
}
kernel void LagrangeMovingComb(constant LagrangeComb &p [[buffer(0)]], device const float *signal [[buffer(1)]],
                               device const float *position [[buffer(2)]], device float *output [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index >= p.Frames) return;
    output[index] = signal[index] + p.First * LagrangeDelayed(signal, p.Frames, float(index) - position[index] * p.DelayScale) +
                    p.Second * LagrangeDelayed(signal, p.Frames, float(index) - (p.Length - position[index]) * p.DelayScale);
}
float2 LagrangeDivide(float2 a, float2 b) { return float2(dot(a,b), a.y*b.x-a.x*b.y)/dot(b,b); }
kernel void LagrangeResponse(constant uint2 &p [[buffer(0)]], device const float4 *modes [[buffer(1)]],
                            device float *output [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index >= p.x) return;
    float sum = 0;
    for (uint k = 0; k < p.y; ++k) {
        const float4 first = modes[2 * k], second = modes[2 * k + 1];
        const float product = first.x * float(index), remainder = fma(first.x, float(index), -product) + first.y * float(index);
        const float phase = 6.2831853071795864769f * fract(fract(product) + remainder);
        sum += exp(-first.z * float(index)) * (first.w * cos(phase) - second.x * sin(phase));
    }
    output[index] = sum;
}
kernel void LagrangeInverse(constant LagrangeBlock &p [[buffer(0)]], device const float4 *modes [[buffer(1)]],
                            device const float2 *input [[buffer(2)]], device float2 *output [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index >= p.Size) return;
    const float angle = 6.2831853071795864769f * float(index) / p.Size;
    float2 response(0);
    for (uint k = 0; k < p.Modes; ++k) {
        const float4 mode = modes[k];
        const float radius = exp(-mode.y);
        const float distance = mode.y < .01f ? mode.y * (1 - mode.y * (.5f - mode.y * (1.f / 6 - mode.y / 24))) : 1 - radius;
        const float first = mode.x - angle, second = -mode.x - angle;
        const float2 a(distance + 2 * radius * pow(sin(first * .5f), 2), -radius * sin(first));
        const float2 b(distance + 2 * radius * pow(sin(second * .5f), 2), -radius * sin(second));
        response += .5f * (LagrangeDivide(mode.zw, a) + LagrangeDivide(float2(mode.z, -mode.w), b));
    }
    const float denominator = max(dot(response, response), p.Floor * p.Floor);
    output[index] = denominator > 0 ? float2(dot(input[index], response), input[index].y * response.x - input[index].x * response.y) / denominator : float2(0);
}
kernel void LagrangeEnvelopeInverse(constant LagrangeBlock &p [[buffer(0)]], device const float2 *input [[buffer(1)]],
                                    device float2 *output [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index >= p.Size) return;
    const float2 response = input[p.Size + index];
    const float denominator = dot(response, response) + p.Floor * p.Floor;
    output[index] = float2(dot(input[index], response), input[index].y * response.x - input[index].x * response.y) / denominator;
}
