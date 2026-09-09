#include <metal_stdlib>
using namespace metal;

struct SpectralBlock {
    uint Samples, Size, Hop, Frames, LogSize, Scale;
    float FloorSquared, Delta;
};

inline float Hann(uint index, uint size) { return .5f - .5f * cos(2 * M_PI_F * float(index) / float(size)); }
inline uint Reversed(uint index, uint bits) { return reverse_bits(index) >> (32 - bits); }

inline void Transform(threadgroup float2 *data, uint size, uint lane, float sign) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 2; width <= size; width *= 2) {
        const uint midpoint = width / 2;
        for (uint butterfly = lane; butterfly < size / 2; butterfly += 256) {
            const uint k = butterfly % midpoint, first = (butterfly / midpoint) * width + k;
            const float angle = sign * 2 * M_PI_F * float(k) / float(width);
            const float2 a = data[first], b = data[first + midpoint];
            const float2 w = float2(cos(angle), sin(angle));
            const float2 product = float2(w.x * b.x - w.y * b.y, w.x * b.y + w.y * b.x);
            data[first] = a + product;
            data[first + midpoint] = a - product;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Normalized (high, low) pairs reduce FFT cancellation near tonal bins' 1e-6 magnitude floor.
inline float2 AddPrecise(float2 a, float2 b) {
    const float sum = a.x + b.x, virtual_b = sum - a.x;
    const float error = ((a.x - (sum - virtual_b)) + (b.x - virtual_b)) + a.y + b.y;
    const float high = sum + error;
    return float2(high, error - (high - sum));
}
inline float2 MultiplyPrecise(float2 a, float2 b) {
    const float product = a.x * b.x;
    const float error = fma(a.x, b.x, -product) + a.x * b.y + a.y * b.x;
    const float high = product + error;
    return float2(high, error - (high - product));
}
inline float4 AddComplexPrecise(float4 a, float4 b) { return float4(AddPrecise(a.xy, b.xy), AddPrecise(a.zw, b.zw)); }
inline float4 MultiplyComplexPrecise(float4 a, float4 b) {
    return float4(AddPrecise(MultiplyPrecise(a.xy, b.xy), -MultiplyPrecise(a.zw, b.zw)), AddPrecise(MultiplyPrecise(a.xy, b.zw), MultiplyPrecise(a.zw, b.xy)));
}

kernel void SpectralForwardPrecise(constant SpectralBlock &p [[buffer(0)]], device const float *waveform [[buffer(1)]], device float4 *transformed [[buffer(2)]], device const float2 *window [[buffer(3)]], device const float4 *twiddles [[buffer(4)]], uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    // Even/odd 2048-point FFTs keep compensated storage within Metal's 32 KiB threadgroup limit.
    threadgroup float4 data[2048];
    const uint parts = p.Size > 2048 ? 2 : 1, size = p.Size / parts;
    const uint frame = group / parts, parity = group % parts;
    for (uint index = lane; index < size; index += 256) {
        const uint position = parts * index + parity;
        const int sample = int(frame * p.Hop + position) - int(p.Size / 2);
        const float value = sample >= 0 && sample < int(p.Samples) ? waveform[sample] : 0;
        data[Reversed(index, p.LogSize - (parts - 1))] = float4(MultiplyPrecise(float2(value, 0), window[position]), 0, 0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 2; width <= size; width *= 2) {
        const uint midpoint = width / 2;
        for (uint butterfly = lane; butterfly < size / 2; butterfly += 256) {
            const uint k = butterfly % midpoint, first = (butterfly / midpoint) * width + k;
            const float4 a = data[first], product = MultiplyComplexPrecise(twiddles[k * (p.Size / width)], data[first + midpoint]);
            data[first] = AddComplexPrecise(a, product);
            data[first + midpoint] = AddComplexPrecise(a, -product);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint index = lane; index < size; index += 256) transformed[frame * p.Size + parity * size + index] = data[index];
}

kernel void SpectralFinishForward(constant SpectralBlock &p [[buffer(0)]], device const float4 *transformed [[buffer(1)]], device float2 *spectrum [[buffer(2)]], device const float4 *twiddles [[buffer(3)]], uint2 index [[thread_position_in_grid]]) {
    if (index.x >= p.Size || index.y >= p.Frames) return;
    float4 value = transformed[index.y * p.Size + index.x];
    if (p.Size > 2048) {
        const uint k = index.x % (p.Size / 2);
        const float4 a = transformed[index.y * p.Size + k];
        const float4 b = MultiplyComplexPrecise(twiddles[k], transformed[index.y * p.Size + p.Size / 2 + k]);
        value = AddComplexPrecise(a, index.x < p.Size / 2 ? b : -b);
    }
    spectrum[index.y * p.Size + index.x] = float2(value.x + value.y, value.z + value.w);
}

inline float Magnitude(float power, uint scale) {
    if (scale == 0) return (10 / log(10.f)) * log(power);
    if (scale == 1) return .5f * log(power);
    return sqrt(power);
}

kernel void SpectralCompareAdjoint(constant SpectralBlock &p [[buffer(0)]], device const float2 *spectrum [[buffer(1)]], device const float2 *target [[buffer(2)]], device float *adjoint [[buffer(3)]], device float *frame_loss [[buffer(4)]], uint frame [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float2 data[4096];
    const float weight = 1.f / (float(p.Frames) * float(p.Size / 2 + 1));
    float loss = 0;
    for (uint index = lane; index < p.Size; index += 256) {
        float2 gradient = 0;
        if (index <= p.Size / 2) {
            const float2 x = spectrum[frame * p.Size + index], y = target[frame * p.Size + index];
            const float power = dot(x, x) + p.FloorSquared;
            const float difference = Magnitude(power, p.Scale) - Magnitude(dot(y, y) + p.FloorSquared, p.Scale);
            const float absolute = abs(difference);
            loss += (absolute <= p.Delta ? .5f * difference * difference : p.Delta * (absolute - .5f * p.Delta)) * weight;
            const float derivative = clamp(difference, -p.Delta, p.Delta) * weight;
            const float factor = p.Scale == 0 ? 20 / log(10.f) / power : (p.Scale == 1 ? 1 / power : rsqrt(power));
            gradient = derivative * factor * x;
        }
        // Real-input adjoint uses positive frequencies without mirroring or inverse normalization.
        data[Reversed(index, p.LogSize)] = gradient;
    }
    Transform(data, p.Size, lane, 1);
    for (uint index = lane; index < p.Size; index += 256) adjoint[frame * p.Size + index] = data[index].x * Hann(index, p.Size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Reuse FFT storage after every lane has written its sample adjoint.
    data[lane] = float2(loss, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128; offset; offset /= 2) {
        if (lane < offset) data[lane] += data[lane + offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!lane) frame_loss[frame] = data[0].x;
}

inline float Overlap(constant SpectralBlock &p, device const float *adjoint, uint sample) {
    const int first = max(0, (int(sample) - int(p.Size / 2) + int(p.Hop)) / int(p.Hop));
    const uint last = min(p.Frames - 1, (sample + p.Size / 2) / p.Hop);
    float gradient = 0;
    for (uint frame = uint(first); frame <= last; ++frame) gradient += adjoint[frame * p.Size + sample + p.Size / 2 - frame * p.Hop];
    return gradient;
}

kernel void SpectralOverlap(constant SpectralBlock &p0 [[buffer(0)]], constant SpectralBlock &p1 [[buffer(1)]], constant SpectralBlock &p2 [[buffer(2)]], constant SpectralBlock &p3 [[buffer(3)]], device const float *a0 [[buffer(4)]], device const float *a1 [[buffer(5)]], device const float *a2 [[buffer(6)]], device const float *a3 [[buffer(7)]], device float *gradient [[buffer(8)]], uint sample [[thread_position_in_grid]]) {
    if (sample >= p0.Samples) return;
    gradient[sample] = Overlap(p0, a0, sample) + Overlap(p1, a1, sample) + Overlap(p2, a2, sample) + Overlap(p3, a3, sample);
}

kernel void SpectralReduce(constant SpectralBlock &p0 [[buffer(0)]], constant SpectralBlock &p1 [[buffer(1)]], constant SpectralBlock &p2 [[buffer(2)]], constant SpectralBlock &p3 [[buffer(3)]], device const float *l0 [[buffer(4)]], device const float *l1 [[buffer(5)]], device const float *l2 [[buffer(6)]], device const float *l3 [[buffer(7)]], device float *loss [[buffer(8)]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float4 partial[256];
    float4 sum = 0;
    for (uint frame = lane; frame < p0.Frames; frame += 256) sum.x += l0[frame];
    for (uint frame = lane; frame < p1.Frames; frame += 256) sum.y += l1[frame];
    for (uint frame = lane; frame < p2.Frames; frame += 256) sum.z += l2[frame];
    for (uint frame = lane; frame < p3.Frames; frame += 256) sum.w += l3[frame];
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128; offset; offset /= 2) {
        if (lane < offset) partial[lane] += partial[lane + offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!lane) {
        loss[0] = partial[0].x + partial[0].y + partial[0].z + partial[0].w;
        for (uint index = 0; index < 4; ++index) loss[index + 1] = partial[0][index];
    }
}

struct SpectralPoolBlock {
    uint Pools, Size, Bins, Frames;
    float Weight, FloorSquared;
};
struct SpectralPoolRegion {
    uint FirstBin, EndBin, FirstFrame, EndFrame, Band;
};

kernel void SpectralPoolEnergy(constant SpectralPoolBlock &p [[buffer(0)]], device const SpectralPoolRegion *regions [[buffer(1)]], device const float2 *spectrum [[buffer(2)]], device float *energy [[buffer(3)]], uint pool [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const SpectralPoolRegion region = regions[pool];
    const uint bins = region.EndBin - region.FirstBin, count = bins * (region.EndFrame - region.FirstFrame);
    float sum = 0;
    for (uint index = lane; index < count; index += 256) {
        const float2 value = spectrum[(region.FirstFrame + index / bins) * p.Size + region.FirstBin + index % bins];
        sum += dot(value, value) / float(count);
    }
    threadgroup float partial[256];
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride /= 2) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!lane) energy[pool] = partial[0];
}

kernel void SpectralPoolCompare(constant SpectralPoolBlock &p [[buffer(0)]], device const SpectralPoolRegion *regions [[buffer(1)]], device const float *energy [[buffer(2)]], device const float2 *target_scale [[buffer(3)]], device float *coefficients [[buffer(4)]], device float *loss [[buffer(5)]], uint pool [[thread_position_in_grid]]) {
    if (pool >= p.Pools) return;
    const SpectralPoolRegion region = regions[pool];
    const float count = float(region.EndBin - region.FirstBin) * float(region.EndFrame - region.FirstFrame);
    const float rms = sqrt(energy[pool] + p.FloorSquared), difference = rms - sqrt(target_scale[pool].x + p.FloorSquared);
    const float weight = p.Weight / float(p.Pools), denominator_squared = target_scale[pool].y;
    loss[pool + 1] = weight * .5f * difference * difference / denominator_squared;
    coefficients[pool] = weight * difference / (denominator_squared * count * rms);
}

kernel void SpectralPoolAdjoint(constant SpectralBlock &p [[buffer(0)]], device const uint *membership [[buffer(1)]], device const float *coefficients [[buffer(2)]], device const float2 *spectrum [[buffer(3)]], device float *adjoint [[buffer(4)]], uint frame [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float2 data[4096];
    for (uint bin = lane; bin < p.Size; bin += 256) {
        const uint pool = bin <= p.Size / 2 ? membership[frame * (p.Size / 2 + 1) + bin] : UINT_MAX;
        const float2 gradient = pool != UINT_MAX ? coefficients[pool] * spectrum[frame * p.Size + bin] : float2(0);
        data[Reversed(bin, p.LogSize)] = gradient;
    }
    Transform(data, p.Size, lane, 1);
    for (uint index = lane; index < p.Size; index += 256) adjoint[frame * p.Size + index] = data[index].x * Hann(index, p.Size);
}

kernel void SpectralPoolAdd(constant SpectralBlock &p [[buffer(0)]], constant SpectralPoolBlock &pool [[buffer(1)]], device const float *adjoint [[buffer(2)]], device float *pooled_loss [[buffer(3)]], device float *gradient [[buffer(4)]], device float *loss [[buffer(5)]], uint sample [[thread_position_in_grid]]) {
    if (sample >= p.Samples) return;
    gradient[sample] += Overlap(p, adjoint, sample);
    if (!sample) {
        float total = 0;
        for (uint index = 0; index < pool.Pools; ++index) total += pooled_loss[index + 1];
        pooled_loss[0] = total;
        loss[0] += total;
    }
}
