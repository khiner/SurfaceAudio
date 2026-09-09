#include <metal_stdlib>
using namespace metal;

struct ModalBlock {
    uint Voices, Modes, Frames;
};

kernel void ModalSynthesize(constant ModalBlock &p [[buffer(0)]], device const float *coefficients [[buffer(1)]], device float *state [[buffer(2)]], device const float *input [[buffer(3)]], device float *output [[buffer(4)]], uint voice [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]], uint lanes [[threads_per_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]], uint simd_id [[simdgroup_index_in_threadgroup]]) {
    const uint count = p.Voices * p.Modes, index = voice * p.Modes + lane;
    const bool active = lane < p.Modes;
    const float cosine = active ? coefficients[index] : 0, sine = active ? coefficients[count + index] : 0, amplitude = active ? coefficients[2 * count + index] : 0;
    float real = active ? state[index] : 0, imaginary = active ? state[count + index] : 0;
    threadgroup float sums[32];
    for (uint frame = 0; frame < p.Frames; ++frame) {
        const float sample = active ? input[voice * p.Frames + frame] : 0;
        const float next_real = fma(amplitude, sample, fma(cosine, real, -sine * imaginary));
        imaginary = fma(sine, real, cosine * imaginary);
        real = next_real;
        const float sum = simd_sum(imaginary);
        if (lanes == 32) {
            if (!lane) output[voice * p.Frames + frame] = sum;
        } else {
            if (!simd_lane) sums[simd_id] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!lane) {
                float total = 0;
                for (uint index = 0; index < lanes / 32; ++index) total += sums[index];
                output[voice * p.Frames + frame] = total;
            }
            // Keep the next frame from overwriting sums before lane zero has consumed them.
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (active) {
        state[index] = real;
        state[count + index] = imaginary;
    }
}
