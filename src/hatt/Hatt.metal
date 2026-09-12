#include "HattLayout.h"
#include "core/ExtendedFloat.h"
using namespace metal;
using namespace surface_audio;
using namespace surface_audio::hatt;

kernel void HattRender(constant uint2 &p [[buffer(0)]], device const float2 *coefficients [[buffer(1)]], device const uint4 *metadata [[buffer(2)]], device const float *noise [[buffer(3)]], device float *output [[buffer(4)]], uint voice [[thread_position_in_grid]]) {
    if (voice >= p.y) return;
    const uint2 orders = metadata[voice].zw;
    float2 history[MaximumOrder]{}, innovations[MaximumOrder]{};
    for (uint frame = 0; frame < p.x; ++frame) {
        const uint index = voice * p.x + frame;
        const auto c = coefficients + index * FilterStride;
        const float2 excitation = MultiplyExtended(float2(noise[index], 0), c[ModelStride]);
        float2 value = MultiplyExtended(c[PolynomialSize], excitation);
        for (uint i = 0; i < orders.x; ++i) value = AddExtended(value, -MultiplyExtended(c[i + 1], history[i]));
        for (uint i = 0; i < orders.y; ++i) value = AddExtended(value, MultiplyExtended(c[i + PolynomialSize + 1], innovations[i]));
        for (uint i = MaximumOrder - 1; i > 0; --i) {
            history[i] = history[i - 1];
            innovations[i] = innovations[i - 1];
        }
        history[0] = value;
        innovations[0] = excitation;
        output[index] = value.x + value.y;
    }
}

float2 HattCosine(float2 angle, device const float2 *coefficients) {
    const float2 square = MultiplyExtended(angle, angle);
    float2 value = coefficients[14];
    for (int i = 13; i >= 0; --i) value = AddExtended(MultiplyExtended(value, square), coefficients[i]);
    return value;
}
float2 HattBlend(device const float2 *models, uint3 nodes, thread const float2 *weights, uint field) {
    float2 result = 0;
    for (uint i = 0; i < 3; ++i) result = AddExtended(result, MultiplyExtended(weights[i], models[nodes[i] * ModelStride + field]));
    return result;
}
void HattPolynomial(device const float2 *models, uint3 nodes, thread const float2 *weights, uint offset, uint order, device const float2 *cosine, device float2 *output, float2 gain) {
    float2 sum[MaximumOrder + 2]{}, difference[MaximumOrder + 2]{};
    sum[0] = difference[0] = float2(1, 0);
    for (uint i = 0; i < order; ++i) {
        const float2 middle = -2 * HattCosine(HattBlend(models, nodes, weights, offset + i), cosine);
        thread float2 *polynomial = i % 2 ? difference : sum;
        for (uint j = i - i % 2 + 2; j > 0; --j)
            polynomial[j] = AddExtended(polynomial[j], AddExtended(MultiplyExtended(middle, polynomial[j - 1]), j > 1 ? polynomial[j - 2] : float2(0)));
    }
    for (uint j = order + 1; j > 0; --j) {
        if (order % 2) difference[j] = AddExtended(difference[j], j > 1 ? -difference[j - 2] : float2(0));
        else {
            sum[j] = AddExtended(sum[j], sum[j - 1]);
            difference[j] = AddExtended(difference[j], -difference[j - 1]);
        }
    }
    for (uint i = 0; i < PolynomialSize; ++i) output[i] = i <= order ? MultiplyExtended(.5f * AddExtended(sum[i], difference[i]), gain) : float2(0);
}
kernel void HattInterpolate(constant uint2 &p [[buffer(0)]], device const float2 *models [[buffer(1)]], device const float2 *coordinates [[buffer(2)]], device const uint4 *triangles [[buffer(3)]], device const float2 *affine [[buffer(4)]], device const uint4 *metadata [[buffer(5)]], device const float2 *cosine [[buffer(6)]], device float2 *coefficients [[buffer(7)]], uint index [[thread_position_in_grid]]) {
    if (index >= p.x * p.y) return;
    const auto meta = metadata[index / p.x];
    const float2 speed = coordinates[2 * index], force = coordinates[2 * index + 1];
    for (uint triangle = meta.x; triangle < meta.x + meta.y; ++triangle) {
        const auto a = affine + 6 * triangle;
        const float2 v = AddExtended(AddExtended(MultiplyExtended(a[0], speed), MultiplyExtended(a[1], force)), a[2]);
        const float2 w = AddExtended(AddExtended(MultiplyExtended(a[3], speed), MultiplyExtended(a[4], force)), a[5]);
        const float2 u = AddExtended(float2(1, 0), -AddExtended(v, w));
        if (min(u.x, min(v.x, w.x)) < -1e-8f) continue;
        float2 weights[3] = {u, v, w};
        float2 total = 0;
        for (uint i = 0; i < 3; ++i) {
            if (weights[i].x < 0) weights[i] = 0;
            total = AddExtended(total, weights[i]);
        }
        const float2 reciprocal = AddExtended(float2(1, 0), AddExtended(float2(1, 0), -total));
        for (uint i = 0; i < 3; ++i) weights[i] = MultiplyExtended(weights[i], reciprocal);
        const auto nodes = triangles[triangle].xyz;
        const float2 variance = HattBlend(models, nodes, weights, 2 * MaximumOrder), gain = HattBlend(models, nodes, weights, 2 * MaximumOrder + 1);
        HattPolynomial(models, nodes, weights, 0, meta.z, cosine, coefficients + FilterStride * index, float2(1, 0));
        HattPolynomial(models, nodes, weights, MaximumOrder, meta.w, cosine, coefficients + FilterStride * index + PolynomialSize, gain);
        const float root = sqrt(max(variance.x, 0.f));
        const float2 error = AddExtended(variance, -MultiplyExtended(float2(root, 0), float2(root, 0)));
        coefficients[FilterStride * index + ModelStride] = root > 0 ? AddExtended(float2(root, 0), float2((error.x + error.y) / (2 * root), 0)) : float2(0);
        return;
    }
    for (uint i = 0; i < FilterStride; ++i) coefficients[FilterStride * index + i] = float2(NAN);
}
