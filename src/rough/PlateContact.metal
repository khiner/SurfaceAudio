#include "core/ExtendedFloat.h"
using namespace metal;
using namespace surface_audio;
struct PlateConstants {
    uint SlaveX, SlaveY, MasterX, MasterY, SlaveModes, MasterModes, SlaveBase, MasterBase;
    uint BeginX, BeginY, Width, Height, Top, Capacity;
    float2 OffsetX, OffsetY, RatioX, RatioY, Area, Separation, DisplacementBound, MotionX, MotionY;
};
float2 PlateLerp(float2 a, float2 b, float2 t) { return AddExtended(a, MultiplyExtended(t, AddExtended(b, -a))); }
float2 PlateMasterShape(device const float2 *xs, device const float2 *ys, uint k, constant PlateConstants &c, uint ix, uint iy, float2 tx, float2 ty) {
    return MultiplyExtended(PlateLerp(xs[k * c.MasterX + ix], xs[k * c.MasterX + ix + 1], tx), PlateLerp(ys[k * c.MasterY + iy], ys[k * c.MasterY + iy + 1], ty));
}
kernel void PlateContactDetect(constant PlateConstants &c [[buffer(0)]], device const float2 *slave_x [[buffer(1)]], device const float2 *slave_y [[buffer(2)]], device const float2 *master_x [[buffer(3)]], device const float2 *master_y [[buffer(4)]], device const float2 *slave_height [[buffer(5)]], device const float2 *master_height [[buffer(6)]], device const float2 *state [[buffer(7)]], device float2 *rows [[buffer(8)]], device atomic_uint *count [[buffer(9)]], device uint *nodes [[buffer(10)]], uint2 node [[thread_position_in_grid]]) {
    if (node.x >= c.Width || node.y >= c.Height) return;
    const uint x = node.x + c.BeginX, y = node.y + c.BeginY;
    const float2 cx = AddExtended(c.OffsetX, MultiplyExtended(c.RatioX, {float(x), 0})), cy = AddExtended(c.OffsetY, MultiplyExtended(c.RatioY, {float(y), 0}));
    uint ix = uint(clamp(floor(cx.x + cx.y), 0.f, float(c.MasterX - 2))), iy = uint(clamp(floor(cy.x + cy.y), 0.f, float(c.MasterY - 2)));
    const float2 rx = AddExtended(cx, {-float(ix), 0}), ry = AddExtended(cy, {-float(iy), 0});
    if (ix && rx.x + rx.y < 0) --ix;
    if (iy && ry.x + ry.y < 0) --iy;
    const float2 tx0 = AddExtended(cx, {-float(ix), 0}), ty0 = AddExtended(cy, {-float(iy), 0});
    const float2 tx = tx0.x + tx0.y < 0 ? float2{} : tx0.x + tx0.y > 1 ? float2{1, 0} :
                                                                         tx0;
    const float2 ty = ty0.x + ty0.y < 0 ? float2{} : ty0.x + ty0.y > 1 ? float2{1, 0} :
                                                                         ty0;
    const uint base = iy * c.MasterX + ix;
    const float2 height = AddExtended(slave_height[y * c.SlaveX + x], PlateLerp(PlateLerp(master_height[base], master_height[base + 1], tx), PlateLerp(master_height[base + c.MasterX], master_height[base + c.MasterX + 1], tx), ty));
    float2 gap = AddExtended(c.Separation, -height);
    const float2 lower_gap = AddExtended(gap, -c.DisplacementBound);
    if (lower_gap.x > 0 || (lower_gap.x == 0 && lower_gap.y > 0)) return;
    for (uint k = 0; k < c.SlaveModes; ++k) {
        const float2 shape = MultiplyExtended(slave_x[k * c.SlaveX + x], slave_y[k * c.SlaveY + y]);
        gap = AddExtended(gap, MultiplyExtended(c.Top ? shape : -shape, state[c.SlaveBase + k]));
    }
    for (uint k = 0; k < c.MasterModes; ++k) {
        const float2 shape = PlateMasterShape(master_x, master_y, k, c, ix, iy, tx, ty);
        gap = AddExtended(gap, MultiplyExtended(c.Top ? -shape : shape, state[c.MasterBase + k]));
    }
    if (gap.x > 0 || (gap.x == 0 && gap.y >= 0)) return;
    const uint row = atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
    if (row >= c.Capacity) return;
    nodes[row] = (c.Top ? 0 : c.MasterX * c.MasterY) + y * c.SlaveX + x;
    device float2 *record = rows + ulong(row) * (c.SlaveModes + c.MasterModes + 2);
    record[0] = -gap;
    const float quadrature = (x == 0 || x + 1 == c.SlaveX ? .5f : 1.f) * (y == 0 || y + 1 == c.SlaveY ? .5f : 1.f);
    record[1] = MultiplyExtended(c.Area, {quadrature, 0});
    for (uint k = 0; k < c.SlaveModes; ++k) {
        const float2 shape = MultiplyExtended(slave_x[k * c.SlaveX + x], slave_y[k * c.SlaveY + y]);
        record[2 + c.SlaveBase + k] = c.Top ? shape : -shape;
    }
    for (uint k = 0; k < c.MasterModes; ++k) {
        const float2 shape = PlateMasterShape(master_x, master_y, k, c, ix, iy, tx, ty);
        record[2 + c.MasterBase + k] = c.Top ? -shape : shape;
    }
}

kernel void PlateContactCandidates(constant PlateConstants &c [[buffer(0)]], device const float2 *slave_height [[buffer(5)]], device const float2 *master_height [[buffer(6)]], device atomic_uint *count [[buffer(9)]], device uint *nodes [[buffer(10)]], uint2 node [[thread_position_in_grid]]) {
    if (node.x >= c.Width || node.y >= c.Height) return;
    const uint x = node.x + c.BeginX, y = node.y + c.BeginY;
    const float2 cx = AddExtended(c.OffsetX, MultiplyExtended(c.RatioX, {float(x), 0})), cy = AddExtended(c.OffsetY, MultiplyExtended(c.RatioY, {float(y), 0}));
    const float ex = 0x1p-20f * (abs(cx.x) + abs(cx.y) + abs(c.MotionX.x) + abs(c.MotionX.y) + 1);
    const float ey = 0x1p-20f * (abs(cy.x) + abs(cy.y) + abs(c.MotionY.x) + abs(c.MotionY.y) + 1);
    const uint x0 = uint(clamp(floor(cx.x + cx.y - c.MotionX.x - c.MotionX.y - ex), 0.f, float(c.MasterX - 1)));
    const uint x1 = uint(clamp(ceil(cx.x + cx.y + c.MotionX.x + c.MotionX.y + ex), 0.f, float(c.MasterX - 1)));
    const uint y0 = uint(clamp(floor(cy.x + cy.y - c.MotionY.x - c.MotionY.y - ey), 0.f, float(c.MasterY - 1)));
    const uint y1 = uint(clamp(ceil(cy.x + cy.y + c.MotionY.x + c.MotionY.y + ey), 0.f, float(c.MasterY - 1)));
    float maximum = -INFINITY, magnitude = 0;
    for (uint iy = y0; iy <= y1; ++iy)
        for (uint ix = x0; ix <= x1; ++ix) {
            const float2 h = master_height[iy * c.MasterX + ix];
            maximum = max(maximum, h.x + abs(h.y));
            magnitude = max(magnitude, abs(h.x) + abs(h.y));
        }
    const float2 h = slave_height[y * c.SlaveX + x];
    const float bound = h.x + abs(h.y) + maximum + c.DisplacementBound.x + c.DisplacementBound.y;
    const float roundoff = 1e-5f * (magnitude + abs(h.x) + abs(h.y) + abs(c.DisplacementBound.x) + abs(c.DisplacementBound.y)) + 1e-30f;
    if (bound < -roundoff) return;
    const uint row = atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
    if (row < c.Capacity) nodes[row] = (c.Top ? 0 : c.MasterX * c.MasterY) + y * c.SlaveX + x;
}
