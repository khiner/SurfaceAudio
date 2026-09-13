#include "ContactEvents.h"
#include "core/ExtendedFloat.h"
using namespace metal;
using namespace surface_audio;
using namespace surface_audio::rough;
struct RoughConstants {
    uint BottomNodes, TopNodes, BottomModes, TopModes, Steps, OutputOffset, ShortEventSamples, EventStride, DisplacementRefresh, Padding;
    float2 Offset, Increment, Separation, Stiffness, InverseTimeStep, InverseLength, TimeStep;
};
struct RoughMap {
    uint4 Node;
    float2 Weight[4];
    uint Active;
    uint Padding[3];
};
struct RoughJob {
    RoughConstants Parameters;
    device const float2 *Shapes, *Heights, *Coefficients;
    device float2 *State, *Displacement;
    device RoughMap *Maps;
    device float2 *Forces;
    device const float2 *Receiver;
    device float *Output;
    device uint *Active;
    device NodeContactEvent<float2> *Events;
    device atomic_uint *Statistics;
};
float2 RoughReduce(float2 value, uint lane) {
    for (uint shift = 16; shift; shift >>= 1) {
        const float2 other{simd_shuffle_down(value.x, shift), simd_shuffle_down(value.y, shift)};
        if (lane + shift < 32) value = AddExtended(value, other);
    }
    return value;
}
float2 RoughSample(device const float2 *values, device const RoughMap &map) {
    float2 value{};
    for (uint i = 0; i < 4; ++i) value = AddExtended(value, MultiplyExtended(map.Weight[i], values[map.Node[i]]));
    return value;
}
uint2 RoughWindow(uint nodes, uint other_nodes, float offset) {
    return uint2(clamp(floor(offset) - 2, 0.f, float(nodes)), clamp(ceil(offset + float(other_nodes - 1)) + 3, 0.f, float(nodes)));
}
uint RoughNode(uint index, uint2 bottom, uint2 top, uint bottom_nodes) {
    const uint count = bottom.y - bottom.x;
    return index < count ? bottom.x + index : bottom_nodes + top.x + index - count;
}
uint2 RoughUnion(uint2 a, uint2 b) {
    if (a.x == a.y) return b;
    if (b.x == b.y) return a;
    return {min(a.x, b.x), max(a.y, b.y)};
}
void RoughSortContacts(device uint *active, uint count, uint tid, uint lanes) {
    if (count < 2) return;
    if (count <= 32) {
        if (tid < 32) {
            uint key = tid < count ? active[tid] : 0xffffffffu;
            for (uint width = 2; width <= 32; width *= 2)
                for (uint stride = width / 2; stride; stride /= 2) {
                    const uint other = simd_shuffle_xor(key, stride);
                    key = bool(tid & width) == bool(tid & stride) ? min(key, other) : max(key, other);
                }
            if (tid < count) active[tid] = key;
        }
    } else {
        const uint padded = 1u << (32 - clz(count - 1));
        for (uint i = count + tid; i < padded; i += lanes) active[i] = 0xffffffffu;
        threadgroup_barrier(mem_flags::mem_device);
        for (uint width = 2; width <= padded; width *= 2)
            for (uint stride = width / 2; stride; stride /= 2) {
                for (uint i = tid; i < padded; i += lanes) {
                    const uint other = i ^ stride;
                    if (other > i && ((i & width) ? active[i] < active[other] : active[i] > active[other])) {
                        const uint value = active[i];
                        active[i] = active[other];
                        active[other] = value;
                    }
                }
                threadgroup_barrier(mem_flags::mem_device);
            }
    }
    threadgroup_barrier(mem_flags::mem_device);
}
float2 RoughNodalForce(uint node, constant RoughConstants &c, float2 offset, uint2 bottom, uint2 top, device const RoughMap *maps, device const float2 *forces) {
    const bool upper = node >= c.BottomNodes;
    const uint n = upper ? node - c.BottomNodes : node, base = upper ? 0 : c.BottomNodes;
    const uint2 window = upper ? bottom : top;
    const float2 coordinate = AddExtended({float(n), 0}, upper ? offset : -offset);
    const int center = int(floor(coordinate.x + coordinate.y));
    float2 force = forces[node];
    for (int candidate = center - 3; candidate <= center + 3; ++candidate) {
        if (candidate < int(window.x) || candidate >= int(window.y)) continue;
        const uint source = base + uint(candidate);
        if (!maps[source].Active || (forces[source].x == 0 && forces[source].y == 0)) continue;
        for (uint i = 0; i < 4; ++i)
            if (maps[source].Node[i] == n) force = AddExtended(force, MultiplyExtended(maps[source].Weight[i], forces[source]));
    }
    return upper ? force : -force;
}
void RoughIncrement(device atomic_uint *statistics, uint index) {
    if (atomic_fetch_add_explicit(statistics + index, 1, memory_order_relaxed) == 0xffffffffu) atomic_store_explicit(statistics + EventOverflow, 1, memory_order_relaxed);
}
void RoughHistogram(device atomic_uint *statistics, uint histogram, float value, float minimum, float maximum) {
    const float logarithm = log10(value);
    const uint bin = logarithm < minimum ? 0 : logarithm >= maximum ? EventHistogramSize - 1 :
                                                                      1 + min(EventHistogramBins - 1, uint((logarithm - minimum) * EventHistogramBins / (maximum - minimum)));
    RoughIncrement(statistics, EventHistogramStart + histogram * EventHistogramSize + bin);
}
void RoughEventEnd(device NodeContactEvent<float2> &event, device atomic_uint *statistics, constant RoughConstants &c) {
    if (event.LeftCensored) RoughIncrement(statistics, EventLeftCensored);
    else {
        RoughIncrement(statistics, EventCompleted);
        const float peak = event.Peak.x + event.Peak.y, duration = float(event.Frames) * (c.TimeStep.x + c.TimeStep.y);
        RoughHistogram(statistics, 0, peak, EventForceLogMinimum, EventForceLogMaximum);
        RoughHistogram(statistics, 1, duration, EventDurationLogMinimum, EventDurationLogMaximum);
        if (peak < .78f) RoughIncrement(statistics, EventBelowWeight);
        if (peak < 7.8f) RoughIncrement(statistics, EventBelowTenWeights);
        if (peak < 78.f) RoughIncrement(statistics, EventBelowHundredWeights);
        if (!c.ShortEventSamples || event.Frames < c.ShortEventSamples) RoughIncrement(statistics, EventShort);
        if (event.Work.x == 0 && event.Work.y == 0) RoughIncrement(statistics, EventZeroWork);
        else RoughHistogram(statistics, event.Work.x + event.Work.y > 0 ? 2 : 3, abs(event.Work.x + event.Work.y), EventWorkLogMinimum, EventWorkLogMaximum);
    }
    event.Peak = {};
    event.Work = {};
    event.Frames = 0;
    event.LeftCensored = 0;
}
RoughMap RoughInterpolation(uint nodes, float2 coordinate) {
    const float tolerance = 0x1p-45f * nodes;
    const float2 upper = AddExtended(coordinate, {-float(nodes - 1), 0});
    if (coordinate.x + coordinate.y < -tolerance || upper.x + upper.y > tolerance) return {};
    if (coordinate.x + coordinate.y < 0) coordinate = {};
    if (upper.x + upper.y > 0) coordinate = {float(nodes - 1), 0};
    uint left = min(uint(floor(coordinate.x + coordinate.y)), nodes - 2);
    const float2 remainder = AddExtended(coordinate, {-float(left), 0});
    if (left && remainder.x + remainder.y < 0) --left;
    const float2 t = AddExtended(coordinate, {-float(left), 0}), t2 = MultiplyExtended(t, t), t3 = MultiplyExtended(t2, t);
    if (!left || left + 2 >= nodes) return {{left, left + 1, left, left}, {AddExtended({1, 0}, -t), t, {}, {}}, 1, {}};
    return {{left - 1, left, left + 1, left + 2}, {AddExtended(AddExtended(MultiplyExtended({-.5f, 0}, t), t2), MultiplyExtended({-.5f, 0}, t3)), AddExtended(AddExtended({1, 0}, MultiplyExtended({-2.5f, 0}, t2)), MultiplyExtended({1.5f, 0}, t3)), AddExtended(AddExtended(MultiplyExtended({.5f, 0}, t), MultiplyExtended({2, 0}, t2)), MultiplyExtended({-1.5f, 0}, t3)), MultiplyExtended({.5f, 0}, AddExtended(t3, -t2))}, 1, {}};
}
float2 RoughDisplacement(uint node, uint count, uint stride, device const float2 *basis, threadgroup const float2 *q) {
    float2 value{};
    for (uint k = 0; k < count; ++k) value = AddExtended(value, MultiplyExtended(basis[k * stride + node], q[k]));
    return value;
}
float RoughMagnitude(float2 value) {
    return abs(value.x) + abs(value.y);
}
kernel void RoughContactPenalty(constant RoughJob *jobs [[buffer(0)]], uint3 job_index [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint3 threads [[threads_per_threadgroup]]) {
    constant auto &job = jobs[job_index.x];
    constant auto &c = job.Parameters;
    device const float2 *shapes = job.Shapes, *heights = job.Heights, *coefficients = job.Coefficients, *receiver = job.Receiver;
    device float2 *state = job.State, *displacement = job.Displacement, *forces = job.Forces;
    device auto *maps = job.Maps;
    device auto *output = job.Output;
    device auto *active = job.Active;
    device auto *events = job.Events;
    device auto *statistics = job.Statistics;
    threadgroup float2 q[256], previous[256], next[256], totals[32], squares[32];
    threadgroup float2 cached_q[256];
    threadgroup float bounds[2];
    threadgroup atomic_uint contact_count;
    const uint modes = c.BottomModes + c.TopModes, lane = tid % 32, group = tid / 32, lanes = threads.x, groups = lanes / 32;
    device const float2 *top_shapes = shapes + c.BottomModes * c.BottomNodes;
    for (uint k = tid; k < modes; k += lanes) {
        q[k] = state[k];
        previous[k] = state[modes + k];
    }
    uint2 cached_bottom{}, cached_top{};
    bool dense = false;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 0; step < c.Steps; ++step) {
        // Direct evaluation bounds coordinate error at coincident contact endpoints.
        const float2 offset = AddExtended(c.Offset, MultiplyExtended(c.Increment, {float(step), 0}));
        if (!tid) atomic_store_explicit(&contact_count, 0, memory_order_relaxed);
        // Include the Hermite stencil around both ends of the overlap.
        const uint2 bottom = RoughWindow(c.BottomNodes, c.TopNodes, offset.x), top = RoughWindow(c.TopNodes, c.BottomNodes, -offset.x);
        const uint nodes = bottom.y - bottom.x + top.y - top.x;
        const bool refresh = dense || step % c.DisplacementRefresh == 0 || any(bottom != cached_bottom) || any(top != cached_top);
        if (refresh) {
            for (uint index = tid; index < nodes; index += lanes) {
                const uint node = RoughNode(index, bottom, top, c.BottomNodes);
                const bool upper = node >= c.BottomNodes;
                displacement[node] = RoughDisplacement(upper ? node - c.BottomNodes : node, upper ? c.TopModes : c.BottomModes, upper ? c.TopNodes : c.BottomNodes, upper ? top_shapes : shapes, q + (upper ? c.BottomModes : 0));
            }
            for (uint k = tid; k < modes; k += lanes) cached_q[k] = q[k];
            cached_bottom = bottom;
            cached_top = top;
        } else if (tid < 2) {
            float change{}, magnitude{};
            const uint begin = tid ? c.BottomModes : 0, end = tid ? modes : c.BottomModes;
            for (uint k = begin; k < end; ++k) {
                const float shape = RoughMagnitude(coefficients[5 * k + 4]);
                change += shape * (abs(q[k].x - cached_q[k].x) + abs(q[k].y - cached_q[k].y));
                magnitude += shape * (RoughMagnitude(q[k]) + RoughMagnitude(cached_q[k]));
            }
            // Inflate FP32 bounds beyond 256-term accumulation error and extended-arithmetic rounding, including underflow.
            bounds[tid] = 1.001f * change + 0x1p-16f * magnitude + 0x1p-100f;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        float2 total{};
        for (uint index = tid; index < nodes; index += lanes) {
            const uint node = RoughNode(index, bottom, top, c.BottomNodes);
            const bool top = node >= c.BottomNodes;
            const uint n = top ? node - c.BottomNodes : node, master_nodes = top ? c.BottomNodes : c.TopNodes, base = top ? 0 : c.BottomNodes;
            maps[node] = RoughInterpolation(master_nodes, AddExtended({float(n), 0}, top ? offset : -offset));
            float2 force{};
            if (maps[node].Active) {
                const float2 height = AddExtended(heights[node], RoughSample(heights + base, maps[node]));
                float2 delta = AddExtended(displacement[node], -RoughSample(displacement + base, maps[node]));
                float2 gap = AddExtended(AddExtended(c.Separation, -height), top ? delta : -delta);
                bool separated = false;
                if (!refresh) {
                    float weight{};
                    for (uint i = 0; i < 4; ++i) weight += RoughMagnitude(maps[node].Weight[i]);
                    const float bound = 1.001f * (bounds[top] + weight * bounds[!top]) +
                        0x1p-16f * (RoughMagnitude(c.Separation) + RoughMagnitude(height) + RoughMagnitude(delta));
                    separated = isfinite(bound) && gap.x > bound;
                    if (!separated) {
                        const float2 slave = RoughDisplacement(n, top ? c.TopModes : c.BottomModes, top ? c.TopNodes : c.BottomNodes, top ? top_shapes : shapes, q + (top ? c.BottomModes : 0));
                        float2 master{};
                        for (uint i = 0; i < 4; ++i) {
                            const float2 value = RoughDisplacement(maps[node].Node[i], top ? c.BottomModes : c.TopModes, master_nodes, top ? shapes : top_shapes, q + (top ? 0 : c.BottomModes));
                            master = AddExtended(master, MultiplyExtended(maps[node].Weight[i], value));
                        }
                        delta = AddExtended(slave, -master);
                        gap = AddExtended(AddExtended(c.Separation, -height), top ? delta : -delta);
                    }
                }
                if (!separated && (gap.x < 0 || (gap.x == 0 && gap.y < 0))) {
                    const bool end = n == 0 || n + 1 == (top ? c.TopNodes : c.BottomNodes);
                    force = MultiplyExtended(MultiplyExtended(c.Stiffness, end ? float2{.5f, 0} : float2{1, 0}), -gap);
                }
            }
            forces[node] = force;
            if (force.x != 0 || force.y != 0) active[atomic_fetch_add_explicit(&contact_count, 1, memory_order_relaxed)] = node;
            total = AddExtended(total, force);
        }
        total = RoughReduce(total, lane);
        if (!lane) totals[group] = total;
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        const uint contacts = atomic_load_explicit(&contact_count, memory_order_relaxed);
        dense = contacts > nodes / 8;
        // Fixed node order prevents scheduling-dependent force summation from changing long contact trajectories.
        RoughSortContacts(active, contacts, tid, lanes);
        if (!tid) {
            float2 force{};
            for (uint i = 0; i < groups; ++i) force = AddExtended(force, totals[i]);
            output[3 * (c.OutputOffset + step) + 1] = force.x + force.y;
        }
        for (uint k = group; k < modes; k += groups) {
            const bool top_mode = k >= c.BottomModes;
            const uint local_mode = top_mode ? k - c.BottomModes : k;
            device const float2 *basis = top_mode ? top_shapes + local_mode * c.TopNodes : shapes + local_mode * c.BottomNodes;
            float2 force{};
            for (uint index = lane; index < contacts; index += 32) {
                const uint node = active[index];
                const bool top_node = node >= c.BottomNodes;
                const uint n = top_node ? node - c.BottomNodes : node;
                const float2 shape = top_node == top_mode ? basis[n] : RoughSample(basis, maps[node]);
                force = AddExtended(force, MultiplyExtended(top_mode ? shape : -shape, forces[node]));
            }
            force = RoughReduce(force, lane);
            if (!lane) {
                force = AddExtended(force, coefficients[5 * k + 3]);
                next[k] = AddExtended(q[k], AddExtended(MultiplyExtended(coefficients[5 * k + 1], AddExtended(q[k], -previous[k])), AddExtended(MultiplyExtended(coefficients[5 * k + 2], force), -MultiplyExtended(coefficients[5 * k], q[k]))));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float2 velocity{}, square{};
        for (uint k = tid; k < modes; k += lanes) {
            const float2 v = MultiplyExtended(AddExtended(next[k], -previous[k]), c.InverseTimeStep);
            velocity = AddExtended(velocity, MultiplyExtended(receiver[k], v));
            if (k < c.BottomModes) square = AddExtended(square, MultiplyExtended(v, v));
        }
        velocity = RoughReduce(velocity, lane);
        square = RoughReduce(square, lane);
        if (!lane) {
            totals[group] = velocity;
            squares[group] = square;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!tid) {
            float2 value{}, sum{};
            for (uint i = 0; i < groups; ++i) {
                value = AddExtended(value, totals[i]);
                sum = AddExtended(sum, squares[i]);
            }
            sum = MultiplyExtended(sum, c.InverseLength);
            output[3 * (c.OutputOffset + step)] = value.x + value.y;
            output[3 * (c.OutputOffset + step) + 2] = sum.x + sum.y;
        }
        const bool record_sample = atomic_load_explicit(statistics + EventFrames, memory_order_relaxed) % c.EventStride == 0;
        if (record_sample) {
            const uint2 old_bottom{atomic_load_explicit(statistics + EventBottomBegin, memory_order_relaxed), atomic_load_explicit(statistics + EventBottomEnd, memory_order_relaxed)};
            const uint2 old_top{atomic_load_explicit(statistics + EventTopBegin, memory_order_relaxed), atomic_load_explicit(statistics + EventTopEnd, memory_order_relaxed)};
            const uint2 event_bottom = RoughUnion(bottom, old_bottom), event_top = RoughUnion(top, old_top);
            const uint event_nodes = event_bottom.y - event_bottom.x + event_top.y - event_top.x;
            const bool first_sample = atomic_load_explicit(statistics + EventSamples, memory_order_relaxed) == 0;
            for (uint index = tid; index < event_nodes; index += lanes) {
                const uint node = RoughNode(index, event_bottom, event_top, c.BottomNodes);
                const bool upper = node >= c.BottomNodes;
                const uint n = upper ? node - c.BottomNodes : node, stride = upper ? c.TopNodes : c.BottomNodes, count = upper ? c.TopModes : c.BottomModes;
                const uint2 window = upper ? top : bottom;
                const float2 force = n >= window.x && n < window.y ? RoughNodalForce(node, c, offset, bottom, top, maps, forces) : float2{};
                device auto &event = events[node];
                if (force.x == 0 && force.y == 0) {
                    if (event.Frames) RoughEventEnd(event, statistics + uint(upper) * EventStorageSize, c);
                    continue;
                }
                device const float2 *basis = upper ? top_shapes : shapes;
                float2 velocity{};
                for (uint k = 0; k < count; ++k) {
                    const uint mode = (upper ? c.BottomModes : 0) + k;
                    velocity = AddExtended(velocity, MultiplyExtended(basis[k * stride + n], MultiplyExtended(AddExtended(next[mode], -previous[mode]), c.InverseTimeStep)));
                }
                const float2 work = MultiplyExtended(MultiplyExtended(force, velocity), c.TimeStep);
                const float2 magnitude = force.x + force.y < 0 ? -force : force;
                if (magnitude.x > event.Peak.x || (magnitude.x == event.Peak.x && magnitude.y > event.Peak.y)) event.Peak = magnitude;
                event.Work = AddExtended(event.Work, work);
                event.TotalWork = AddExtended(event.TotalWork, work);
                if (event.Frames == 0xffffffffu) atomic_store_explicit(statistics + EventOverflow, 1, memory_order_relaxed);
                ++event.Frames;
                if (first_sample) event.LeftCensored = 1;
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (!tid) {
            RoughIncrement(statistics, EventFrames);
            if (record_sample) {
                RoughIncrement(statistics, EventSamples);
                atomic_store_explicit(statistics + EventBottomBegin, bottom.x, memory_order_relaxed);
                atomic_store_explicit(statistics + EventBottomEnd, bottom.y, memory_order_relaxed);
                atomic_store_explicit(statistics + EventTopBegin, top.x, memory_order_relaxed);
                atomic_store_explicit(statistics + EventTopEnd, top.y, memory_order_relaxed);
            }
        }
        for (uint k = tid; k < modes; k += lanes) {
            previous[k] = q[k];
            q[k] = next[k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
    for (uint k = tid; k < modes; k += lanes) {
        state[k] = q[k];
        state[modes + k] = previous[k];
    }
}
