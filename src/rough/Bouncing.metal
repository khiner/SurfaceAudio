#include <metal_stdlib>
#include "core/Random.h"
#include "rough/BouncingDynamics.h"
#include "rough/ContactGpuTypes.h"
using namespace metal;
using namespace surface_audio;
using namespace surface_audio::rough;
struct alignas(128) BounceGpuState {
    RandomState Random;
    float3 Filter;
    BouncingState<float> Ball;
    PlateMotion<float> Plate;
    uint Flights;
    float Compensation[5];
    float LastDepartureHeight, DepartureChange;
    bool HasDeparture, HasDeparturePair;
};
static_assert(sizeof(BounceGpuState) == 128);
void Accumulate(thread float &sum, float value, thread float &correction) {
    const float increment = value - correction, next = sum + increment;
    correction = (next - sum) - increment;
    sum = next;
}
void RecordDeparture(thread BounceGpuState &s, float height) {
    s.DepartureChange = height - s.LastDepartureHeight;
    s.HasDeparturePair = s.HasDeparture;
    s.LastDepartureHeight = height;
    s.HasDeparture = true;
}
kernel void CorrelatedBouncing(constant BounceConstants &c [[buffer(0)]], device BounceGpuState *states [[buffer(1)]],
                               device BounceStatistics *statistics [[buffer(2)]], device float4 *trace [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index >= c.Streams) return;
    auto s = c.Start ? states[index] : BounceGpuState{};
    if (!c.Start) s.Random = MakeIndexedRandom(c.Seed, index);
    auto total = c.Start ? statistics[index] : BounceStatistics{};
    for (uint step = 0; step < c.Steps; ++step) {
        const ulong sample = c.Start + step;
        if (s.Flights >= c.Flights && (index || sample >= c.TraceSamples)) break;
        const auto plate = AdvancePlate(c, s.Filter, Normal(s.Random));
        if (s.Ball.Stuck && s.Plate.Acceleration < -c.Gravity) RecordDeparture(s, s.Plate.Height);
        const auto event = AdvanceBouncing(s.Ball, s.Plate, plate, c.Dt, c.Gravity, c.Restitution, c.StickVelocity, c.RelativeSticking);
        s.Plate = plate;
        if (!index && sample < c.TraceSamples) trace[sample] = {plate.Height, plate.Velocity, s.Ball.Height, s.Ball.Velocity};
        const bool collect = s.Flights >= c.Discard && s.Flights < c.Flights;
        if (collect) {
            ++total.Steps;
            Accumulate(total.HeightSquare, plate.Height * plate.Height, s.Compensation[2]);
            Accumulate(total.VelocitySquare, plate.Velocity * plate.Velocity, s.Compensation[3]);
            Accumulate(total.AccelerationSquare, plate.Acceleration * plate.Acceleration, s.Compensation[4]);
            total.Sticking += s.Ball.Stuck;
        }
        if (event.Completed) {
            if (collect) {
                const float value = event.Velocity * c.VelocityScale;
                ++total.Flights;
                Accumulate(total.VelocitySum, value, s.Compensation[0]);
                Accumulate(total.FlightTimeSum, event.Duration, s.Compensation[1]);
                total.TakeoffPairs += s.HasDeparturePair;
                total.Descending += s.HasDeparturePair && s.DepartureChange < 0;
                total.NegativeVelocity += value < 0;
                const int bin = int(floor((value + 5) * 10));
                if (bin >= 0 && bin < 200) ++total.Histogram[bin];
            }
            ++s.Flights;
            if (!s.Ball.Stuck) RecordDeparture(s, s.Ball.TakeoffHeight);
        }
    }
    states[index] = s;
    statistics[index] = total;
}
