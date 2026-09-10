// SPDX-License-Identifier: GPL-3.0-only
#include "core/FiniteDifference.h"
#include "matusiak/DistributedGpuTypes.h"
#include "matusiak/Solve.h"
using namespace surface_audio::matusiak;
float MatD2(device const float *u, uint i, uint n) { return surface_audio::SecondDifference(i ? u[i - 1] : 0, u[i], i + 1 < n ? u[i + 1] : 0); }
float MatD4(device const float *u, uint i, uint n) { return (i ? MatD2(u, i - 1, n) : 0) - 2 * MatD2(u, i, n) + (i + 1 < n ? MatD2(u, i + 1, n) : 0); }
float MatDot4(device const float *a, device const float *b, uint offset) {
    float sum{};
    for (uint i = offset; i < offset + 4; ++i) sum += a[i] * b[i];
    return sum;
}
kernel void MatusiakDistributed(constant DistributedConstants &c [[buffer(0)]], device const BowDrive *drives [[buffer(1)]], device const float *matrices [[buffer(2)]], device float *states [[buffer(3)]], device float *output [[buffer(4)]], device float *residual_out [[buffer(5)]], device uint *failures [[buffer(6)]], uint voice [[thread_position_in_grid]]) {
    if (voice >= c.Voices) return;
    const uint n = c.Nodes, nt = c.TorsionNodes, m = c.Contacts;
    const float k = c.Step, h = c.Spacing, ht = c.TorsionSpacing;
    const auto drive = drives[voice];
    device const float *iu = matrices, *iw = iu + m * n, *coupling = iw + m * nt;
    device float *u = states + voice * c.StateStride, *up = u + n, *su = up + n, *w = su + n, *wp = w + nt, *sw = wp + nt, *hair = sw + nt, *hp = hair + m, *z = hp + m, *v = z + m, *mid = v + m;
    float rhs[8], sh[8], force[8], jacobian[64], delta[8], rate_v[8], inverse_z[8], force_z[8], bristle_residual[8];
    // Cubic stencils occupy at most four adjacent entries.
    uint offset_u[8], offset_w[8];
    for (uint j = 0; j < m; ++j) {
        uint a = 0, b = 0;
        while (a + 4 < n && iu[j * n + a] == 0) ++a;
        while (b + 4 < nt && iw[j * nt + b] == 0) ++b;
        offset_u[j] = a;
        offset_w[j] = b;
    }
    float maximum_residual{};
    uint failed{};
    output[voice * c.Frames] = 0;
    const float ramp = max(1.0f, ceil(drive.Velocity / max(drive.Acceleration, 1e-30f) / k) - 1);
    for (uint frame = 1; frame < c.Frames; ++frame) {
        const float vb = drive.Acceleration > 0 ? min(drive.Velocity, drive.Velocity * float(frame) / ramp) : drive.Velocity;
        for (uint i = 0; i < n; ++i) su[i] = c.XS * (c.WaveSpeed * c.WaveSpeed / (h * h) * MatD2(u, i, n) - c.Bending / (c.Density * h * h * h * h) * MatD4(u, i, n) + 2 * c.Damping1 / (k * h * h) * (MatD2(u, i, n) - MatD2(up, i, n)) + 2 / (k * k) * (u[i] - up[i]));
        for (uint i = 0; i < nt; ++i) sw[i] = c.XT * (c.TorsionSpeed * c.TorsionSpeed / (ht * ht) * MatD2(w, i, nt) + 2 / (k * k) * (w[i] - wp[i]));
        for (uint i = 0; i < m; ++i) {
            sh[i] = c.XH * ((k * c.HairStiffness / 2 + 2 * c.HairMass / k) * (hair[i] - hp[i]) / k - c.HairStiffness * hair[i]);
            rhs[i] = MatDot4(iu + i * n, su, offset_u[i]) - c.Radius * h / ht * MatDot4(iw + i * nt, sw, offset_w[i]) + sh[i] - vb;
        }
        float error{};
        bool converged = false;
        for (uint iteration = 0; iteration < 100; ++iteration) {
            error = 0;
            // Eliminate the diagonal bristle block before solving the coupled slip system.
            for (uint j = 0; j < m; ++j) {
                const auto f = Evaluate(c.Friction, drive.NormalForce, mid[j], v[j]);
                force[j] = f.Force;
                bristle_residual[j] = f.Rate - 2 / k * (mid[j] - z[j]);
                error = max(error, abs(bristle_residual[j]));
                rate_v[j] = f.RateVelocity;
                inverse_z[j] = 1 / (f.RateBristle - 2 / k);
                force_z[j] = c.Friction.Stiffness + f.Damping * f.RateBristle;
                const float derivative = f.Damping * f.RateVelocity + f.DampingVelocity * f.Rate - force_z[j] * inverse_z[j] * rate_v[j];
                for (uint i = 0; i < m; ++i) jacobian[i * m + j] = (i == j ? 1 : 0) + coupling[i * m + j] * derivative;
            }
            for (uint i = 0; i < m; ++i) {
                float residual = v[i] - rhs[i], correction{};
                for (uint j = 0; j < m; ++j) {
                    residual += coupling[i * m + j] * force[j];
                    correction += coupling[i * m + j] * force_z[j] * inverse_z[j] * bristle_residual[j];
                }
                error = max(error, abs(residual));
                delta[i] = residual - correction;
            }
            if (error < 2e-6f) {
                converged = true;
                break;
            }
            Solve(jacobian, delta, m, 0.f);
            const float scale = iteration > 50 ? 1 / 1.1f : 1;
            for (uint i = 0; i < m; ++i) {
                mid[i] -= scale * (bristle_residual[i] - rate_v[i] * delta[i]) * inverse_z[i];
                v[i] -= scale * delta[i];
            }
        }
        maximum_residual = max(maximum_residual, error);
        failed += !converged;
        for (uint j = 0; j < m; ++j) {
            const auto f = Evaluate(c.Friction, drive.NormalForce, mid[j], v[j]);
            force[j] = c.Friction.Stiffness * mid[j] + f.Damping * 2 / k * (mid[j] - z[j]);
            z[j] = 2 * mid[j] - z[j];
            const float next = hp[j] + 2 * k * (sh[j] - c.XH * force[j] / m);
            hp[j] = hair[j];
            hair[j] = next;
        }
        for (uint i = 0; i < n; ++i) {
            float f{};
            for (uint j = 0; j < m; ++j) f += iu[j * n + i] * force[j] / (m * c.Density * h);
            up[i] += 2 * k * (su[i] - c.XS * f);
        }
        for (uint i = 0; i < nt; ++i) {
            float f{};
            for (uint j = 0; j < m; ++j) f += iw[j * nt + i] * force[j] * c.Radius / (m * c.PolarInertia * ht);
            wp[i] += 2 * k * (sw[i] + c.XT * f);
        }
        device float *tmp = u;
        u = up;
        up = tmp;
        tmp = w;
        w = wp;
        wp = tmp;
        const float bridge = c.Tension / h * u[0] - c.Bending / (h * h * h) * (u[1] - 2 * u[0]);
        output[voice * c.Frames + frame] = bridge;
        failed += !isfinite(bridge);
    }
    residual_out[voice] = maximum_residual;
    failures[voice] = failed;
}
