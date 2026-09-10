// SPDX-License-Identifier: GPL-3.0-only
#include "core/FiniteDifference.h"
#include "core/PivotedSolve.h"
#include "matusiak2024/DistributedGpuTypes.h"
using namespace surface_audio::matusiak2024;
float Mat24D2(device const float *u, uint i, uint n) { return surface_audio::SecondDifference(i ? u[i - 1] : 0, u[i], i + 1 < n ? u[i + 1] : 0); }
float Mat24D4(device const float *u, uint i, uint n) { return (i ? Mat24D2(u, i - 1, n) : 0) - 2 * Mat24D2(u, i, n) + (i + 1 < n ? Mat24D2(u, i + 1, n) : 0); }
float Mat24Dot4(device const float *a, device const float *b, uint offset) {
    float sum{};
    for (uint i = offset; i < offset + 4; ++i) sum += a[i] * b[i];
    return sum;
}
kernel void Matusiak2024Distributed(constant DistributedConstants &c [[buffer(0)]], device const BowDrive *drives [[buffer(1)]], device const float *matrices [[buffer(2)]], device float *states [[buffer(3)]], device float *output [[buffer(4)]], device float *residual_out [[buffer(5)]], device uint *failures [[buffer(6)]], device float *energy_out [[buffer(7)]], device float *balance_out [[buffer(8)]], uint voice [[thread_position_in_grid]]) {
    if (voice >= c.Voices) return;
    const uint n = c.Nodes, nt = c.TorsionNodes, m = c.Contacts;
    const float k = c.Step, h = c.Spacing, ht = c.TorsionSpacing;
    const auto drive = drives[voice];
    device const float *iu = matrices, *iw = iu + m * n, *coupling = iw + m * nt;
    device float *u = states + voice * c.StateStride, *up = u + n, *su = up + n, *w = su + n, *wp = w + nt, *sw = wp + nt, *hair = sw + nt, *hp = hair + m, *z = hp + m, *v = z + m, *mid = v + m;
    float rhs[8], sh[8], force[8], jacobian[64], delta[8], rate_v[8], inverse_z[8], force_z[8], bristle_residual[8];
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
    output[voice * c.Frames] = energy_out[voice * c.Frames] = balance_out[voice * c.Frames] = 0;
    float integrated_loss{};
    for (uint frame = 1; frame < c.Frames; ++frame) {
        const float vb = drive.Acceleration > 0 ? min(drive.Velocity, drive.Acceleration * float(frame) * k) : drive.Velocity;
        for (uint i = 0; i < n; ++i) su[i] = c.XS * (c.WaveSpeed * c.WaveSpeed / (h * h) * Mat24D2(u, i, n) - c.Bending / (c.Density * h * h * h * h) * Mat24D4(u, i, n) + 2 * c.Damping1 / (k * h * h) * (Mat24D2(u, i, n) - Mat24D2(up, i, n)) + 2 / (k * k) * (u[i] - up[i]));
        for (uint i = 0; i < nt; ++i) sw[i] = c.XT * (c.TorsionSpeed * c.TorsionSpeed / (ht * ht) * Mat24D2(w, i, nt) + 2 / (k * k) * (w[i] - wp[i]));
        for (uint i = 0; i < m; ++i) {
            sh[i] = -c.XH * c.HairStiffness * hp[i];
            rhs[i] = Mat24Dot4(iu + i * n, su, offset_u[i]) - c.TorsionFeedback * c.Radius * Mat24Dot4(iw + i * nt, sw, offset_w[i]) + sh[i] - vb;
        }
        float error{};
        bool converged = false;
        for (uint iteration = 0; iteration < 100; ++iteration) {
            if (iteration == 25 || iteration == 50 || iteration == 75) {
                for (uint j = 0; j < m; ++j) {
                    v[j] = rhs[j] * (iteration == 25 ? 1 : iteration == 50 ? .5f :
                                                                             2);
                    const float steady = Evaluate(c.Friction, drive.NormalForce, 0.f, v[j]).Steady;
                    float low = min(min(z[j], z[j] + k / 2 * v[j]), steady), high = max(max(z[j], z[j] + k / 2 * v[j]), steady);
                    for (uint bisection = 0; bisection < 30; ++bisection) {
                        const float candidate = (low + high) / 2;
                        const float residual = Evaluate(c.Friction, drive.NormalForce, candidate, v[j]).Rate - 2 / k * (candidate - z[j]);
                        if (residual > 0) low = candidate;
                        else high = candidate;
                    }
                    mid[j] = (low + high) / 2;
                }
            }
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
            if (!surface_audio::Solve(jacobian, delta, m, 0.f)) break;
            float delta_z[8], trial_force[8];
            for (uint j = 0; j < m; ++j) delta_z[j] = (bristle_residual[j] - rate_v[j] * delta[j]) * inverse_z[j];
            float scale = 1;
            for (uint search = 0; search < 24; ++search) {
                float trial_error{};
                for (uint j = 0; j < m; ++j) {
                    const float trial_z = mid[j] - scale * delta_z[j];
                    const auto trial = Evaluate(c.Friction, drive.NormalForce, trial_z, v[j] - scale * delta[j]);
                    trial_force[j] = trial.Force;
                    trial_error = max(trial_error, abs(trial.Rate - 2 / k * (trial_z - z[j])));
                }
                for (uint j = 0; j < m; ++j) {
                    float r = v[j] - scale * delta[j] - rhs[j];
                    for (uint i = 0; i < m; ++i) r += coupling[j * m + i] * trial_force[i];
                    trial_error = max(trial_error, abs(r));
                }
                if (trial_error < error || trial_error < 2e-6f) break;
                scale *= .5f;
            }
            for (uint j = 0; j < m; ++j) {
                mid[j] -= scale * delta_z[j];
                v[j] -= scale * delta[j];
            }
        }
        maximum_residual = max(maximum_residual, error);
        failed += !converged;
        float loss{};
        for (uint j = 0; j < m; ++j) {
            force[j] = c.Friction.Stiffness * mid[j] + c.Friction.Damping * 2 / k * (mid[j] - z[j]);
            loss += (v[j] * force[j] - c.Friction.Stiffness * mid[j] * 2 / k * (mid[j] - z[j]) + vb * force[j]) / m;
            z[j] = 2 * mid[j] - z[j];
            const float next = hp[j] + 2 * k * (sh[j] - c.XH * force[j] / m);
            const float hv = (next - hp[j]) / (2 * k);
            loss += c.HairDamping * hv * hv;
            hp[j] = hair[j];
            hair[j] = next;
        }
        for (uint i = 0; i < n; ++i) {
            float f{};
            for (uint j = 0; j < m; ++j) f += iu[j * n + i] * force[j] / (m * c.Density * h);
            const float velocity = su[i] - c.XS * f;
            loss += 2 * c.Damping0 * c.Density * h * velocity * velocity - 2 * c.Damping1 * c.Density / h * velocity * (Mat24D2(u, i, n) - Mat24D2(up, i, n)) / k;
            su[i] = up[i] + 2 * k * velocity;
        }
        for (uint i = 0; i < n; ++i) up[i] = su[i];
        for (uint i = 0; i < nt; ++i) {
            float f{};
            for (uint j = 0; j < m; ++j) f += iw[j * nt + i] * force[j] * c.Radius / (m * c.PolarInertia * ht);
            const float velocity = sw[i] + c.XT * f;
            loss += c.TorsionFeedback * 2 * c.TorsionDamping * c.PolarInertia * ht * velocity * velocity;
            wp[i] += 2 * k * velocity;
        }
        device float *tmp = u;
        u = up;
        up = tmp;
        tmp = w;
        w = wp;
        wp = tmp;
        float energy{};
        for (uint i = 0; i < n; ++i) {
            const float velocity = (u[i] - up[i]) / k;
            energy += c.Density * h / 2 * velocity * velocity + c.Bending / (2 * h * h * h) * Mat24D2(u, i, n) * Mat24D2(up, i, n);
        }
        for (uint i = 0; i <= n; ++i) energy += c.Tension / (2 * h) * ((i < n ? u[i] : 0) - (i ? u[i - 1] : 0)) * ((i < n ? up[i] : 0) - (i ? up[i - 1] : 0));
        for (uint i = 0; i < nt; ++i) {
            const float velocity = (w[i] - wp[i]) / k;
            energy += c.TorsionFeedback * c.PolarInertia * ht / 2 * velocity * velocity;
        }
        for (uint i = 0; i <= nt; ++i) energy += c.TorsionFeedback * c.TorsionSpeed * c.TorsionSpeed * c.PolarInertia / (2 * ht) * ((i < nt ? w[i] : 0) - (i ? w[i - 1] : 0)) * ((i < nt ? wp[i] : 0) - (i ? wp[i - 1] : 0));
        for (uint j = 0; j < m; ++j) energy += c.HairStiffness / 4 * (hair[j] * hair[j] + hp[j] * hp[j]) + c.Friction.Stiffness / (2 * m) * z[j] * z[j];
        integrated_loss += k * loss;
        energy_out[voice * c.Frames + frame] = energy;
        balance_out[voice * c.Frames + frame] = energy + integrated_loss;
        const float bridge = c.Tension / h * u[0] - c.Bending / (h * h * h) * (u[1] - 2 * u[0]);
        output[voice * c.Frames + frame] = bridge;
        failed += !isfinite(bridge);
    }
    residual_out[voice] = maximum_residual;
    failures[voice] = failed;
}
