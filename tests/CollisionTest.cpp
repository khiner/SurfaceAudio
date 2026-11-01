#include "collision/Collision.h"
#include "Require.h"
#include <algorithm>
#include <array>
#include <iostream>
using namespace surface_audio;
using namespace surface_audio::collision;

Model<double> CoupledModel(unsigned n = 2) {
    Model<double> m;
    m.Config = {n, 2, n, 0, 3, 0, 1./44100};
    const double dt2 = m.Config.Dt * m.Config.Dt;
    m.Data.resize(4*n);
    for (unsigned i = 0; i < n; ++i) m.Data[i] = .01;
    m.Data[1] = .02; m.Data[2*n] = 1;
    for (double x : {0., 0., 1e8, 2e8, 1.3, 1.2, 1., 0., 0., 0.}) m.Data.push_back(x);
    for (unsigned i = 0; i < n; ++i) m.Data.push_back(20+10*i);
    for (double x : {1., -1., -1., 150*dt2, -100*dt2, -100*dt2, 100*dt2}) m.Data.push_back(x);
    for (unsigned i = 0; i <= n; ++i) m.Indices.push_back(i);
    for (unsigned i = 0; i < n; ++i) m.Indices.push_back(i);
    m.Indices.insert(m.Indices.end(), n+1, 0);
    for (unsigned i : {0, 2, 3, 0, 1, 0}) m.Indices.push_back(i);
    m.State.resize(StateSize(m.Config));
    m.State[0] = -.001; m.State[1] = .0005;
    m.State[n] = .00001; m.State[n+1] = .00001;
    for (unsigned i = 2; i < n; ++i) m.State[i] = .001*std::sin(i);
    m.State[3*n] = std::sqrt(2e8/2.3*std::pow(.0015, 2.3));
    m.State[3*n+1] = std::sqrt(4e8/2.2*std::pow(.00101, 2.2));
    return m;
}

int main() {
    try {
        const auto model = CoupledModel();
        auto state = model.State;
        // Solve the full mechanical 2x2 system independently of contact-space elimination.
        const double dt = model.Config.Dt, masses[]{.01, .02}, stiffness[]{20, 30}, j[2][2]{{1, -1}, {-1, 0}};
        double a[2][2]{{masses[0], 0}, {0, masses[1]}}, rhs[2], g[2];
        for (unsigned i = 0; i < 2; ++i) rhs[i] = masses[i]*(state[i]+state[i+2])-dt*dt*stiffness[i]*state[i];
        for (unsigned c = 0; c < 2; ++c) {
            const double eta = j[c][0]*state[0]+j[c][1]*state[1], before = j[c][0]*(state[0]-state[2])+j[c][1]*(state[1]-state[3]);
            g[c] = std::sqrt(.5*model.Data[10+c]*(model.Data[12+c]+1))*std::pow(std::abs(eta), (model.Data[12+c]-1)/2)*(eta < 0 ? -1 : 1);
            for (unsigned i = 0; i < 2; ++i) {
                rhs[i] += dt*dt*j[c][i]*(g[c]*g[c]*before/4-g[c]*state[6+c]);
                for (unsigned k = 0; k < 2; ++k) a[i][k] += dt*dt*j[c][i]*g[c]*g[c]*j[c][k]/4;
            }
        }
        const double determinant = a[0][0]*a[1][1]-a[0][1]*a[1][0];
        const std::array expected{(rhs[0]*a[1][1]-rhs[1]*a[0][1])/determinant, (a[0][0]*rhs[1]-a[1][0]*rhs[0])/determinant};
        std::vector<Sample<double>> output(2000);
        Render<double>(model, state, output);
        auto first = model.State;
        Render<double>(model, first, std::span(output).first(1));
        for (unsigned i = 0; i < 2; ++i) Require(std::abs(first[i]-expected[i]) < 1e-15, "Contact elimination differs from full mechanical solve");
        for (const auto &s : output) Require(std::abs(s.Energy/output[0].Energy-1) < 1e-9, "Collision energy drift");
        auto streamed = model.State;
        std::vector<Sample<double>> split(2000);
        Render<double>(model, streamed, std::span(split).first(137));
        Render<double>(model, streamed, std::span(split).subspan(137));
        Require(state == streamed, "Collision state depends on block boundaries");
        auto rest = model.State;
        std::fill(rest.begin(), rest.end(), 0);
        Render<double>(model, rest, std::span(output).first(2));
        Require(output[1].Energy == 0 && output[1].Output == 0, "Rest must remain finite and silent");
        Model<double> bowed;
        bowed.Config = {1, 0, 1, 0, 0, 0, 1./44100, 1, -.2, 100};
        bowed.Data = {1e-5, 0, 1, 1, 10};
        bowed.Indices = {0, 1, 0, 0, 0, 0};
        bowed.State.resize(StateSize(bowed.Config));
        auto bow_state = bowed.State;
        Render<double>(bowed, bow_state, output);
        for (const auto &s : output) Require(std::abs(s.Balance) < 1e-12, "Implicit bow work balance failed");
        auto free = FloatModel(bowed);
        free.Config.BowForce = 0;
        free.Data[3] = 0; free.Data[4] = 0;
        free.State[0] = .5f; free.State[1] = 1e-5f;
        const float increment = free.State[1];
        std::vector<Sample<float>> flight(100000);
        Render<float>(free, free.State, flight);
        Require(std::abs(free.State[1]/increment-1) < 1e-5f, "Free-flight velocity loses precision at large displacement");
        const auto floating = FloatModel(CoupledModel(64));
        auto float_state = floating.State;
        std::vector<Sample<float>> cpu(64);
        Render<float>(floating, float_state, cpu);
        auto gpu = CreateGpu();
        auto source = CreateGpuSource(gpu, floating, 3, 32);
        for (unsigned block = 0; block < 2; ++block) {
            BeginGpu(gpu); EncodeSource(gpu, source); SubmitGpu(gpu); WaitGpu(gpu);
            const auto samples = BufferSpan<Sample<float>>(source.Output);
            for (unsigned voice = 0; voice < 3; ++voice)
                for (unsigned frame = 0; frame < 32; ++frame) {
                    const auto &s = samples[voice*32+frame], &reference = cpu[block*32+frame];
                    Require(std::isfinite(s.Energy) && std::abs(s.Energy/reference.Energy-1) < 1e-4f, "Metal collision energy mismatch");
                    Require(std::abs(s.Output-reference.Output) < 2e-7f, "Metal collision trajectory mismatch");
                }
        }
        const auto gpu_state = BufferSpan<float>(source.State);
        for (unsigned voice = 0; voice < 3; ++voice)
            for (unsigned i = 0; i < 2*floating.Config.Dofs; ++i)
                Require(std::abs(gpu_state[voice*StateSize(floating.Config)+i]-float_state[i]) < 1e-6f, "Parallel Metal coordinate state mismatch");
        std::cout << "Collision: full-system equivalence, energy, rest, streaming and three Metal voices passed\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
