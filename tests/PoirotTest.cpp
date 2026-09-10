#include "poirot/Poirot.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
#include <numeric>
#include <stdexcept>
using namespace surface_audio;
using namespace surface_audio::poirot;
static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
int main() {
    try {
        StringParameters string;
        auto modes = StringModes(string);
        Require(std::abs(modes[0].Frequency - std::sqrt(404.02 * 404.02 + 1.297 * 1.297 * 4 * std::numbers::pi * std::numbers::pi)) < .0001, "Equation 8 frequency");
        Require(std::abs(modes[0].Damping - (.05 + .002 * 4 * std::numbers::pi * std::numbers::pi)) < 1e-7, "Equation 8 damping");
        const std::array<float, 3> power{10, 2, 1}, threshold{1, 1, INFINITY}, weights{.4, .6, 0};
        std::array<float, 3> transfer{};
        SignalParameters p{.Activation = 0, .Lambda = .1};
        const auto total = TransferPower(p, power, threshold, weights, transfer);
        Require(std::abs(total - 1) < 1e-6 && std::abs(std::accumulate(transfer.begin(), transfer.end(), 0.f)) < 1e-6, "Recipient weights conserve power");
        Require(transfer[2] == 0, "Nodal mode unaffected");
        // Literal Eq.10, including its inconsistent donor index.
        const double literal_inflow = .1 * (.4 * 9 + .6);
        Require(std::abs(3 * literal_inflow - 1 - .26) < 1e-6, "Printed donor indexing conservation counterexample");
        p.ReturnGain = .3;
        TransferPower(p, power, threshold, weights, transfer);
        Require(std::abs(std::accumulate(transfer.begin(), transfer.end(), 0.f) + .7) < 1e-6, "Choke dissipates unrecovered power");
        for (size_t i = 0; i < modes.size(); ++i) modes[i].Amplitude = 1.f / (i + 1);
        p = {.Activation = .01f, .SplitThreshold = 0, .SplitSlope = 10, .PowerScale = 10000, .ShapeWeightedSplit = true, .ResetPhaseAtActivation = true};
        auto whole = MakeSignal(p, modes), chunks = whole;
        Require(whole.Weight[1] == 0 && std::isinf(whole.Threshold[1]), "Nodal thresholds are unbounded even at zero height");
        const float max_weight = *std::max_element(whole.Weight.begin(), whole.Weight.end());
        Require(max_weight < .1, "Literal normalized Eq16 cannot reach claimed f1/3 shift");
        std::vector<float> a(8192), b(a.size());
        RenderSignal(whole, a);
        for (size_t offset = 0; offset < b.size();) {
            const size_t count = std::min<size_t>(137, b.size() - offset);
            RenderSignal(chunks, std::span(b).subspan(offset, count));
            offset += count;
        }
        Require(a == b, "CPU signal streaming is exact");
        auto gpu = CreateGpu();
        auto initial = MakeSignal(p, modes);
        auto synth = CreateGpuSignal(gpu, std::span(&initial, 1), 1024);
        std::vector<float> g;
        for (int i = 0; i < 8; ++i) {
            BeginGpu(gpu);
            EncodeSignal(gpu, synth, 1024);
            SubmitGpu(gpu);
            WaitGpu(gpu);
            const auto block = BufferSpan<float>(synth.Output);
            g.insert(g.end(), block.begin(), block.end());
        }
        double err = 0, norm = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            err += std::pow(a[i] - g[i], 2);
            norm += a[i] * a[i];
        }
        std::cout << "GPU waveform relative error " << std::sqrt(err / norm) << '\n';
        Require(std::sqrt(err / norm) < .003, "CPU/GPU waveform agreement");
        const std::array<Mode, 1> nodal_modes{{{440, 1, 1, 0}}};
        const SignalParameters nodal_parameters{.Activation = 0, .Height = 0, .Position = 1e-8f, .SplitThreshold = 0, .SplitSlope = 10, .PowerScale = 10000};
        auto nodal = MakeSignal(nodal_parameters, nodal_modes);
        Require(nodal.Weight[0] == 0 && std::isinf(nodal.Threshold[0]), "All-nodal bank preserves zero weights");
        auto nodal_gpu = CreateGpuSignal(gpu, std::span(&nodal, 1), 2048);
        std::array<float, 2048> nodal_cpu{};
        RenderSignal(nodal, nodal_cpu);
        BeginGpu(gpu);
        EncodeSignal(gpu, nodal_gpu, 2048);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        const auto nodal_output = BufferSpan<float>(nodal_gpu.Output);
        for (size_t frame = 0; frame < nodal_cpu.size(); ++frame) {
            const double time = double(frame) / nodal_parameters.SampleRate;
            const double expected = std::exp(-time) * std::sin(2 * std::numbers::pi * 440 * time);
            Require(std::isfinite(nodal_cpu[frame]) && std::isfinite(nodal_output[frame]), "All-nodal CPU/GPU output is finite");
            Require(std::abs(nodal_cpu[frame] - expected) < 1e-4 && std::abs(nodal_output[frame] - expected) < 1e-4, "All-nodal CPU/GPU bank matches independent uncoupled oscillator");
        }
        auto queued_initial = MakeSignal(p, modes), queued_cpu = queued_initial;
        auto queued_gpu = CreateGpuSignal(gpu, std::span(&queued_initial, 1), 1024);
        std::array<float, 257> first_cpu{};
        std::array<float, 509> second_cpu{};
        RenderSignal(queued_cpu, first_cpu);
        RenderSignal(queued_cpu, second_cpu);
        BeginGpu(gpu);
        EncodeSignal(gpu, queued_gpu, first_cpu.size());
        EncodeSignal(gpu, queued_gpu, second_cpu.size());
        SubmitGpu(gpu);
        WaitGpu(gpu);
        const auto queued_output = BufferSpan<float>(queued_gpu.Output);
        const auto queued_state = BufferSpan<float>(queued_gpu.State);
        for (size_t frame = 0; frame < second_cpu.size(); ++frame) Require(std::abs(queued_output[frame] - second_cpu[frame]) < .001, "Two queued encodes crossing activation/reset preserve each dispatch frame range");
        Require(queued_gpu.Frame == queued_cpu.Frame, "Queued dispatch host frame count");
        for (size_t mode = 0; mode < modes.size(); ++mode) {
            Require(std::abs(queued_state[3 * mode] - queued_cpu.Power[mode]) < 3e-5 * std::max(queued_cpu.Power[mode], 1e-6f), "Queued dispatch final power state");
            Require(std::abs(std::remainder(queued_state[3 * mode + 1] - queued_cpu.UpperPhase[mode], float(2 * std::numbers::pi))) < .001, "Queued dispatch final upper phase");
            Require(std::abs(std::remainder(queued_state[3 * mode + 2] - queued_cpu.LowerPhase[mode], float(2 * std::numbers::pi))) < .001, "Queued dispatch final lower phase");
        }
        auto late_parameters = p;
        late_parameters.Activation = 1000;
        auto late_cpu = MakeSignal(late_parameters, modes);
        late_cpu.Frame = 44100000 - 2;
        auto late_gpu = CreateGpuSignal(gpu, std::span(&late_cpu, 1), 6);
        std::array<float, 6> late_output{};
        RenderSignal(late_cpu, late_output);
        BeginGpu(gpu);
        EncodeSignal(gpu, late_gpu, late_output.size());
        SubmitGpu(gpu);
        WaitGpu(gpu);
        const auto late_device = BufferSpan<float>(late_gpu.Output);
        for (size_t frame = 0; frame < late_output.size(); ++frame) Require(std::abs(late_device[frame] - late_output[frame]) < 1e-5, "Activation remains sample-exact beyond float integer precision");
        auto out_of_range = initial;
        out_of_range.Frame = uint64_t(UINT32_MAX) + 1;
        bool rejected = false;
        try {
            CreateGpuSignal(gpu, std::span(&out_of_range, 1), 1024);
        } catch (const std::invalid_argument &) { rejected = true; }
        Require(rejected, "GPU initial frame cannot truncate to 32 bits");
        queued_gpu.Frame = UINT64_MAX;
        rejected = false;
        try {
            EncodeSignal(gpu, queued_gpu, 1);
        } catch (const std::invalid_argument &) { rejected = true; }
        Require(rejected, "GPU frame range rejects addition overflow");
        for (double old : {-.001, 0., .0001, .003})
            for (double next : {-.002, 0., .0002, .002}) {
                const double work = ContactGradient(next, old, 5e10, 1.4) * (next - old), potential = ContactPotential(next, 5e10, 1.4) - ContactPotential(old, 5e10, 1.4);
                Require(std::abs(work - potential) < 1e-8 * std::max(1., std::abs(potential)), "Discrete gradient independent potential-work identity");
            }
        string.WaveSpeed = 410.6;
        string.Activation = .5;
        string.ObstacleHeight = 0;
        auto physical = MakeString(string), physical_chunks = physical;
        std::vector<float> contact(32768), physical_a(32768), physical_b(32768);
        RenderString(physical, physical_a, contact);
        RenderString(physical_chunks, std::span(physical_b).first(3001));
        RenderString(physical_chunks, std::span(physical_b).subspan(3001));
        Require(physical_a == physical_b, "Physical string streaming is exact");
        Require(std::ranges::all_of(physical_a, [](float x) { return std::isfinite(x) && std::abs(x) < 1; }), "Stiff contact remains bounded");
        Require(std::count_if(contact.begin(), contact.end(), [](float x) { return x > 0; }) > 100, "Repeated collision texture beyond isolated impact");
        Require(physical.MaximumPenetration < .001, "Stiff obstacle small penetration");
        string.ContactStiffness = 0;
        string.ExcitationForce = 0;
        string.Loss0 = string.Loss1 = 0;
        auto eigen = MakeString(string);
        const double q = 2 * std::sin(std::numbers::pi / (2 * eigen.Segments)) / eigen.Spacing;
        const double omega_k = 2 * std::asin(std::sqrt(string.WaveSpeed * string.WaveSpeed * q * q + string.Stiffness * string.Stiffness * std::pow(q, 4)) / (2 * string.SampleRate));
        for (uint32_t i = 1; i < eigen.Segments; ++i) {
            eigen.Current[i] = std::sin(std::numbers::pi * i / eigen.Segments);
            eigen.Previous[i] = eigen.Current[i] * std::cos(omega_k);
        }
        std::array<float, 1000> eigen_out{};
        RenderString(eigen, eigen_out);
        const double at = string.ReadoutPosition * eigen.Segments, fraction = at - std::floor(at);
        const double shape = (1 - fraction) * std::sin(std::numbers::pi * std::floor(at) / eigen.Segments) + fraction * std::sin(std::numbers::pi * (std::floor(at) + 1) / eigen.Segments);
        for (size_t i = 0; i < eigen_out.size(); ++i) Require(std::abs(eigen_out[i] - shape * std::cos((i + 1) * omega_k)) < 1e-6, "Discrete string eigenfrequency independent oracle");
        // Independent staggered discrete energy, including the averaged contact potential.
        string.Activation = 0;
        string.ContactStiffness = 5e10;
        string.ObstacleHeight = 0;
        auto conserved = MakeString(string);
        for (uint32_t i = 1; i < conserved.Segments; ++i) {
            conserved.Current[i] = -.0001 * std::sin(std::numbers::pi * i / conserved.Segments);
            conserved.Previous[i] = conserved.Current[i] * std::cos(omega_k);
        }
        auto energy = [](const StringState &z) {
            const auto &p = z.Parameters;
            const double h = z.Spacing, mass = h * p.Density * p.Area;
            double value = 0;
            for (uint32_t i = 1; i < z.Segments; ++i) {
                value += .5 * mass * std::pow((z.Current[i] - z.Previous[i]) * p.SampleRate, 2);
                const double d2 = (z.Current[i - 1] - 2 * z.Current[i] + z.Current[i + 1]) / (h * h);
                const double d2p = (z.Previous[i - 1] - 2 * z.Previous[i] + z.Previous[i + 1]) / (h * h);
                value += .5 * mass * p.Stiffness * p.Stiffness * d2 * d2p;
            }
            for (uint32_t i = 0; i < z.Segments; ++i) value += .5 * mass * p.WaveSpeed * p.WaveSpeed * (z.Current[i + 1] - z.Current[i]) * (z.Previous[i + 1] - z.Previous[i]) / (h * h);
            const double at = p.ObstaclePosition * z.Segments, fraction = at - std::floor(at);
            const auto index = uint32_t(at);
            for (const auto *u : {&z.Current, &z.Previous}) value += .5 * ContactPotential((1 - fraction) * (*u)[index] + fraction * (*u)[index + 1] - p.ObstacleHeight, p.ContactStiffness, p.ContactExponent);
            return value;
        };
        const double initial_energy = energy(conserved);
        std::array<float, 1> one{};
        for (int frame = 0; frame < 5000; ++frame) {
            RenderString(conserved, one);
            Require(std::abs(energy(conserved) / initial_energy - 1) < 2e-8, "Independent full discrete energy conservation through repeated contacts");
        }
        std::cout << "Poirot equations, conservation, contact, streaming and GPU checks passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
