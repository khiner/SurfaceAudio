#include "sdt/Sdt.h"
#include "core/Gpu.h"
#include "core/Random.h"
#include "sdt/SdtGpu.h"
#include "sdt/reference/Upstream.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace surface_audio::sdt;

namespace {
void Check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void Near(double value, double expected, double absolute, double relative, const char *message) {
    if (!std::isfinite(value) || std::abs(value - expected) > absolute + relative * std::abs(expected)) {
        std::cerr << message << ": " << value << " != " << expected << '\n';
        throw std::runtime_error(message);
    }
}

Body Inertial() {
    const std::array modes{ModeParameters{.Mass = 0.03}};
    return MakeBody(modes, 48000);
}

Body Modal() {
    const std::array modes{ModeParameters{230, 0.3, 0.4, 0.9, 0.3}, ModeParameters{690, 0.15, 0.25, 0.4, 0.8}, ModeParameters{1430, 0.08, 0.2, 0.25, 0.2}};
    return MakeBody(modes, 48000);
}

void Reference() {
    const auto path = std::filesystem::path(__FILE__).parent_path().parent_path() / "src/sdt/reference/Trace.f64";
    std::ifstream reference(path, std::ios::binary);
    Check(bool(reference), "Open SDT upstream reference");
    double max_error{};
    for (bool friction : {false, true}) {
        auto body0 = Inertial(), body1 = Modal();
        auto corrected0 = body0, corrected1 = body1;
        ContactState state, corrected_state;
        double force_error{}, force_norm{}, output_error{}, output_norm{};
        FrictionParameters parameters{.NormalForce = 0.7, .Dissipation = 0.015, .Viscosity = 0.01, .Noisiness = 0.0005};
        SetVelocity(body0, friction ? -0.05 : -0.5);
        SetVelocity(corrected0, friction ? -0.05 : -0.5);
        uint32_t random_state = 42;
        for (unsigned frame = 0; frame < 4096; ++frame) {
            random_state = random_state * 1664525u + 1013904223u;
            const double noise = double(random_state) / double(0x7fffffffu) - 1;
            const auto sample = friction ? reference::StepFriction(body0, body1, state, parameters, noise, -0.03) : reference::StepImpact(body0, body1, state, {});
            const auto corrected = friction ? StepFriction(corrected0, corrected1, corrected_state, parameters, noise, -0.03) : StepImpact(corrected0, corrected1, corrected_state, {});
            force_error += std::pow(corrected.Force - sample.Force, 2);
            force_norm += sample.Force * sample.Force;
            output_error += std::pow(corrected.Output1 - sample.Output1, 2);
            output_norm += sample.Output1 * sample.Output1;
            const std::array actual{sample.Force, sample.Position0, sample.Position1, sample.Velocity0, sample.Velocity1, sample.Output1};
            std::array<double, 6> expected;
            reference.read(reinterpret_cast<char *>(expected.data()), sizeof(expected));
            Check(bool(reference), "Read SDT contact reference");
            for (size_t field = 0; field < actual.size(); ++field) {
                max_error = std::max(max_error, std::abs(actual[field] - expected[field]));
                Near(actual[field], expected[field], 2e-10, 2e-8, "SDT coupled full-trace equivalence");
            }
        }
        std::cout << "SDT analytic refinement versus upstream " << (friction ? "friction" : "impact") << " force/output L2: " << std::sqrt(force_error / force_norm) << '/' << std::sqrt(output_error / output_norm) << '\n';
    }
    RollingState rolling;
    ScrapingState scraping;
    for (unsigned frame = 0; frame < 4096; ++frame) {
        const double surface = std::sin(frame * 0.13) + 0.3 * std::cos(frame * 0.71);
        const double rolling_value = StepRolling(RollingParameters{0.07, 2, 0.04, 1.3}, rolling, surface);
        const double scraping_value = StepScraping(ScrapingParameters{0.07, 0.8, 1.3}, scraping, surface);
        std::array<double, 2> expected;
        reference.read(reinterpret_cast<char *>(expected.data()), sizeof(expected));
        Check(bool(reference), "Read SDT controller reference");
        Near(rolling_value, expected[0], 1e-14, 1e-13, "SDT rolling equivalence");
        Near(scraping_value, expected[1], 1e-14, 1e-13, "SDT scraping equivalence");
    }
    Check(reference.peek() == std::char_traits<char>::eof(), "SDT reference has exact expected length");
    std::cout << "SDT upstream 4096-frame impact/friction/rolling/scraping max contact error: " << max_error << '\n';
}

void Physics() {
    Near(ImpactForce(ImpactParameters{}, -0.1, 1.0), 0, 0, 0, "No separated impact");
    Check(ImpactForce(ImpactParameters{}, 0.1, -100.0) < 0, "Published impact law outside nonadhesive velocity domain");
    Near(UnilateralImpactForce(ImpactParameters{}, 0.1, -100.0), 0, 0, 0, "Explicit unilateral extension");
    FrictionParameters parameters{.Noisiness = 0};
    FrictionState state;
    Near(FrictionForce(parameters, state, 0.0, 0.0, 1.0 / 48000), 0, 0, 0, "Rest equilibrium");
    for (unsigned frame = 0; frame < 48000; ++frame) FrictionForce(parameters, state, 0.1, 0.0, 1.0 / 48000);
    const double equilibrium = parameters.NormalForce * (parameters.DynamicCoefficient + (parameters.StaticCoefficient - parameters.DynamicCoefficient) * std::exp(-1.0));
    Near(parameters.Stiffness * state.Bristle, equilibrium, 1e-8, 1e-8, "Stribeck sliding equilibrium");
    const double force = FrictionForce(parameters, state, 0.1, 0.0, 1.0 / 48000);
    Check(force * 0.1 > 0, "Steady friction opposes relative motion");
    parameters.NormalForce = 0;
    Near(FrictionForce(parameters, state, 0.1, 0.0, 1.0 / 48000), 0, 0, 0, "Unloaded friction");
    Near(state.Bristle, 0, 0, 0, "Unloading resets bristles");

    auto body0 = Inertial(), body1 = Inertial();
    SetVelocity(body0, -0.5);
    ContactState contact;
    double initial_energy = 0.5 * 0.03 * 0.5 * 0.5, peak_energy{};
    bool hit = false, separated = false;
    for (unsigned frame = 0; frame < 4096; ++frame) {
        const auto sample = StepImpact(body0, body1, contact, {});
        hit |= sample.Force > 0;
        separated |= hit && sample.Position1 < sample.Position0;
        Check(sample.Force >= 0, "Unilateral coupled force");
        const double energy = 0.5 * 0.03 * (sample.Velocity0 * sample.Velocity0 + sample.Velocity1 * sample.Velocity1);
        peak_energy = std::max(peak_energy, energy);
        Near(0.03 * (sample.Velocity0 + sample.Velocity1), -0.015, 1e-10, 1e-9, "Two-body momentum conservation");
    }
    Check(hit && separated, "Impact collides and separates");
    Check(peak_energy <= initial_energy * (1 + 1e-8), "Impact energy bound");

    RollingState rolling;
    ScrapingState scraping;
    Near(StepRolling(RollingParameters{.Velocity = 0}, rolling, 1.0), -0.0981, 1e-14, 0, "Stationary rolling gravity");
    Near(StepScraping(ScrapingParameters{.Velocity = 0}, scraping, 1.0), 0, 0, 0, "Stationary scraping");
}

void Streaming() {
    auto whole0 = Inertial(), whole1 = Modal();
    SetVelocity(whole0, -0.5);
    auto split0 = whole0, split1 = whole1;
    ContactState whole, split;
    std::array<ContactSample, 1024> expected;
    for (auto &sample : expected) sample = StepImpact(whole0, whole1, whole, {});
    unsigned index{};
    for (unsigned block : {1u, 31u, 127u, 2u, 511u, 352u}) {
        for (unsigned frame = 0; frame < block; ++frame, ++index) {
            const auto sample = StepImpact(split0, split1, split, {});
            Near(sample.Force, expected[index].Force, 0, 0, "Streaming contact force");
            Near(sample.Output1, expected[index].Output1, 0, 0, "Streaming modal output");
        }
    }
    Check(index == expected.size(), "Streaming covers all frames");
    Check(whole0.Position == split0.Position && whole1.Velocity == split1.Velocity && whole.Energy == split.Energy, "Streaming terminal state");
}

void LimiterConvergence() {
    auto first = Inertial(), second = Modal();
    SetVelocity(first, -0.05);
    ContactState state;
    const FrictionParameters parameters{.NormalForce = 0.7, .Dissipation = 0.015, .Viscosity = 0.01, .Noisiness = 0};
    bool checked = false;
    for (unsigned frame = 0; frame < 48000 && !checked; ++frame) {
        auto forced = first;
        ApplyForce(forced, -0.03);
        auto bristle = state.Friction;
        const double candidate = FrictionForce(parameters, bristle, Velocity(second) - Velocity(first), 0.0, 1.0 / 48000);
        const auto e0 = ContactEnergy(forced), e1 = ContactEnergy(second);
        const ContactEnergyT<double> polynomial{e0.Free + e1.Free, e0.Linear - e1.Linear, e0.Quadratic + e1.Quadratic};
        double energy = 0;
        const double exact = LimitContactForce(polynomial, energy, candidate);
        double old_energy = 0;
        const double old = reference::LimitForce(forced, second, old_energy, candidate);
        if (exact > 0.001 && std::abs(old - exact) > 0.001) {
            std::cout << "SDT fixed-state limiter frame=" << frame << " candidate=" << candidate << " exact=" << exact << " free_energy=" << polynomial.Free << '\n';
            double previous_error = std::numeric_limits<double>::infinity();
            for (double tolerance : {0.001, 1e-6, 1e-9}) {
                double allowance = 0;
                const double force = reference::LimitForce(forced, second, allowance, candidate, tolerance);
                const double error = std::abs(force - exact);
                const double residual = (polynomial.Quadratic * force + polynomial.Linear) * force;
                std::cout << "  tolerance=" << tolerance << " force=" << force << " residual_energy=" << residual << '\n';
                Check(error < previous_error, "Tightened upstream bisection converges to analytic root");
                previous_error = error;
            }
            Check(previous_error < 1e-7, "Tight upstream approximation matches exact force bound");
            Check(std::abs((polynomial.Quadratic * exact + polynomial.Linear) * exact) < 1e-18, "Analytic limiter energy residual");
            checked = true;
        }
        StepFriction(first, second, state, parameters, 0, -0.03);
    }
    Check(checked, "Observed nontrivial source force-limiter truncation");

    for (double candidate : {-1e6, -1.0, -1e-9, 0.0, 1e-9, 1.0, 1e6}) {
        for (double linear : {-0.1, 0.0, 0.1}) {
            const ContactEnergyT<double> polynomial{1, linear, 0.01};
            double allowance = 0.001;
            const double force = LimitContactForce(polynomial, allowance, candidate);
            Check(std::isfinite(force) && std::abs(force) <= std::abs(candidate), "Energy limiter bounds force magnitude");
            Check((0.01 * force + linear) * force <= 0.001 + 1e-14, "Energy limiter satisfies source inequality");
        }
    }
}

void StationaryEquilibrium() {
    auto first = Inertial(), second = Inertial();
    ContactState state;
    const FrictionParameters parameters{.NormalForce = 0.7, .Dissipation = 0.015, .Viscosity = 0.01, .Noisiness = 0};
    state.Friction.Bristle = 0.03 / parameters.Stiffness;
    for (unsigned frame = 0; frame < 192000; ++frame) {
        const auto sample = StepFriction(first, second, state, parameters, 0, -0.03, 0.03);
        Near(sample.Force, 0.03, 1e-12, 0, "Balanced stationary friction force");
        Near(sample.Velocity0, 0, 1e-12, 0, "Balanced stationary first body");
        Near(sample.Velocity1, 0, 1e-12, 0, "Balanced stationary second body");
    }
}

void TimeStepConvergence() {
    double previous_velocity = 1, previous_audio = 1;
    for (double sample_rate : {24000., 48000., 96000., 192000.}) {
        auto first = MakeBody(std::array{ModeParameters{.Mass = 0.03}}, sample_rate);
        auto second = MakeBody(std::array{ModeParameters{230, 0.3, 0.4, 0.9, 0.3}, ModeParameters{690, 0.15, 0.25, 0.4, 0.8}, ModeParameters{1430, 0.08, 0.2, 0.25, 0.2}}, sample_rate);
        SetVelocity(first, -0.05);
        ContactState state;
        const FrictionParameters parameters{.NormalForce = 0.7, .Dissipation = 0.015, .Viscosity = 0.01, .Noisiness = 0};
        double force_sum{}, force_square{}, nyquist{}, audio_sum{}, audio_square{};
        for (unsigned frame = 0; frame < unsigned(2 * sample_rate); ++frame) {
            const auto sample = StepFriction(first, second, state, parameters, 0, -0.03);
            if (frame >= unsigned(1.5 * sample_rate)) {
                force_sum += sample.Force;
                force_square += sample.Force * sample.Force;
                nyquist += frame % 2 ? -sample.Force : sample.Force;
                audio_sum += sample.Output1;
                audio_square += sample.Output1 * sample.Output1;
            }
        }
        const double count = 0.5 * sample_rate;
        const double force_mean = force_sum / count, force_variance = force_square / count - force_mean * force_mean;
        const double audio_rms = std::sqrt(std::max(0.0, audio_square / count - std::pow(audio_sum / count, 2)));
        Near(force_mean, 0.03, 2e-7, 0, "Constant-drive friction mean equilibrium");
        Check(std::abs(Velocity(first)) < previous_velocity * 0.51, "Friction drift decreases linearly with timestep");
        Check(audio_rms < previous_audio, "Steady modal output decreases with timestep");
        Check(std::pow(nyquist / count, 2) / force_variance > 0.99, "Steady force chatter concentrates at the sample Nyquist frequency");
        previous_velocity = std::abs(Velocity(first));
        previous_audio = audio_rms;
    }
}

void SustainedRubbing() {
    auto first = Inertial(), second = Modal();
    SetVelocity(first, -0.3);
    auto original0 = first, original1 = second;
    ContactState state, original;
    const FrictionParameters parameters{.NormalForce = 0.7, .Dissipation = 0.015, .Viscosity = 0.01, .Noisiness = 0.01};
    auto random = surface_audio::MakeRandom(42);
    double output_sum{}, output_square{}, force_error{}, force_norm{};
    for (unsigned frame = 0; frame < 96000; ++frame) {
        const double velocity = Velocity(second) - Velocity(first);
        const double noise = 2 * surface_audio::Uniform(random) - 1;
        const auto sample = StepFriction(first, second, state, parameters, noise, -0.15);
        const auto source = reference::StepFriction(original0, original1, original, parameters, noise, -0.15);
        Check(sample.Force * velocity > 0, "Sustained rubbing opposes slip");
        Check(sample.Velocity0 < -0.29 && sample.Velocity0 > -0.65, "Sustained rubbing bounded slip");
        force_error += std::pow(sample.Force - source.Force, 2);
        force_norm += source.Force * source.Force;
        if (frame >= 72000) {
            output_sum += sample.Output1;
            output_square += sample.Output1 * sample.Output1;
        }
    }
    const double output_rms = std::sqrt(output_square / 24000 - std::pow(output_sum / 24000, 2));
    Check(output_rms > 1e-10 && output_rms < 1e-8, "Sustained rubbing retains textured modal output");
    Check(std::sqrt(force_error / force_norm) < 1e-7, "Sliding source-reference force agreement");
    std::cout << "SDT sustained rubbing upstream force L2=" << std::sqrt(force_error / force_norm) << " late output AC RMS=" << output_rms << '\n';
}

void Gpu(uint32_t contacts, uint32_t frames, bool sustained = false) {
    using namespace surface_audio;
    std::vector<GpuContact> configurations;
    std::vector<GpuMode> modes;
    std::vector<GpuModeState> initial_modes;
    std::vector<GpuModeState> expected_modes;
    std::vector<GpuOutput> expected;
    std::vector<GpuInput> full_inputs;
    for (uint32_t index = 0; index < contacts; ++index) {
        const bool friction = sustained || index % 2;
        auto first = sustained ? MakeBody(std::array{ModeParameters{.Mass = 0.3}}, 48000) : Inertial();
        auto second = Modal();
        SetVelocity(first, friction ? -0.05 : -0.5);
        GpuContact configuration;
        configuration.Kind = friction ? GpuContactKind::Friction : GpuContactKind::Impact;
        configuration.Friction.NormalForce = 0.7f;
        configuration.Friction.Dissipation = 0.015f;
        configuration.Friction.Viscosity = 0.01f;
        configuration.Friction.Noisiness = 0;
        if (sustained) {
            configuration.Friction.NormalForce = std::array{0.2f, 0.7f, 1.2f}[index % 3];
            configuration.Friction.Stiffness = std::array{100.f, 500.f, 1000.f}[index % 3];
            configuration.Friction.DynamicCoefficient = index < 3 ? 0.2f : 0.4f;
        }
        const auto range0 = AppendBody(first, modes, initial_modes), range1 = AppendBody(second, modes, initial_modes);
        configuration.Offset0 = range0.Offset;
        configuration.Count0 = range0.Count;
        configuration.Offset1 = range1.Offset;
        configuration.Count1 = range1.Count;
        configurations.push_back(configuration);
        ContactState state;
        const auto fp = configuration.Friction;
        const FrictionParameters parameters{fp.NormalForce, fp.StribeckVelocity, fp.StaticCoefficient, fp.DynamicCoefficient, fp.BreakAway, fp.Stiffness, fp.Dissipation, fp.Viscosity, fp.Noisiness};
        for (uint32_t frame = 0; frame < frames; ++frame) {
            const float external = sustained ? float(-1.1 * parameters.NormalForce * std::sin(2 * 3.14159265358979323846 * 0.5 * frame / 48000)) : friction ? -0.03f :
                                                                                                                                                              0;
            full_inputs.push_back({external, 0, 0});
            const auto sample = friction ? StepFriction(first, second, state, parameters, 0, external) : StepImpact(first, second, state, {});
            expected.push_back({float(sample.Force), float(sample.Position0), float(sample.Position1), float(sample.Velocity0), float(sample.Velocity1), float(sample.Output0), float(sample.Output1)});
        }
        for (const auto *body : {&first, &second}) {
            for (size_t mode = 0; mode < body->Position.size(); ++mode) expected_modes.push_back({float(body->Position[mode]), float(body->PreviousPosition[mode]), float(body->Velocity[mode]), 0});
        }
    }
    auto gpu = CreateGpu();
    const auto kernel = CreateKernel(gpu, "SdtContacts");
    const auto configuration_buffer = Upload<GpuContact>(gpu, configurations);
    const auto modes_buffer = Upload<GpuMode>(gpu, modes);
    std::vector<GpuModeState> previous_modes;
    std::vector<GpuContactState> previous_contacts;
    const auto render = [&](std::span<const uint32_t> blocks) {
        const auto dispatch_buffer = Upload(gpu, GpuDispatch{contacts, frames, 1.f / 48000});
        const auto mode_states_buffer = Upload<GpuModeState>(gpu, initial_modes);
        const std::vector<GpuContactState> initial_contacts(contacts);
        const auto contact_states_buffer = Upload<GpuContactState>(gpu, initial_contacts);
        const auto inputs_buffer = CreateBuffer(gpu, sizeof(GpuInput) * contacts * frames);
        const auto outputs_buffer = CreateBuffer(gpu, sizeof(GpuOutput) * contacts * frames);
        std::vector<GpuOutput> result(contacts * frames);
        uint32_t completed{};
        for (uint32_t block : blocks) {
            BufferSpan<GpuDispatch>(dispatch_buffer)[0].Frames = block;
            auto inputs = BufferSpan<GpuInput>(inputs_buffer);
            for (uint32_t index = 0; index < contacts; ++index) {
                for (uint32_t frame = 0; frame < block; ++frame) inputs[index * block + frame] = full_inputs[index * frames + completed + frame];
            }
            const std::array bindings{GpuBinding{dispatch_buffer, 0}, GpuBinding{configuration_buffer, 1}, GpuBinding{contact_states_buffer, 2}, GpuBinding{modes_buffer, 3}, GpuBinding{mode_states_buffer, 4}, GpuBinding{inputs_buffer, 5}, GpuBinding{outputs_buffer, 6}};
            BeginGpu(gpu);
            DispatchGpu(gpu, kernel, bindings, {contacts, 1, 1});
            SubmitGpu(gpu);
            WaitGpu(gpu);
            const auto outputs = BufferSpan<GpuOutput>(outputs_buffer);
            for (uint32_t index = 0; index < contacts; ++index) std::copy_n(outputs.begin() + index * block, block, result.begin() + index * frames + completed);
            completed += block;
        }
        Check(completed == frames, "GPU streaming covers all samples");
        const auto final_modes = BufferSpan<GpuModeState>(mode_states_buffer);
        const auto final_contacts = BufferSpan<GpuContactState>(contact_states_buffer);
        ValidateGpuContactStates(final_contacts);
        for (size_t mode = 0; mode < final_modes.size(); ++mode) {
            Near(final_modes[mode].Position, expected_modes[mode].Position, 1e-5, 1e-3, "GPU terminal mode position");
            Near(final_modes[mode].Velocity, expected_modes[mode].Velocity, 5e-5, 1e-3, "GPU terminal mode velocity");
        }
        if (!previous_modes.empty()) {
            Check(std::memcmp(final_modes.data(), previous_modes.data(), final_modes.size_bytes()) == 0, "GPU streaming terminal modes");
            Check(std::memcmp(final_contacts.data(), previous_contacts.data(), final_contacts.size_bytes()) == 0, "GPU streaming terminal contacts");
        }
        previous_modes.assign(final_modes.begin(), final_modes.end());
        previous_contacts.assign(final_contacts.begin(), final_contacts.end());
        return result;
    };
    const std::array whole_blocks{frames};
    const std::array split_blocks{frames / 2 - 1, frames / 2 + 1};
    const auto whole = render(whole_blocks), split = render(split_blocks);
    std::array<double, 2> error{}, norm{};
    for (size_t sample = 0; sample < whole.size(); ++sample) {
        Check(std::memcmp(&whole[sample], &split[sample], sizeof(GpuOutput)) == 0, "GPU streaming equality");
        Check(std::isfinite(whole[sample].Output1) && std::isfinite(whole[sample].Force), "GPU finite contact output");
        if (!sustained && (sample / frames) % 2 == 0) Check(whole[sample].Force >= 0, "GPU nonadhesive-domain impact");
        const std::array actual{whole[sample].Force, whole[sample].Output1};
        const std::array reference{expected[sample].Force, expected[sample].Output1};
        for (size_t field = 0; field < error.size(); ++field) {
            error[field] += std::pow(double(actual[field]) - reference[field], 2);
            norm[field] += double(reference[field]) * reference[field];
        }
    }
    std::cout << "SDT Metal4 " << DeviceName(gpu) << ' ' << contacts << " contacts x " << frames << " frames relative L2 force/output: " << std::sqrt(error[0] / norm[0]) << '/' << std::sqrt(error[1] / norm[1]) << '\n';
    Check(std::sqrt(error[0] / norm[0]) < 0.003 && std::sqrt(error[1] / norm[1]) < 0.001, "GPU float versus CPU double full-trace agreement");
    for (uint32_t contact = 0; contact < contacts; ++contact) {
        double force_error{}, force_norm{}, output_error{}, output_norm{};
        for (uint32_t frame = 0; frame < frames; ++frame) {
            const uint32_t sample = contact * frames + frame;
            force_error += std::pow(double(whole[sample].Force) - expected[sample].Force, 2);
            force_norm += std::pow(double(expected[sample].Force), 2);
            output_error += std::pow(double(whole[sample].Output1) - expected[sample].Output1, 2);
            output_norm += std::pow(double(expected[sample].Output1), 2);
        }
        Check(std::sqrt(force_error / force_norm) < 0.003, "GPU per-contact force agreement");
        Check(std::sqrt(output_error / output_norm) < 0.001, "GPU per-contact audio agreement");
        if (frames > 256) std::cout << "  contact=" << contact << " force/output L2=" << std::sqrt(force_error / force_norm) << '/' << std::sqrt(output_error / output_norm) << '\n';
    }
}

void GpuControllers() {
    using namespace surface_audio;
    constexpr uint32_t voices = 64, frames = 4096;
    auto gpu = CreateGpu();
    const auto kernel = CreateKernel(gpu, "SdtSurfaceForces");
    std::vector<GpuSurfaceParameters> parameters(voices);
    std::vector<GpuSurfaceState> initial_states(voices), expected_states(voices);
    std::vector<float> heights(voices * frames), expected(voices * frames);
    for (uint32_t voice = 0; voice < voices; ++voice) {
        parameters[voice].Kind = voice % 2 ? GpuSurfaceKind::Scraping : GpuSurfaceKind::Rolling;
        parameters[voice].Rolling = {0.07f, 2, 0.04f, 1.3f};
        parameters[voice].Scraping = {0.07f, 0.8f, 1.3f};
        for (uint32_t frame = 0; frame < frames; ++frame) {
            const uint32_t sample = voice * frames + frame;
            heights[sample] = float(std::sin(frame * 0.13 + voice) + 0.3 * std::cos(frame * 0.71));
            expected[sample] = voice % 2 ? StepScraping(parameters[voice].Scraping, expected_states[voice].Scraping, heights[sample]) : StepRolling(parameters[voice].Rolling, expected_states[voice].Rolling, heights[sample]);
        }
    }
    const auto parameter_buffer = Upload<GpuSurfaceParameters>(gpu, parameters);
    const auto run = [&](std::span<const uint32_t> blocks) {
        const auto dispatch_buffer = Upload(gpu, GpuDispatch{voices, frames, 1.f / 48000});
        const auto state_buffer = Upload<GpuSurfaceState>(gpu, initial_states);
        const auto height_buffer = CreateBuffer(gpu, sizeof(float) * heights.size());
        const std::vector<GpuInput> initial_inputs(heights.size(), GpuInput{0, 0.125f, -0.25f});
        const auto output_buffer = Upload<GpuInput>(gpu, initial_inputs);
        std::vector<float> result(heights.size());
        uint32_t completed{};
        for (uint32_t block : blocks) {
            BufferSpan<GpuDispatch>(dispatch_buffer)[0].Frames = block;
            for (uint32_t voice = 0; voice < voices; ++voice) std::copy_n(heights.begin() + voice * frames + completed, block, BufferSpan<float>(height_buffer).begin() + voice * block);
            const std::array bindings{GpuBinding{dispatch_buffer, 0}, GpuBinding{parameter_buffer, 1}, GpuBinding{state_buffer, 2}, GpuBinding{height_buffer, 3}, GpuBinding{output_buffer, 4}};
            BeginGpu(gpu);
            DispatchGpu(gpu, kernel, bindings, {voices, 1, 1});
            SubmitGpu(gpu);
            WaitGpu(gpu);
            const auto outputs = BufferSpan<GpuInput>(output_buffer);
            for (uint32_t voice = 0; voice < voices; ++voice) {
                for (uint32_t frame = 0; frame < block; ++frame) {
                    const auto output = outputs[voice * block + frame];
                    result[voice * frames + completed + frame] = output.External0;
                    Check(output.External1 == 0.125f && output.Noise == -0.25f, "Controller preserves other input fields");
                }
            }
            completed += block;
        }
        const auto states = BufferSpan<GpuSurfaceState>(state_buffer);
        for (uint32_t voice = 0; voice < voices; ++voice) {
            Near(states[voice].Rolling.GroundTrace, expected_states[voice].Rolling.GroundTrace, 2e-6, 2e-6, "GPU rolling terminal ground");
            Near(states[voice].Rolling.BallFlight, expected_states[voice].Rolling.BallFlight, 2e-6, 2e-6, "GPU rolling terminal flight");
            Near(states[voice].Scraping.GroundTrace, expected_states[voice].Scraping.GroundTrace, 2e-6, 2e-6, "GPU scraping terminal ground");
        }
        return result;
    };
    const std::array whole_blocks{frames};
    const std::array split_blocks{2047u, 2049u};
    const auto whole = run(whole_blocks), split = run(split_blocks);
    Check(whole == split, "GPU controllers streaming equality");
    for (size_t sample = 0; sample < whole.size(); ++sample) Near(whole[sample], expected[sample], 2e-6, 2e-6, "GPU rolling/scraping full-trace equivalence");
    std::cout << "SDT Metal4 rolling/scraping 64 voices x 4096 frames passed\n";
}

void EndToEndControllers() {
    using namespace surface_audio;
    constexpr uint32_t contacts = 2, frames = 96000;
    std::vector<GpuMode> modes;
    std::vector<GpuModeState> initial_modes;
    std::array<GpuContact, contacts> configurations;
    std::array<GpuSurfaceParameters, contacts> surface_parameters;
    std::vector<float> heights(contacts * frames), cpu_forces(contacts * frames);
    std::vector<ContactSample> expected(contacts * frames);
    std::array<Body, contacts> expected0, expected1;
    std::array<ContactState, contacts> expected_contacts;
    for (uint32_t contact = 0; contact < contacts; ++contact) {
        auto first = Inertial(), second = Modal();
        SetVelocity(first, -0.5);
        const auto first_range = AppendBody(first, modes, initial_modes), second_range = AppendBody(second, modes, initial_modes);
        configurations[contact].Offset0 = first_range.Offset;
        configurations[contact].Count0 = first_range.Count;
        configurations[contact].Offset1 = second_range.Offset;
        configurations[contact].Count1 = second_range.Count;
        auto &parameters = surface_parameters[contact];
        parameters.Kind = contact == 0 ? GpuSurfaceKind::Rolling : GpuSurfaceKind::Scraping;
        parameters.Rolling = {0.07f, 2, 0.04f, 1.3f};
        parameters.Scraping = {0.07f, 0.8f, 1.3f};
        GpuSurfaceState surface;
        for (uint32_t frame = 0; frame < frames; ++frame) {
            const size_t sample = size_t(contact) * frames + frame;
            heights[sample] = float(std::sin(float(frame) * .13f) + .3f * std::cos(float(frame) * .71f));
            cpu_forces[sample] = contact == 0 ? StepRolling(parameters.Rolling, surface.Rolling, heights[sample]) : StepScraping(parameters.Scraping, surface.Scraping, heights[sample]);
            expected[sample] = StepImpact(first, second, expected_contacts[contact], {}, cpu_forces[sample]);
        }
        expected0[contact] = std::move(first);
        expected1[contact] = std::move(second);
    }
    auto gpu = CreateGpu();
    const auto controller_kernel = CreateKernel(gpu, "SdtSurfaceForces"), contact_kernel = CreateKernel(gpu, "SdtContacts");
    const auto configuration_buffer = Upload<GpuContact>(gpu, configurations);
    const auto coefficient_buffer = Upload<GpuMode>(gpu, modes);
    const auto surface_parameter_buffer = Upload<GpuSurfaceParameters>(gpu, surface_parameters);
    const auto height_buffer = Upload<float>(gpu, heights);
    const auto dispatch = Upload(gpu, GpuDispatch{contacts, frames, 1.f / 48000});
    struct Result {
        std::vector<GpuOutput> Samples;
        std::vector<GpuModeState> Modes;
        std::vector<GpuContactState> Contacts;
        std::vector<GpuInput> Inputs;
    };
    const auto render = [&](bool separate_submissions) {
        const std::array<GpuSurfaceState, contacts> initial_surfaces{};
        const auto surface_states = Upload<GpuSurfaceState>(gpu, initial_surfaces);
        const std::array<GpuContactState, contacts> initial_contacts{};
        const auto contact_states = Upload<GpuContactState>(gpu, initial_contacts);
        const auto mode_states = Upload<GpuModeState>(gpu, initial_modes);
        const std::vector<GpuInput> initial_inputs(contacts * frames);
        const auto inputs = Upload<GpuInput>(gpu, initial_inputs);
        const auto outputs = CreateBuffer(gpu, sizeof(GpuOutput) * contacts * frames);
        const std::array controller_bindings{GpuBinding{dispatch, 0}, GpuBinding{surface_parameter_buffer, 1}, GpuBinding{surface_states, 2}, GpuBinding{height_buffer, 3}, GpuBinding{inputs, 4}};
        const std::array contact_bindings{GpuBinding{dispatch, 0}, GpuBinding{configuration_buffer, 1}, GpuBinding{contact_states, 2}, GpuBinding{coefficient_buffer, 3}, GpuBinding{mode_states, 4}, GpuBinding{inputs, 5}, GpuBinding{outputs, 6}};
        BeginGpu(gpu);
        DispatchGpu(gpu, controller_kernel, controller_bindings, {contacts, 1, 1});
        if (separate_submissions) {
            SubmitGpu(gpu);
            WaitGpu(gpu);
            BeginGpu(gpu);
        }
        DispatchGpu(gpu, contact_kernel, contact_bindings, {contacts, 1, 1});
        SubmitGpu(gpu);
        WaitGpu(gpu);
        ValidateGpuContactStates(BufferSpan<GpuContactState>(contact_states));
        const auto copy = []<typename T>(GpuBuffer buffer) { const auto values = BufferSpan<T>(buffer); return std::vector<T>(values.begin(), values.end()); };
        return Result{copy.template operator()<GpuOutput>(outputs), copy.template operator()<GpuModeState>(mode_states), copy.template operator()<GpuContactState>(contact_states), copy.template operator()<GpuInput>(inputs)};
    };
    const auto pipeline = render(false), staged = render(true);
    Check(std::memcmp(pipeline.Samples.data(), staged.Samples.data(), pipeline.Samples.size() * sizeof(GpuOutput)) == 0, "SDT controller/contact dependency equals separate submissions");
    Check(std::memcmp(pipeline.Modes.data(), staged.Modes.data(), pipeline.Modes.size() * sizeof(GpuModeState)) == 0, "SDT pipeline terminal modes equal separate submissions");
    Check(std::memcmp(pipeline.Contacts.data(), staged.Contacts.data(), pipeline.Contacts.size() * sizeof(GpuContactState)) == 0, "SDT pipeline terminal contacts equal separate submissions");
    for (uint32_t contact = 0; contact < contacts; ++contact) {
        double force_error{}, force_norm{}, output_error{}, output_norm{}, gpu_force_norm{}, gpu_output_norm{}, impulse{}, drive_impulse{};
        double gpu_energy{}, cpu_energy{};
        const auto &configuration = configurations[contact];
        for (const auto [offset, count, body] : {std::tuple{configuration.Offset0, configuration.Count0, &expected0[contact]}, std::tuple{configuration.Offset1, configuration.Count1, &expected1[contact]}}) {
            for (uint32_t mode = 0; mode < count; ++mode) {
                const auto &actual = pipeline.Modes[offset + mode];
                const auto &parameters = modes[offset + mode];
                Check(std::isfinite(actual.Position) && std::isfinite(actual.Velocity), "SDT end-to-end finite terminal motion");
                gpu_energy += .5 * (parameters.Mass * actual.Velocity * actual.Velocity + parameters.Stiffness * actual.Position * actual.Position) * parameters.ContactGain;
                cpu_energy += .5 * (body->Mass[mode] * body->Velocity[mode] * body->Velocity[mode] + body->Stiffness[mode] * body->Position[mode] * body->Position[mode]) * body->ContactGain[mode];
            }
        }
        for (uint32_t frame = 0; frame < frames; ++frame) {
            const size_t sample = size_t(contact) * frames + frame;
            const auto &actual = pipeline.Samples[sample];
            const auto &reference = expected[sample];
            Near(pipeline.Inputs[sample].External0, cpu_forces[sample], 2e-6, 2e-6, "SDT end-to-end controller force");
            Check(std::isfinite(actual.Force) && std::isfinite(actual.Output1) && actual.Force >= 0, "SDT end-to-end finite compressive contact");
            force_error += std::pow(double(actual.Force) - reference.Force, 2);
            force_norm += reference.Force * reference.Force;
            output_error += std::pow(double(actual.Output1) - reference.Output1, 2);
            output_norm += reference.Output1 * reference.Output1;
            gpu_force_norm += double(actual.Force) * actual.Force;
            gpu_output_norm += double(actual.Output1) * actual.Output1;
            impulse += actual.Force / 48000.;
            drive_impulse += pipeline.Inputs[sample].External0 / 48000.;
        }
        const auto &terminal = pipeline.Samples[(contact + 1) * frames - 1];
        Near(.03 * (terminal.Velocity0 + .5), impulse + drive_impulse, 2e-7, 1e-5, "SDT end-to-end inertial momentum balance");
        Check(gpu_energy >= 0 && std::isfinite(gpu_energy) && pipeline.Contacts[contact].Energy >= 0, "SDT end-to-end finite nonnegative terminal energy");
        std::cout << "SDT end-to-end " << (contact == 0 ? "rolling" : "scraping") << " 2s force/audio L2=" << std::sqrt(force_error / force_norm) << '/' << std::sqrt(output_error / output_norm) << " RMS ratios=" << std::sqrt(gpu_force_norm / force_norm) << '/' << std::sqrt(gpu_output_norm / output_norm) << " terminal energy GPU/CPU=" << gpu_energy << '/' << cpu_energy << " terminal velocity GPU/CPU=" << terminal.Velocity0 << '/' << Velocity(expected0[contact]) << '\n';
        Check(std::abs(std::sqrt(gpu_force_norm / force_norm) - 1) < .02 && std::abs(std::sqrt(gpu_output_norm / output_norm) - 1) < .02, "SDT end-to-end force/audio RMS agreement");
        std::vector<std::vector<std::array<double, 2>>> first_impacts;
        for (unsigned substeps : {1u, 2u, 4u, 8u, 16u, 32u}) {
            auto first = MakeBody(std::array{ModeParameters{.Mass = .03}}, 48000. * substeps);
            auto second = MakeBody(std::array{ModeParameters{230, .3, .4, .9, .3}, ModeParameters{690, .15, .25, .4, .8}, ModeParameters{1430, .08, .2, .25, .2}}, 48000. * substeps);
            SetVelocity(first, -.5);
            ContactState state;
            std::vector<std::array<double, 2>> samples(512);
            for (unsigned frame = 0; frame < samples.size(); ++frame) {
                for (unsigned step = 0; step < substeps; ++step) {
                    const auto sample = StepImpact(first, second, state, {}, cpu_forces[size_t(contact) * frames + frame]);
                    samples[frame][0] += sample.Force / substeps;
                    samples[frame][1] += sample.Output1 / substeps;
                }
            }
            first_impacts.push_back(std::move(samples));
        }
        std::array<double, 2> previous_error{1e9, 1e9};
        for (size_t level = 0; level + 1 < first_impacts.size(); ++level) {
            std::array<double, 2> error{}, norm{};
            for (size_t frame = 0; frame < first_impacts[level].size(); ++frame) {
                for (size_t field = 0; field < 2; ++field) {
                    error[field] += std::pow(first_impacts[level][frame][field] - first_impacts.back()[frame][field], 2);
                    norm[field] += std::pow(first_impacts.back()[frame][field], 2);
                }
            }
            std::cout << "  first-impact convergence substeps=" << (1u << level) << " force/audio L2=" << std::sqrt(error[0] / norm[0]) << '/' << std::sqrt(error[1] / norm[1]) << '\n';
            for (size_t field = 0; field < 2; ++field) {
                Check(error[field] < previous_error[field], "SDT isolated first-impact timestep convergence");
                previous_error[field] = error[field];
            }
        }
        for (unsigned substeps : {1u, 2u, 4u, 8u}) {
            const double sample_rate = 48000. * substeps;
            auto first = MakeBody(std::array{ModeParameters{.Mass = .03}}, sample_rate);
            auto second = MakeBody(std::array{ModeParameters{230, .3, .4, .9, .3}, ModeParameters{690, .15, .25, .4, .8}, ModeParameters{1430, .08, .2, .25, .2}}, sample_rate);
            SetVelocity(first, -.5);
            ContactState state;
            double refined_force_norm{}, refined_output_norm{}, refined_force_error{}, refined_output_error{};
            for (uint32_t frame = 0; frame < frames; ++frame) {
                const size_t index = size_t(contact) * frames + frame;
                const double drive = substeps == 1 ? std::nextafter(cpu_forces[index], std::numeric_limits<float>::infinity()) : cpu_forces[index];
                for (unsigned step = 0; step < substeps; ++step) {
                    const auto sample = StepImpact(first, second, state, {}, drive);
                    refined_force_norm += sample.Force * sample.Force / substeps;
                    refined_output_norm += sample.Output1 * sample.Output1 / substeps;
                    if (substeps == 1) {
                        refined_force_error += std::pow(sample.Force - expected[index].Force, 2);
                        refined_output_error += std::pow(sample.Output1 - expected[index].Output1, 2);
                    }
                }
            }
            std::cout << "  " << (substeps == 1 ? "one-ULP drive perturbation" : "fixed-forcing body refinement") << " substeps=" << substeps << " force/audio RMS ratios=" << std::sqrt(refined_force_norm / force_norm) << '/' << std::sqrt(refined_output_norm / output_norm);
            if (substeps == 1) std::cout << " force/audio L2=" << std::sqrt(refined_force_error / force_norm) << '/' << std::sqrt(refined_output_error / output_norm);
            std::cout << '\n';
        }
    }
}

void ExternalRouting() {
    using namespace surface_audio;
    constexpr uint32_t contacts = 4, frames = 4096;
    const std::array<ModeParameters, 3> parameters{{{500, .03, 1, 100, 100}, {1300, .02, 1, 100, 100}, {1700, .01, 1, 100, 100}}};
    std::vector<GpuMode> modes;
    std::vector<GpuModeState> initial_modes;
    std::array<GpuContact, contacts> configurations{};
    std::vector<GpuInput> input(contacts * frames);
    std::vector<double> expected(input.size());
    for (uint32_t index = 0; index < contacts; ++index) {
        auto body = MakeBody(parameters, 44100);
        const auto range = AppendBody(body, modes, initial_modes);
        configurations[index].Offset0 = range.Offset;
        configurations[index].Count0 = range.Count;
        configurations[index].Kind = GpuContactKind::External;
        for (uint32_t frame = 0; frame < frames; ++frame) {
            const size_t sample = index * frames + frame;
            input[sample].External0 = static_cast<float>(std::sin(frame * .071 + index) + .2 * std::cos(frame * .193));
            reference::ApplyForce(body, input[sample].External0);
            reference::StepBody(body);
            expected[sample] = Output(body);
        }
    }
    auto gpu = CreateGpu();
    const auto kernel = CreateKernel(gpu, "SdtContacts");
    const auto dispatch = Upload(gpu, GpuDispatch{contacts, frames, 1.f / 44100});
    const auto configuration = Upload<GpuContact>(gpu, configurations);
    const std::array<GpuContactState, contacts> initial_contacts{};
    const auto states = Upload<GpuContactState>(gpu, initial_contacts);
    const auto coefficients = Upload<GpuMode>(gpu, modes);
    const auto mode_states = Upload<GpuModeState>(gpu, initial_modes);
    const auto inputs = Upload<GpuInput>(gpu, input);
    const auto outputs = CreateBuffer(gpu, sizeof(GpuOutput) * input.size());
    const std::array bindings{GpuBinding{dispatch, 0}, GpuBinding{configuration, 1}, GpuBinding{states, 2}, GpuBinding{coefficients, 3}, GpuBinding{mode_states, 4}, GpuBinding{inputs, 5}, GpuBinding{outputs, 6}};
    BeginGpu(gpu);
    DispatchGpu(gpu, kernel, bindings, {contacts, 1, 1});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    ValidateGpuContactStates(BufferSpan<GpuContactState>(states));
    double difference = 0, power = 0;
    for (size_t sample = 0; sample < input.size(); ++sample) {
        const auto output = BufferSpan<GpuOutput>(outputs)[sample];
        Check(output.Force == 0 && output.Output1 == 0, "Author scraping routing node has no second body or interaction force");
        difference += std::pow(output.Output0 - expected[sample], 2);
        power += expected[sample] * expected[sample];
    }
    Check(std::sqrt(difference / power) < 2e-5, "Author modal-only routing matches source recurrence");
}

void PredictionDomain() {
    using namespace surface_audio;
    auto first = Inertial();
    SetPosition(first, 9999);
    SetVelocity(first, 96000);
    bool rejected = false;
    try {
        ContactEnergy(first);
    } catch (const std::domain_error &) { rejected = true; }
    Check(rejected, "Reject clipped-free-state quadratic counterexample");
    rejected = false;
    try {
        SetPosition(first, 10000);
    } catch (const std::domain_error &) { rejected = true; }
    Check(rejected, "Reject initial position at affine domain boundary");

    std::vector<GpuContact> contacts;
    std::vector<GpuMode> modes;
    std::vector<GpuModeState> initial_modes;
    for (unsigned index = 0; index < 2; ++index) {
        auto body0 = Inertial(), body1 = Inertial();
        SetPosition(body0, 9999);
        if (index == 0) SetVelocity(body0, 96000);
        else SetPosition(body1, 9999.9);
        const auto range0 = AppendBody(body0, modes, initial_modes), range1 = AppendBody(body1, modes, initial_modes);
        GpuContact contact;
        contact.Offset0 = range0.Offset;
        contact.Count0 = range0.Count;
        contact.Offset1 = range1.Offset;
        contact.Count1 = range1.Count;
        contact.Impact.Stiffness = 1e12f;
        contacts.push_back(contact);
        ContactState state;
        rejected = false;
        try {
            StepImpact(body0, body1, state, ImpactParameters{.Stiffness = 1e12});
        } catch (const std::domain_error &) { rejected = true; }
        Check(rejected, "Reject invalid free or candidate CPU prediction");
    }
    auto gpu = CreateGpu();
    const auto kernel = CreateKernel(gpu, "SdtContacts");
    const auto dispatch = Upload(gpu, GpuDispatch{2, 2, 1.f / 48000});
    const auto configuration = Upload<GpuContact>(gpu, contacts);
    const std::array<GpuContactState, 2> initial_contacts{};
    const auto states = Upload<GpuContactState>(gpu, initial_contacts);
    const auto coefficients = Upload<GpuMode>(gpu, modes);
    const auto mode_states = Upload<GpuModeState>(gpu, initial_modes);
    const std::array<GpuInput, 4> initial_inputs{};
    const auto inputs = Upload<GpuInput>(gpu, initial_inputs);
    const auto outputs = CreateBuffer(gpu, sizeof(GpuOutput) * 4);
    const std::array bindings{GpuBinding{dispatch, 0}, GpuBinding{configuration, 1}, GpuBinding{states, 2}, GpuBinding{coefficients, 3}, GpuBinding{mode_states, 4}, GpuBinding{inputs, 5}, GpuBinding{outputs, 6}};
    BeginGpu(gpu);
    DispatchGpu(gpu, kernel, bindings, {2, 1, 1});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    for (const auto &state : BufferSpan<GpuContactState>(states)) Check(state.Error == GpuContactError::PredictionDomain && state.Energy == 0, "GPU prediction domain is explicitly reported");
    rejected = false;
    try {
        ValidateGpuContactStates(BufferSpan<GpuContactState>(states));
    } catch (const std::domain_error &) { rejected = true; }
    Check(rejected, "Host rejects invalid GPU result");
    Check(std::memcmp(mode_states.Data, initial_modes.data(), initial_modes.size() * sizeof(GpuModeState)) == 0, "Invalid GPU contact preserves body motion state");
    for (const auto &output : BufferSpan<GpuOutput>(outputs)) Check(output.Force == 0 && output.Output0 == 0 && output.Output1 == 0, "Invalid GPU contact output is zero through the block");
}
} // namespace

int main() {
    try {
        Reference();
        Physics();
        Streaming();
        LimiterConvergence();
        StationaryEquilibrium();
        TimeStepConvergence();
        SustainedRubbing();
        Gpu(64, 256);
        Gpu(2, 48000);
        Gpu(6, 192000, true);
        GpuControllers();
        EndToEndControllers();
        ExternalRouting();
        PredictionDomain();
        std::cout << "SDT tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
