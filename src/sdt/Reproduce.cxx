// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/AudioFile.h"
#include "core/BinaryFile.h"
#include "core/Gpu.h"
#include "sdt/SdtGpu.h"
#include "sdt/reference/Upstream.h"

#include <array>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>

using namespace surface_audio;
using namespace surface_audio::sdt;
namespace {
struct Api {
    void *Library;
};
template<typename T> T Function(const Api &api, const char *name) {
    auto symbol = dlsym(api.Library, name);
    if (!symbol) throw std::runtime_error(std::string("missing upstream symbol ") + name);
    return reinterpret_cast<T>(symbol);
}
void Set(const Api &api, const char *name, void *object, double value) { Function<void (*)(void *, double)>(api, name)(object, value); }
struct Preset {
    std::string Name;
    GpuContactKind Kind{GpuContactKind::Impact};
    bool Surface{}, Scraping{};
    double Gain{50000}, Strike{}, Drive{}, Mass{.01};
    ImpactParameters Impact{1e8, .8, 1.5};
    FrictionParameters Friction{.5, .1, .95, .15, .2, 500, 20, 14, .5};
    RollingParameters Rolling{.001, 100, .01, 2};
    ScrapingParameters Scrape{.001, 2, 1};
    std::array<ModeParameters, 3> Modes{{{500, .03, 1, 100, 100}, {1300, .02, 1, 100, 100}, {1700, .01, 1, 100, 100}}};
};
struct Trace {
    double Force{}, Output{}, Position0{}, Position1{}, Velocity0{}, Velocity1{}, External{};
};
struct Signals {
    std::vector<float> Height, Noise, External;
};

Signals Inputs(const Preset &preset, size_t frames, double rate) {
    Signals inputs{std::vector<float>(frames), std::vector<float>(frames), std::vector<float>(frames)};
    // Pd 0.55-2 noise~ and lop~ scalar equations. First noise~ instance seed.
    uint32_t random = 307U * 1319U, sdt_random = 42;
    float last = 0, coefficient = 20 * static_cast<float>(2 * 3.14159 / rate);
    const float feedback = 1 - coefficient;
    for (size_t frame = 0; frame < frames; ++frame) {
        const float white = static_cast<float>(static_cast<int64_t>(random & 0x7fffffffU) - 0x40000000) * (1.0f / 0x40000000);
        random = random * 435898247U + 382842987U;
        const float feed = coefficient * white, history = feedback * last;
        last = feed + history;
        inputs.Height[frame] = 10 * last;
        sdt_random = sdt_random * 1664525U + 1013904223U;
        inputs.Noise[frame] = static_cast<float>(static_cast<double>(sdt_random) / 0x7fffffff - 1);
        inputs.External[frame] = static_cast<float>(preset.Drive);
    }
    return inputs;
}

Body FirstBody(const Preset &preset, double rate) {
    if (preset.Scraping) return MakeBody(preset.Modes, rate);
    const std::array<ModeParameters, 1> inertia{{{0, 0, preset.Mass, 1, 1}}};
    auto body = MakeBody(inertia, rate);
    SetVelocity(body, preset.Strike);
    return body;
}

void *UpstreamBody(const Api &api, std::span<const ModeParameters> modes) {
    auto object = Function<void *(*)(unsigned, unsigned)>(api, "SDTResonator_new")(static_cast<unsigned>(modes.size()), 1);
    auto mode_set = [&](const char *name, unsigned index, double value) { Function<void (*)(void *, unsigned, double)>(api, name)(object, index, value); };
    for (unsigned mode = 0; mode < modes.size(); ++mode) {
        mode_set("SDTResonator_setFrequency", mode, modes[mode].Frequency);
        mode_set("SDTResonator_setDecay", mode, modes[mode].Decay);
        mode_set("SDTResonator_setWeight", mode, modes[mode].Mass);
        Function<void (*)(void *, unsigned, unsigned, double)>(api, "SDTResonator_setGain")(object, 0, mode, modes[mode].ContactGain);
    }
    Set(api, "SDTResonator_setFragmentSize", object, 1);
    Function<void (*)(void *, unsigned)>(api, "SDTResonator_setActiveModes")(object, static_cast<unsigned>(modes.size()));
    return object;
}

std::vector<Trace> Upstream(const Api &api, const Preset &preset, const Signals &inputs, double rate) {
    Function<void (*)(double)>(api, "SDT_setSampleRate")(rate);
    *Function<unsigned *>(api, "seed") = 42;
    const std::array<ModeParameters, 1> inertia{{{0, 0, preset.Mass, 1, 1}}};
    auto first = UpstreamBody(api, preset.Scraping ? std::span<const ModeParameters>(preset.Modes) : std::span<const ModeParameters>(inertia));
    auto second = preset.Scraping ? nullptr : UpstreamBody(api, preset.Modes);
    auto interactor = Function<void *(*)()>(api, preset.Kind == GpuContactKind::Friction ? "SDTFriction_new" : "SDTImpact_new")();
    Function<void (*)(void *, void *)>(api, "SDTInteractor_setFirstResonator")(interactor, first);
    Function<void (*)(void *, void *)>(api, "SDTInteractor_setSecondResonator")(interactor, second);
    if (preset.Kind == GpuContactKind::Friction) {
        const auto &p = preset.Friction;
        for (auto [name, value] : std::initializer_list<std::pair<const char *, double>>{{"NormalForce", p.NormalForce}, {"StribeckVelocity", p.StribeckVelocity}, {"StaticCoefficient", p.StaticCoefficient}, {"DynamicCoefficient", p.DynamicCoefficient}, {"BreakAway", p.BreakAway}, {"Stiffness", p.Stiffness}, {"Dissipation", p.Dissipation}, {"Viscosity", p.Viscosity}, {"Noisiness", p.Noisiness}}) Set(api, (std::string("SDTFriction_set") + name).c_str(), interactor, value);
    } else {
        Set(api, "SDTImpact_setStiffness", interactor, preset.Impact.Stiffness);
        Set(api, "SDTImpact_setDissipation", interactor, preset.Impact.Dissipation);
        Set(api, "SDTImpact_setShape", interactor, preset.Impact.Shape);
    }
    auto set_velocity = Function<void (*)(void *, unsigned, double)>(api, "SDTResonator_setVelocity");
    set_velocity(first, 0, preset.Strike);
    void *controller = nullptr;
    double (*surface_step)(void *, double) = nullptr;
    if (preset.Surface) {
        const std::string prefix = preset.Scraping ? "SDTScraping" : "SDTRolling";
        controller = Function<void *(*)()>(api, (prefix + "_new").c_str())();
        Set(api, (prefix + "_setGrain").c_str(), controller, preset.Scraping ? preset.Scrape.Grain : preset.Rolling.Grain);
        Set(api, (prefix + "_setVelocity").c_str(), controller, preset.Scraping ? preset.Scrape.Velocity : preset.Rolling.Velocity);
        if (preset.Scraping) Set(api, "SDTScraping_setForce", controller, preset.Scrape.Force);
        else {
            Set(api, "SDTRolling_setMass", controller, preset.Rolling.Mass);
            Set(api, "SDTRolling_setDepth", controller, preset.Rolling.Depth);
        }
        surface_step = Function<double (*)(void *, double)>(api, (prefix + "_dsp").c_str());
    }
    auto apply = Function<void (*)(void *, unsigned, double)>(api, "SDTResonator_applyForce");
    auto compute = Function<double (*)(void *)>(api, "SDTInteractor_computeForce");
    auto step = Function<void (*)(void *)>(api, "SDTResonator_dsp");
    auto position = Function<double (*)(void *, unsigned)>(api, "SDTResonator_getPosition");
    auto velocity = Function<double (*)(void *, unsigned)>(api, "SDTResonator_getVelocity");
    std::vector<Trace> trace(inputs.Height.size());
    for (size_t frame = 0; frame < trace.size(); ++frame) {
        // Pd signal wires store t_sample float, including the controller output.
        const double external = controller ? static_cast<float>(surface_step(controller, inputs.Height[frame])) : inputs.External[frame];
        apply(first, 0, external);
        const double force = second ? compute(interactor) : 0;
        if (second) {
            apply(first, 0, force);
            apply(second, 0, -force);
        }
        step(first);
        if (second) step(second);
        trace[frame] = {force, position(second ? second : first, 0), position(first, 0), second ? position(second, 0) : 0, velocity(first, 0), second ? velocity(second, 0) : 0, external};
    }
    Function<void (*)(void *)>(api, preset.Kind == GpuContactKind::Friction ? "SDTFriction_free" : "SDTImpact_free")(interactor);
    Function<void (*)(void *)>(api, "SDTResonator_free")(first);
    if (second) Function<void (*)(void *)>(api, "SDTResonator_free")(second);
    if (controller) Function<void (*)(void *)>(api, preset.Scraping ? "SDTScraping_free" : "SDTRolling_free")(controller);
    return trace;
}

std::vector<Trace> Cpu(const Preset &preset, const Signals &inputs, double rate, bool source) {
    auto first = FirstBody(preset, rate), second = MakeBody(preset.Modes, rate);
    ContactState contact;
    RollingState rolling;
    ScrapingState scraping;
    uint32_t random = 42;
    std::vector<Trace> trace(inputs.Height.size());
    for (size_t frame = 0; frame < trace.size(); ++frame) {
        const double external = preset.Surface ? static_cast<float>(preset.Scraping ? StepScraping(preset.Scrape, scraping, double(inputs.Height[frame])) : StepRolling(preset.Rolling, rolling, double(inputs.Height[frame]))) : inputs.External[frame];
        random = random * 1664525U + 1013904223U;
        const double noise = double(random) / 0x7fffffff - 1;
        ContactSample sample;
        if (preset.Scraping) {
            if (source) reference::ApplyForce(first, external);
            else ApplyForce(first, external);
            if (source) reference::StepBody(first);
            else StepBody(first);
            sample = {0, Position(first), 0, Velocity(first), 0, Output(first), 0};
        } else if (preset.Kind == GpuContactKind::Friction) sample = source ? reference::StepFriction(first, second, contact, preset.Friction, noise, external) : StepFriction(first, second, contact, preset.Friction, noise, external);
        else sample = source ? reference::StepImpact(first, second, contact, preset.Impact, external) : StepImpact(first, second, contact, preset.Impact, external);
        trace[frame] = {sample.Force, preset.Scraping ? sample.Output0 : sample.Output1, sample.Position0, sample.Position1, sample.Velocity0, sample.Velocity1, external};
    }
    return trace;
}

std::vector<Trace> Metal(Gpu &gpu, const Preset &preset, const Signals &inputs, double rate) {
    std::vector<GpuMode> modes;
    std::vector<GpuModeState> states;
    auto first = AppendBody(FirstBody(preset, rate), modes, states);
    auto second = preset.Scraping ? GpuBodyRange{} : AppendBody(MakeBody(preset.Modes, rate), modes, states);
    const auto &p = preset.Friction;
    GpuContact contact{first.Offset, first.Count, second.Offset, second.Count, preset.Kind, {float(preset.Impact.Stiffness), float(preset.Impact.Dissipation), float(preset.Impact.Shape)}, {float(p.NormalForce), float(p.StribeckVelocity), float(p.StaticCoefficient), float(p.DynamicCoefficient), float(p.BreakAway), float(p.Stiffness), float(p.Dissipation), float(p.Viscosity), float(p.Noisiness)}};
    const auto frames = static_cast<uint32_t>(inputs.Height.size());
    auto dispatch = Upload(gpu, GpuDispatch{1, frames, float(1 / rate)});
    auto configuration = Upload(gpu, contact), contact_state = Upload(gpu, GpuContactState{});
    auto mode_buffer = Upload<GpuMode>(gpu, modes), state_buffer = Upload<GpuModeState>(gpu, states);
    std::vector<GpuInput> input(frames);
    for (size_t frame = 0; frame < frames; ++frame) input[frame] = {inputs.External[frame], 0, inputs.Noise[frame]};
    auto input_buffer = Upload<GpuInput>(gpu, input), output_buffer = CreateBuffer(gpu, frames * sizeof(GpuOutput));
    auto kernel = CreateKernel(gpu, "SdtContacts");
    GpuBuffer heights{}, surface_parameters{}, surface_state{};
    GpuKernel surface_kernel{};
    if (preset.Surface) {
        heights = Upload<float>(gpu, inputs.Height);
        surface_parameters = Upload(gpu, GpuSurfaceParameters{preset.Scraping ? GpuSurfaceKind::Scraping : GpuSurfaceKind::Rolling, {.001f, 100, .01f, 2}, {.001f, 2, 1}});
        surface_state = Upload(gpu, GpuSurfaceState{});
        surface_kernel = CreateKernel(gpu, "SdtSurfaceForces");
    }
    BeginGpu(gpu);
    if (preset.Surface) DispatchGpu(gpu, surface_kernel, std::array{GpuBinding{dispatch, 0}, GpuBinding{surface_parameters, 1}, GpuBinding{surface_state, 2}, GpuBinding{heights, 3}, GpuBinding{input_buffer, 4}}, {1, 1, 1}, {1, 1, 1});
    DispatchGpu(gpu, kernel, std::array{GpuBinding{dispatch, 0}, GpuBinding{configuration, 1}, GpuBinding{contact_state, 2}, GpuBinding{mode_buffer, 3}, GpuBinding{state_buffer, 4}, GpuBinding{input_buffer, 5}, GpuBinding{output_buffer, 6}}, {1, 1, 1}, {1, 1, 1});
    SubmitGpu(gpu);
    WaitGpu(gpu);
    ValidateGpuContactStates(BufferSpan<GpuContactState>(contact_state));
    std::vector<Trace> trace(frames);
    const auto output = BufferSpan<GpuOutput>(output_buffer);
    const auto actual_input = BufferSpan<GpuInput>(input_buffer);
    for (size_t frame = 0; frame < frames; ++frame) {
        const auto &sample = output[frame];
        trace[frame] = {sample.Force, preset.Scraping ? sample.Output0 : sample.Output1, sample.Position0, sample.Position1, sample.Velocity0, sample.Velocity1, actual_input[frame].External0};
    }
    return trace;
}

void Save(const std::filesystem::path &directory, const std::string &name, std::span<const Trace> trace, const Preset &preset, uint32_t rate) {
    static_assert(sizeof(Trace) == 7 * sizeof(double));
    WriteBinary<Trace>(directory / (name + ".f64"), trace);
    std::vector<float> audio(trace.size());
    for (size_t frame = 0; frame < trace.size(); ++frame) audio[frame] = static_cast<float>(trace[frame].Output) * static_cast<float>(preset.Gain);
    WriteWave(directory / (name + ".wav"), rate, 1, audio);
}
void Validate(const Preset &preset, std::span<const Trace> upstream, std::span<const Trace> source, std::span<const Trace> cpu, std::span<const Trace> metal) {
    auto check = [&](std::span<const Trace> a, std::span<const Trace> b, double Trace::*field, double tolerance) {
        double power = 0, difference = 0;
        for (size_t frame = 0; frame < a.size(); ++frame) {
            if (!std::isfinite(a[frame].*field) || !std::isfinite(b[frame].*field)) throw std::runtime_error(preset.Name + " nonfinite reproduction trace");
            power += std::pow(a[frame].*field, 2);
            difference += std::pow(a[frame].*field - b[frame].*field, 2);
        }
        if (difference > tolerance * tolerance * power + 1e-28) throw std::runtime_error(preset.Name + " reproduction accuracy regression");
    };
    for (auto field : {&Trace::Force, &Trace::Output}) {
        check(upstream, source, field, 1e-7);
        check(cpu, metal, field, preset.Name == "rolling" ? .02 : 1e-4);
    }
    for (const auto &trace : {upstream, source, cpu, metal}) {
        for (const auto &sample : trace) {
            if (!std::isfinite(sample.Position0) || !std::isfinite(sample.Position1) || !std::isfinite(sample.Velocity0) || !std::isfinite(sample.Velocity1) || !std::isfinite(sample.External)) throw std::runtime_error(preset.Name + " invalid motion state");
        }
    }
}

void Compare(std::ostream &json, std::span<const Trace> reference_trace, std::span<const Trace> trace) {
    auto metric = [&](double Trace::*field) {
        double difference = 0, scale = 0, maximum = 0, power = 0;
        for (size_t frame = 0; frame < trace.size(); ++frame) {
            const double error = trace[frame].*field - reference_trace[frame].*field;
            difference += error * error;
            scale += std::pow(reference_trace[frame].*field, 2);
            power += std::pow(trace[frame].*field, 2);
            maximum = std::max(maximum, std::abs(error));
        }
        json << "{\"relative_l2\":" << (scale ? std::sqrt(difference / scale) : 0) << ",\"max_absolute\":" << maximum << ",\"rms\":" << std::sqrt(power / trace.size()) << '}';
    };
    json << "{\"force\":";
    metric(&Trace::Force);
    json << ",\"output\":";
    metric(&Trace::Output);
    json << ",\"external\":";
    metric(&Trace::External);
    json << ",\"terminal_velocity0\":" << trace.back().Velocity0 << '}';
}
} // namespace

int main(int argc, char **argv) try {
    if (argc < 3 || argc > 5) throw std::invalid_argument("usage: sdtReproduce <upstream-dylib> <output-directory> [seconds=6] [sample-rate=44100]");
    const double seconds = argc > 3 ? std::stod(argv[3]) : 6;
    const auto rate = argc > 4 ? static_cast<uint32_t>(std::stoul(argv[4])) : 44100U;
    if (!(seconds > 0 && seconds <= 60) || rate < 24000 || rate > 192000) throw std::invalid_argument("invalid render duration/rate");
    Api api{dlopen(argv[1], RTLD_NOW | RTLD_LOCAL)};
    if (!api.Library) throw std::runtime_error(dlerror());
    const std::filesystem::path output_directory(argv[2]);
    std::filesystem::create_directories(output_directory);
    Preset impact{.Name = "impact", .Gain = 1000, .Strike = -3, .Impact = {1e7, 1, 1.5}};
    impact.Modes[0].Decay = 1;
    impact.Modes[1].Decay = .5;
    impact.Modes[2].Decay = .25;
    std::array presets{impact, Preset{.Name = "rolling", .Surface = true}, Preset{.Name = "scraping", .Kind = GpuContactKind::External, .Surface = true, .Scraping = true}, Preset{.Name = "friction", .Kind = GpuContactKind::Friction, .Gain = 10000, .Drive = -1}, Preset{.Name = "friction_startup", .Kind = GpuContactKind::Friction, .Gain = 10000}};
    // Pd's messages carry 32-bit t_float values before SDT receives doubles.
    for (auto &preset : presets) {
        auto pd_float = [](double &value) { value = static_cast<float>(value); };
        pd_float(preset.Mass);
        for (double *value : {&preset.Impact.Stiffness, &preset.Impact.Dissipation, &preset.Impact.Shape}) pd_float(*value);
        for (auto &mode : preset.Modes) {
            pd_float(mode.Decay);
            pd_float(mode.Mass);
        }
        auto &f = preset.Friction;
        for (double *value : {&f.NormalForce, &f.StribeckVelocity, &f.StaticCoefficient, &f.DynamicCoefficient, &f.BreakAway, &f.Stiffness, &f.Dissipation, &f.Viscosity, &f.Noisiness}) pd_float(*value);
        auto &r = preset.Rolling;
        for (double *value : {&r.Grain, &r.Depth, &r.Mass, &r.Velocity}) pd_float(*value);
        auto &s = preset.Scrape;
        for (double *value : {&s.Grain, &s.Force, &s.Velocity}) pd_float(*value);
        // Stribeck is not messaged by this help patch: retain the C default .1.
        preset.Friction.StribeckVelocity = .1;
    }
    auto gpu = CreateGpu();
    std::ofstream metrics(output_directory / "metrics.json");
    metrics << std::setprecision(17) << "{\"sample_rate\":" << rate << ",\"seconds\":" << seconds << ",\"trace_columns\":[\"force\",\"output\",\"position0\",\"position1\",\"velocity0\",\"velocity1\",\"external\"],\"presets\":{";
    std::ofstream cases(output_directory / "cases.json");
    cases << "{\"cases\":[";
    bool first_case = true;
    bool first = true;
    for (const auto &preset : presets) {
        std::cout << "Rendering author Pd " << preset.Name << std::endl;
        const auto inputs = Inputs(preset, static_cast<size_t>(seconds * rate), rate);
        const auto upstream = Upstream(api, preset, inputs, rate), source = Cpu(preset, inputs, rate, true), refined = Cpu(preset, inputs, rate, false), metal = Metal(gpu, preset, inputs, rate);
        Validate(preset, upstream, source, refined, metal);
        Save(output_directory, preset.Name + "_upstream", upstream, preset, rate);
        Save(output_directory, preset.Name + "_source", source, preset, rate);
        Save(output_directory, preset.Name + "_cpu", refined, preset, rate);
        Save(output_directory, preset.Name + "_metal", metal, preset, rate);
        for (const std::string variant : {"source", "metal"}) {
            if (!first_case) cases << ',';
            first_case = false;
            const std::string notes = "Pinned Pd help graph and parameters with a preserved common float surface realization. " +
                std::string(preset.Name == "impact" ? "Author strike 0 -3 activated at sample 0. " : preset.Name == "friction" ? "The author lateral-force slider is set to -1 N at sample 0; this recorded gesture is ours. " :
                                preset.Name == "friction_startup"                                                              ? "Author startup lateral force 0: expected silence. " :
                                                                                                                                 "Author startup settings. ") +
                (variant == "source" ? "C++ source recurrence and original 0.001-tolerance limiter compared with the actual unmodified C library." : "Production Metal uses the refined analytic limiter. Rolling waveform differences from the original limiter remain measurable; see metrics.json.");
            cases << "{\"title\":\"SDT Pd " << preset.Name << " / " << variant << "\",\"reference\":\"" << (output_directory / (preset.Name + "_upstream.wav")).generic_string() << "\",\"synthesis\":\"" << (output_directory / (preset.Name + "_" + variant + ".wav")).generic_string() << "\",\"notes\":\"" << notes << "\",\"source_url\":\"https://github.com/SkAT-VG/SDT/blob/0509de418e7bebc8b37866b3b4458e0acc8cf1f4/Pd/" << (preset.Name == "friction_startup" ? "friction" : preset.Name) << "~-help.pd\"}";
        }
        WriteBinary<float>(output_directory / (preset.Name + "_height.f32"), inputs.Height);
        if (!first) metrics << ',';
        first = false;
        metrics << '"' << preset.Name << "\":{\"authored_gain\":" << preset.Gain << ",\"drive\":" << preset.Drive << ",\"source_vs_upstream\":";
        Compare(metrics, upstream, source);
        metrics << ",\"cpu_vs_upstream\":";
        Compare(metrics, upstream, refined);
        metrics << ",\"metal_vs_cpu\":";
        Compare(metrics, refined, metal);
        metrics << ",\"metal_vs_upstream\":";
        Compare(metrics, upstream, metal);
        metrics << '}';
    }
    metrics << "}}\n";
    const auto historical = std::filesystem::path(argv[1]).parent_path() / "RollSDT.wav";
    if (std::filesystem::exists(historical)) cases << ",{\"title\":\"Historical SDT rolling recording / current author Pd preset\",\"reference\":\"" << historical.generic_string() << "\",\"synthesis\":\"" << (output_directory / "rolling_source.wav").generic_string() << "\",\"notes\":\"Historical author-linked recording references the 2010 SDT. Its resonator, surface realization, gestures and version are unidentified. This is a qualitative comparison, not an exact recording reproduction or accuracy gate.\",\"source_url\":\"https://kronland.fr/publications/a-synthesis-model-with-intuitive-control-capabilities-for-rolling-sounds/\"}";
    cases << "]}\n";
    dlclose(api.Library);
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
