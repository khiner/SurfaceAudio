#include "core/AudioFile.h"
#include "falaize/Falaize.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
using namespace surface_audio;
using namespace surface_audio::falaize;
namespace {
using Clock = std::chrono::steady_clock;
void Render(const std::filesystem::path &output, const char *name, Parameters<double> p, bool hammer, unsigned frames, double drive) {
    const auto m = MakeModel(p);
    State<double> s;
    if (hammer) s.HammerVelocity = drive;
    const double initial = Energy(m, s, hammer);
    std::vector<float> wave(frames);
    std::ofstream trace(output / (std::string(name) + ".csv"));
    trace << std::setprecision(17) << "time,displacement,velocity,force,elastic,energy,dissipation,input_power,balance_error,residual\n";
    double error{}, residual{};
    unsigned failures{};
    const auto start = Clock::now();
    for (unsigned i = 0; i < frames; ++i) {
        const auto a = Step(m, s, drive, hammer, 1e-12);
        wave[i] = static_cast<float>(a.Velocity);
        error = std::max(error, std::abs(a.Energy + s.Loss - initial));
        residual = std::max(residual, a.Residual);
        failures += a.Failed;
        trace << (i + 1) / p.SampleRate << ',' << a.Displacement << ',' << a.Velocity << ',' << a.Force << ',' << s.Elastic << ',' << a.Energy << ',' << a.Dissipation << ',' << a.InputPower << ',' << a.BalanceError << ',' << a.Residual << '\n';
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    WriteWave(output / (std::string(name) + "_velocity.wav"), static_cast<unsigned>(p.SampleRate), 1, wave);
    std::ofstream info(output / (std::string(name) + ".json"));
    info << std::setprecision(17) << "{\"energy_error\":" << error << ",\"residual\":" << residual << ",\"failed_steps\":" << failures << ",\"seconds_with_csv_io\":" << seconds << "}\n";
    if (failures) throw std::runtime_error("Falaize reproduction failed to converge");
}
void Pluck(const std::filesystem::path &output) {
    std::ofstream file(output / "pluck.json");
    file << std::setprecision(17) << "[";
    for (const unsigned count : {5u, 10u, 15u, 20u, 25u, 100u}) {
        Parameters<double> p;
        p.SampleRate = 48000;
        p.StringDamping = 0;
        p.Modes = count;
        const auto m = MakeModel(p);
        State<double> s;
        double coefficients[100]{};
        for (unsigned i = 0; i < 100; ++i) coefficients[i] = .01 * std::sqrt(2 * p.Length) * std::sin((i + 1) * std::numbers::pi * .3) / (std::pow((i + 1) * std::numbers::pi, 2) * .3 * .7);
        for (unsigned i = 0; i < count; ++i) s.Displacement[i] = coefficients[i];
        const double initial = Energy(m, s, false);
        double error{}, relative_error{};
        for (unsigned step = 0; step < 4800; ++step) {
            const auto sample = StepString(m, s);
            error = std::max(error, std::abs(sample.Energy - initial) / initial);
            double numerator{}, denominator{};
            for (unsigned i = 0; i < 100; ++i) {
                const double target = coefficients[i] * std::cos((i + 1) * std::numbers::pi / p.Length * std::sqrt(p.Tension / p.Density) * (step + 1) / p.SampleRate);
                const double d = (i < count ? s.Displacement[i] : 0) - target;
                numerator += d * d;
                denominator += target * target;
            }
            relative_error += std::sqrt(numerator / denominator) / 4800;
        }
        if (count != 5) file << ",";
        file << "{\"modes\":" << count << ",\"maximum_relative_energy_drift\":" << error << ",\"mean_relative_spatial_l2_error_against_exact_100_modes\":" << relative_error << "}";
    }
    file << "]\n";
}
void Tolerances(const std::filesystem::path &output) {
    std::ofstream file(output / "tolerances.json");
    file << std::setprecision(17) << "[";
    bool first = true;
    for (bool hammer : {false, true})
        for (double tolerance : {1e-9, 1e-12, 1e-15}) {
            const auto m = MakeModel();
            State<double> s;
            if (hammer) s.HammerVelocity = 1;
            const double initial = Energy(m, s, hammer);
            double residual{}, error{}, iterations{};
            unsigned failures{}, maximum_iterations{};
            const unsigned frames = hammer ? 9600 : 96000;
            for (unsigned i = 0; i < frames; ++i) {
                const auto a = Step(m, s, .1, hammer, tolerance);
                residual = std::max(residual, a.Residual);
                error = std::max(error, std::abs(a.Energy + s.Loss - initial));
                iterations += a.Iterations;
                maximum_iterations = std::max(maximum_iterations, a.Iterations);
                failures += a.Failed;
            }
            if (!first) file << ",";
            first = false;
            file << "{\"hammer\":" << hammer << ",\"tolerance\":" << tolerance << ",\"residual\":" << residual << ",\"energy_error\":" << error << ",\"mean_updates\":" << iterations / frames << ",\"maximum_updates\":" << maximum_iterations << ",\"failed_steps\":" << failures << "}";
        }
    file << "]\n";
}
void Benchmark(const std::filesystem::path &output, unsigned voices, unsigned frames, bool hammer) {
    Parameters<double> p;
    std::vector<float> drives(voices);
    for (unsigned i = 0; i < voices; ++i) drives[i] = (hammer ? 1.f : .1f) * (1 + .0001f * i);
    auto gpu = CreateGpu();
    const auto start = Clock::now();
    const auto batch = RenderGpu(gpu, p, drives, frames, hammer);
    const double gpu_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    double cpu64{}, cpu32{}, error{}, norm{}, maxerror{}, state_error{}, state_norm{}, worst{}, residual{}, fp32_error{}, fp32_norm{};
    unsigned failures{}, cpu64_failed{}, cpu32_failed{};
    double cpu64_balance{}, cpu32_balance{}, cpu64_residual{}, cpu32_residual{};
    double field_error[6]{}, field_norm[6]{}, fp32_state_error{}, fp32_state_norm{};
    for (unsigned voice = 0; voice < voices; ++voice) {
        auto begin = Clock::now();
        const auto m = MakeModel(p);
        State<double> s;
        if (hammer) s.HammerVelocity = drives[voice];
        std::vector<double> wave(frames);
        for (unsigned i = 0; i < frames; ++i) {
            const auto a = Step(m, s, double(drives[voice]), hammer, 1e-12);
            wave[i] = a.Velocity;
            cpu64_balance = std::max(cpu64_balance, std::abs(a.BalanceError));
            cpu64_residual = std::max(cpu64_residual, a.Residual);
            cpu64_failed += a.Failed;
        }
        cpu64 += std::chrono::duration<double>(Clock::now() - begin).count();
        double e{}, n{};
        for (unsigned i = 0; i < frames; ++i) {
            const double d = batch.Velocity[voice * frames + i] - wave[i];
            e += d * d;
            n += wave[i] * wave[i];
            maxerror = std::max(maxerror, std::abs(d));
        }
        error += e;
        norm += n;
        worst = std::max(worst, std::sqrt(e / n));
        unsigned offset = voice * (2 * p.Modes + 4);
        const auto compare = [&](double value, unsigned field) {
            const double d = batch.FinalState[offset++] - value;
            state_error += d * d;
            state_norm += value * value;
            field_error[field] += d * d;
            field_norm[field] += value * value;
        };
        for (unsigned i = 0; i < p.Modes; ++i) compare(s.Displacement[i], 0);
        for (unsigned i = 0; i < p.Modes; ++i) compare(s.Velocity[i], 1);
        unsigned field = 2;
        for (double v : {s.Elastic, s.HammerVelocity, s.ContactVelocity, s.Force}) compare(v, field++);
        begin = Clock::now();
        const auto mf = MakeFloatModel(p);
        State<float> sf;
        if (hammer) sf.HammerVelocity = drives[voice];
        std::vector<float> wf(frames);
        for (unsigned i = 0; i < frames; ++i) {
            const auto a = Step(mf, sf, drives[voice], hammer, 2e-6f);
            wf[i] = a.Velocity;
            cpu32_balance = std::max(cpu32_balance, double(std::abs(a.BalanceError)));
            cpu32_residual = std::max(cpu32_residual, double(a.Residual));
            cpu32_failed += a.Failed;
        }
        cpu32 += std::chrono::duration<double>(Clock::now() - begin).count();
        for (unsigned i = 0; i < frames; ++i) {
            const double d = wf[i] - batch.Velocity[voice * frames + i];
            fp32_error += d * d;
            fp32_norm += double(wf[i]) * wf[i];
        }
        offset = voice * (2 * p.Modes + 4);
        const auto compare32 = [&](float v) {const double d=batch.FinalState[offset++]-v;fp32_state_error+=d*d;fp32_state_norm+=double(v)*v; };
        for (unsigned i = 0; i < p.Modes; ++i) compare32(sf.Displacement[i]);
        for (unsigned i = 0; i < p.Modes; ++i) compare32(sf.Velocity[i]);
        for (float v : {sf.Elastic, sf.HammerVelocity, sf.ContactVelocity, sf.Force}) compare32(v);
        failures += batch.FailedSteps[voice];
        residual = std::max(residual, double(batch.Residual[voice]));
    }
    std::ofstream file(output / (hammer ? "hammer_benchmark.json" : "bow_benchmark.json"));
    file << std::setprecision(17) << "{\"voices\":" << voices << ",\"frames\":" << frames << ",\"cpu_fp64_seconds\":" << cpu64 << ",\"cpu_fp32_seconds\":" << cpu32 << ",\"gpu_fp32_seconds\":" << gpu_seconds << ",\"waveform_l2\":" << std::sqrt(error / norm) << ",\"worst_voice_l2\":" << worst << ",\"maximum_velocity_error\":" << maxerror << ",\"final_state_l2\":" << std::sqrt(state_error / state_norm) << ",\"cpu32_gpu_waveform_l2\":" << std::sqrt(fp32_error / fp32_norm) << ",\"gpu_residual\":" << residual << ",\"failed_steps\":" << failures << ",\"cpu64_failed_steps\":" << cpu64_failed << ",\"cpu32_failed_steps\":" << cpu32_failed << ",\"cpu64_maximum_balance_error\":" << cpu64_balance << ",\"cpu32_maximum_balance_error\":" << cpu32_balance << ",\"cpu64_residual\":" << cpu64_residual << ",\"cpu32_residual\":" << cpu32_residual << ",\"cpu32_gpu_final_state_l2\":" << std::sqrt(fp32_state_error / std::max(fp32_state_norm, 1e-100)) << ",\"final_state_field_relative_l2\":[";
    const auto write_fields = [&](const char *next, auto value) {
        for (unsigned i = 0; i < 6; ++i) file << (i ? "," : "") << std::sqrt(value(i));
        file << next;
    };
    write_fields("],\"final_state_field_rmse\":[", [&](unsigned i) { return field_error[i] / std::max(field_norm[i], 1e-100); });
    write_fields("],\"final_state_field_reference_rms\":[", [&](unsigned i) { return field_error[i] / (voices * (i < 2 ? p.Modes : 1)); });
    write_fields("]}\n", [&](unsigned i) { return field_norm[i] / (voices * (i < 2 ? p.Modes : 1)); });
    if (failures) throw std::runtime_error("Falaize GPU reproduction failed");
}
}
int main(int argc, char **argv) {
    try {
        const std::filesystem::path output = argc > 1 ? argv[1] : "outputs/reproduction/falaize";
        std::filesystem::create_directories(output);
        const unsigned voices = argc > 2 ? static_cast<unsigned>(std::stoul(argv[2])) : 128;
        Pluck(output);
        Tolerances(output);
        Parameters<double> p;
        Render(output, "bow_modal", p, false, 96000, .1);
        Render(output, "hammer_modal", p, true, 9600, 1);
        p.Elements = 20;
        Render(output, "bow_fem", p, false, 96000, .1);
        Render(output, "hammer_fem", p, true, 9600, 1);
        p.Elements = 0;
        p.HammerStiffness /= p.HammerThickness;
        Render(output, "hammer_figure", p, true, 9600, 1);
        p.Tension /= 64;
        Render(output, "hammer_figure55", p, true, 9600, 1);
        p = {};
        p.Theta = .5;
        Render(output, "bow_midpoint", p, false, 96000, .1);
        if (voices) {
            Benchmark(output, voices, 96000, false);
            Benchmark(output, voices, 9600, true);
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
