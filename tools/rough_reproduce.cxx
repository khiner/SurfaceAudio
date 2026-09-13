#include "core/AudioFile.h"
#include "core/GpuDecimate.h"
#include "core/GpuResample.h"
#include "rough/ContactGpu.h"
#include "rough/PlateContactGpu.h"
#include "rough/Profile.h"
#include "rough/Slider.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <numbers>
#include <stdexcept>
#include <tuple>

using namespace surface_audio;
using namespace surface_audio::rough;
namespace {
struct Roughness {
    unsigned Label;
    double Rms, Correlation;
};
constexpr std::array roughness{Roughness{3, 3.57e-6, 400e-6}, Roughness{5, 6.02e-6, 450e-6}, Roughness{8, 9.65e-6, 450e-6}, Roughness{10, 11.88e-6, 500e-6}, Roughness{20, 25.80e-6, 500e-6}, Roughness{30, 38.61e-6, 500e-6}};
constexpr double DangTimeStep = 1e-7, DangSpacing = 5e-6, DangOffset = .1;
constexpr unsigned DangEventStride = 100;
struct DangScene {
    Model Contact;
    double Separation;
};
std::vector<float> Audition(Gpu &gpu, std::span<const float> velocity, std::span<const unsigned> factors) {
    auto input = Upload<float>(gpu, velocity);
    unsigned frames = unsigned(velocity.size());
    for (unsigned factor : factors) {
        const auto plan = CreateDecimatePlan(gpu, frames, 1, factor);
        const auto output = CreateBuffer(gpu, plan.OutputFrames * sizeof(float));
        BeginGpu(gpu);
        DispatchDecimateGpu(gpu, plan, input, output);
        SubmitGpu(gpu);
        WaitGpu(gpu);
        input = output;
        frames = plan.OutputFrames;
    }
    const auto values = BufferSpan<float>(input);
    const auto coefficients = KaiserDecimateCoefficients(500);
    std::vector<float> taps(coefficients.size());
    for (size_t i = 0; i < taps.size(); ++i) taps[i] = float(441 * coefficients[i]);
    const unsigned output_frames = unsigned((uint64_t(frames) * 441 + 499) / 500);
    const FirResampleJob job{0, frames, 0, output_frames, 0, unsigned(taps.size()), 441, 500, int(taps.size() / 2)};
    return ResampleFirGpu(gpu, values, taps, std::span{&job, 1});
}
double SaveVelocity(Gpu &gpu, const std::filesystem::path &output, std::span<const float> velocity, std::span<const unsigned> factors) {
    auto audio = Audition(gpu, velocity, factors);
    double peak{};
    for (float value : audio) peak = std::max(peak, std::abs(double(value)));
    const double gain = peak > 0 ? .85 / peak : 1;
    for (float &value : audio) value *= gain;
    std::filesystem::create_directories(output);
    WriteWave(output / "vibration.wav", 44100, 1, audio);
    return gain;
}
void ProfileMetadata(std::ostream &stream, std::span<const double> values) {
    const double mean = std::accumulate(values.begin(), values.end(), 0.) / values.size();
    double square{}, absolute{};
    for (double h : values) {
        square += (h - mean) * (h - mean);
        absolute += std::abs(h - mean);
    }
    stream << "{\"nodes\":" << values.size() << ",\"mean_m\":" << mean << ",\"rq_m\":" << std::sqrt(square / values.size()) << ",\"ra_m\":" << absolute / values.size() << '}';
}
void EventMetadata(std::ostream &stream, const ContactEventStatistics &events) {
    stream << "{\"completed\":" << events.Completed << ",\"left_censored\":" << events.LeftCensored << ",\"right_censored\":" << events.RightCensored
           << ",\"frames\":" << events.Frames << ",\"recorded_samples\":" << events.Samples << ",\"total_nodal_work_j\":" << events.Work << ",\"zero_work_events\":" << events.ZeroWork
           << ",\"force_thresholds_n\":[0.78,7.8,78],\"force_below_threshold\":[" << events.ForceBelowPaperWeights[0] << ',' << events.ForceBelowPaperWeights[1] << ',' << events.ForceBelowPaperWeights[2]
           << "],\"duration_below_100us\":" << events.ShorterThan100Microseconds << ",\"histograms\":{";
    unsigned index{};
    for (const auto &[name, minimum, maximum, values] : {
             std::tuple{"force_n", EventForceLogMinimum, EventForceLogMaximum, &events.Force},
             std::tuple{"duration_s", EventDurationLogMinimum, EventDurationLogMaximum, &events.Duration},
             std::tuple{"positive_work_j", EventWorkLogMinimum, EventWorkLogMaximum, &events.PositiveWork},
             std::tuple{"negative_work_magnitude_j", EventWorkLogMinimum, EventWorkLogMaximum, &events.NegativeWork}
         }) {
        stream << (index++ ? "," : "") << '"' << name << "\":{\"log10_minimum\":" << minimum << ",\"log10_maximum\":" << maximum
               << ",\"interior_bins\":" << EventHistogramBins << ",\"counts\":[";
        for (unsigned i = 0; i < values->size(); ++i) stream << (i ? "," : "") << (*values)[i];
        stream << "]}";
    }
    stream << "}}";
}
std::vector<double> DangReceiver(const Model &m) {
    std::vector<double> receiver(m.Bottom.Modes.size() + m.Top.Modes.size());
    for (unsigned k = 0; k < m.Bottom.Modes.size(); ++k) receiver[k] = m.Bottom.Shapes[k * m.Bottom.Height.size() + 45000];
    return receiver;
}
DangScene MakeDang(Gpu &gpu, const Roughness &setting, uint64_t seed, double dt = DangTimeStep, double penalty = 2.1e12) {
    constexpr double spacing = DangSpacing, young = 210e9, density = 7800, width = 1;
    const auto bottom_height = GaussianProfile(gpu, 90001, spacing, setting.Rms, setting.Correlation, seed);
    const auto top_height = GaussianProfile(gpu, 4001, spacing, setting.Rms, setting.Correlation, seed ^ 0x9e3779b97f4a7c15ULL);
    const BeamProperties bottom{.Length = .45, .MassPerLength = density * width * .002, .BendingStiffness = young * width * std::pow(.002, 3) / 12, .DampingRatio = .02};
    const BeamProperties top{.Length = .02, .MassPerLength = density * width * .005, .BendingStiffness = young * width * std::pow(.005, 3) / 12, .DampingRatio = .02, .Boundary = BeamBoundary::Free};
    auto model = MakeModel(MakeBeamSurface(bottom, 40, bottom_height, dt), MakeBeamSurface(top, 2, top_height, dt, -9.81), dt, penalty);
    auto state = MakeState(model);
    PrepareContact(model, state, DangOffset, 0);
    const double separation = -*std::min_element(state.Gap.begin(), state.Gap.begin() + state.Rows);
    return {std::move(model), separation};
}
GpuContact UploadDang(Gpu &gpu, const DangScene &scene, unsigned refresh = 32) {
    return MakeGpuContact(gpu, scene.Contact, MakeState(scene.Contact), DangReceiver(scene.Contact), DangEventStride, refresh);
}
void DangBenchmark(unsigned steps, unsigned batch, unsigned refresh) {
    if (!steps || steps > 100000 || !batch || batch > 21 || !refresh || refresh > 256) throw std::invalid_argument("Dang benchmark requires 1–100000 steps and 1–21 trajectories and a refresh interval of 1–256");
    auto gpu = CreateGpu();
    const auto scene = MakeDang(gpu, roughness[1], 2013);
    std::array<std::vector<GpuContact>, 2> models;
    std::array<std::vector<GpuContactRun>, 2> runs;
    for (unsigned method = 0; method < 2; ++method) {
        models[method].reserve(batch);
        for (unsigned i = 0; i < batch; ++i) {
            models[method].push_back(UploadDang(gpu, scene, method ? refresh : 1));
            runs[method].push_back({&models[method].back(), steps, DangOffset, scene.Separation, .02 + .68 * (i + 1) / batch});
        }
    }
    std::array<double, 2> elapsed{};
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        std::array<std::vector<GpuTrace>, 2> traces;
        for (unsigned order = 0; order < 2; ++order) {
            const unsigned method = (order + repeat) % 2;
            const auto start = std::chrono::steady_clock::now();
            traces[method] = RenderGpu(gpu, runs[method]);
            elapsed[method] += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
        for (unsigned i = 0; i < batch; ++i) {
            if (traces[0][i] != traces[1][i])
                throw std::runtime_error("Contact screening changed the full-grid trajectory");
            for (auto &method : runs) method[i].Offset = DangOffset + (repeat + 1) * steps * DangTimeStep * method[i].Speed;
        }
    }
    std::cout << "Dang: 90001 + 4001 nodes, 40 + 4 modes, dt=1e-7 s, event stride=100, " << batch << " trajectories, " << 3 * steps << " steps each, refresh=" << refresh << "\n"
              << "Full displacement: " << elapsed[0] << " s; screened: " << elapsed[1] << " s; speedup: " << elapsed[0] / elapsed[1] << "x; exact outputs, states and events\n";
}
void SaveDang(Gpu &gpu, const std::filesystem::path &output, const DangScene &scene, const GpuTrace &trace, const Roughness &setting, double speed, uint64_t seed, double seconds, unsigned batch_size) {
    const unsigned steps = unsigned(trace.Velocity.size());
    constexpr double dt = DangTimeStep, spacing = DangSpacing, offset = DangOffset, width = 1;
    const double gain = SaveVelocity(gpu, output, trace.Velocity, std::array{100u, 2u});
    std::ofstream metrics(output / "Metrics.json");
    if (!metrics) throw std::runtime_error("Cannot write Dang metrics");
    metrics << std::setprecision(17) << "{\"paper\":\"Dang 2013\",\"profile_generation\":\"exact_period\",\"duration_s\":" << steps * dt << ",\"time_step_s\":" << dt
            << ",\"spacing_m\":" << spacing << ",\"speed_m_s\":" << speed << ",\"offset_m\":" << offset << ",\"initial_separation_m\":" << scene.Separation
            << ",\"event_recording_stride\":" << DangEventStride << ",\"event_sample_rate_hz\":" << 1 / (dt * DangEventStride)
            << ",\"bottom_seed\":\"" << seed << "\",\"top_seed\":\"" << (seed ^ 0x9e3779b97f4a7c15ULL) << "\",\"ra_label_um\":" << setting.Label
            << ",\"target_rq_m\":" << setting.Rms << ",\"correlation_m\":" << setting.Correlation << ",\"width_m\":" << width
            << ",\"slider_weight_n\":" << 7800 * width * .005 * .02 * 9.81 << ",\"bottom_modes\":40,\"top_bending_modes\":2,\"top_rigid_modes\":2"
            << ",\"full_overlap_end_s\":" << (.45 - .02 - offset) / speed << ",\"all_overlap_end_s\":" << (.45 - offset) / speed
            << ",\"render_wall_s\":" << seconds << ",\"render_batch_size\":" << batch_size << ",\"wall_per_simulated_second\":" << seconds / (batch_size * steps * dt)
            << ",\"mean_normal_force_n\":" << std::accumulate(trace.Force.begin(), trace.Force.end(), 0.) / steps
            << ",\"space_time_velocity_rms_m_s\":" << std::sqrt(std::accumulate(trace.MeanSquareVelocity.begin(), trace.MeanSquareVelocity.end(), 0.) / steps)
            << ",\"audio_gain\":" << gain << ",\"audio_quantity\":\"midpoint surface velocity; peak normalized\",\"bottom_profile\":";
    ProfileMetadata(metrics, scene.Contact.Bottom.Height);
    metrics << ",\"top_profile\":";
    ProfileMetadata(metrics, scene.Contact.Top.Height);
    metrics << ",\"contact_events\":{\"resonator\":";
    EventMetadata(metrics, trace.Events[0]);
    metrics << ",\"slider\":";
    EventMetadata(metrics, trace.Events[1]);
    metrics << "}}\n";
    if (!metrics) throw std::runtime_error("Incomplete Dang metrics");
}
void Dang(const std::filesystem::path &output, double duration, double speed, unsigned label, uint64_t seed) {
    const auto setting = std::ranges::find(roughness, label, &Roughness::Label);
    if (setting == roughness.end() || !std::isfinite(duration) || duration < DangTimeStep || duration > 10 || !std::isfinite(speed) || speed <= 0)
        throw std::invalid_argument("Dang requires positive duration <= 10 s, positive speed and Ra label 3, 5, 8, 10, 20 or 30");
    const unsigned steps = unsigned(std::llround(duration / DangTimeStep));
    auto gpu = CreateGpu();
    const auto scene = MakeDang(gpu, *setting, seed);
    const auto device = UploadDang(gpu, scene);
    std::cout << "Dang Ra" << label << ": " << steps << " steps, " << speed << " m/s, 90001 + 4001 nodes, 40 + 4 modes" << std::endl;
    const auto start = std::chrono::steady_clock::now();
    const auto trace = RenderGpu(gpu, device, steps, DangOffset, scene.Separation, speed);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    SaveDang(gpu, output, scene, trace, *setting, speed, seed, seconds, 1);
    std::cout << "Render: " << seconds << " s; " << seconds / (steps * DangTimeStep) << " wall seconds per simulated second\n";
}
double PenaltyEnergy(const Model &m, State &scratch, std::span<const double> q, double offset, double separation) {
    PrepareContact(m, scratch, offset, separation);
    double result{};
    for (unsigned row = 0; row < scratch.Rows; ++row) {
        const double gap = scratch.Gap[row] + std::inner_product(q.begin(), q.end(), scratch.Jacobian.begin() + row * q.size(), 0.);
        result += .5 * m.Penalty * scratch.Weight[row] * std::pow(std::min(0., gap), 2);
    }
    return result;
}
void DangCalibrate(unsigned refinement = 1, double penalty_scale = 1) {
    if (!refinement || refinement > 4 || !std::isfinite(penalty_scale) || penalty_scale < .25 || penalty_scale > 4)
        throw std::invalid_argument("Dang calibration requires refinement 1–4 and penalty scale 0.25–4");
    const double dt = DangTimeStep / refinement;
    auto gpu = CreateGpu();
    const auto scene = MakeDang(gpu, roughness[1], 2013, dt, 2.1e12 * penalty_scale);
    const auto &m = scene.Contact;
    std::array states{MakeState(m), MakeState(m)};
    auto geometry = MakeState(m);
    const unsigned steps = 10000 * refinement;
    constexpr double speed = .1, tolerance = 1e-14;
    constexpr double offset_increment = 1e-9;
    const double initial_contact_energy = .5 * (PenaltyEnergy(m, geometry, states[0].Displacement, DangOffset, scene.Separation) +
                                               PenaltyEnergy(m, geometry, states[0].Previous, DangOffset - speed * dt, scene.Separation));
    std::array<double, 3> error{}, norm{};
    std::array<double, 2> force{};
    std::array<std::vector<double>, 2> forces{std::vector<double>(steps), std::vector<double>(steps)};
    std::array<std::array<double, 2>, 2> work{};
    std::array<double, 2> damping{}, gravity_work{};
    const std::array initial_energy{Energy(m, states[0]), Energy(m, states[1])};
    double residual{}, sliding_work{};
    unsigned iterations{};
    for (unsigned n = 0; n < steps; ++n) {
        const double offset = DangOffset + n * dt * speed;
        sliding_work += speed * dt / (2 * offset_increment) *
                        (PenaltyEnergy(m, geometry, states[0].Displacement, offset + offset_increment, scene.Separation) -
                         PenaltyEnergy(m, geometry, states[0].Displacement, offset - offset_increment, scene.Separation));
        const std::array results{
            Step(m, states[0], ContactMethod::Penalty, DangOffset + n * dt * speed, scene.Separation),
            Step(m, states[1], ContactMethod::Multiplier, DangOffset + (n + 1) * dt * speed, scene.Separation, tolerance)
        };
        if (!results[0].Converged || !results[1].Converged) throw std::runtime_error("Dang contact calibration did not converge");
        residual = std::max(residual, results[1].Residual);
        iterations = std::max(iterations, results[1].Iterations);
        for (unsigned i = 0; i < 2; ++i) {
            const auto &s = states[i];
            forces[i][n] = results[i].NormalForce;
            force[i] += results[i].NormalForce;
            for (unsigned k = 0; k < s.Displacement.size(); ++k) {
                const unsigned body = k >= m.Bottom.Modes.size(), local = body ? k - unsigned(m.Bottom.Modes.size()) : k;
                const auto &surface = body ? m.Top : m.Bottom;
                const auto mode = surface.Modes[local];
                // Next retains q_(n-1) after a successful state rotation.
                const double velocity = (s.Displacement[k] - s.Next[k]) / (2 * dt);
                work[i][body] += (s.Force[k] - surface.Gravity[local]) * velocity * dt;
                gravity_work[i] += surface.Gravity[local] * velocity * dt;
                damping[i] += 2 * mode.Mass * mode.DampingRatio * mode.Omega * velocity * velocity * dt;
            }
        }
        error[0] += std::pow(results[0].NormalForce - results[1].NormalForce, 2);
        norm[0] += std::pow(results[1].NormalForce, 2);
        for (unsigned k = 0; k < states[0].Displacement.size(); ++k) {
            const unsigned body = k < m.Bottom.Modes.size() ? 1 : 2;
            error[body] += std::pow(states[0].Displacement[k] - states[1].Displacement[k], 2);
            norm[body] += std::pow(states[1].Displacement[k], 2);
        }
    }
    std::cout << std::setprecision(9) << "Dang Ra5: 1 ms, 0.1 m/s, seed 2013, dt " << dt << " s, penalty " << m.Penalty << " Pa\nquantity,relative_rms_difference\n";
    constexpr std::array names{"normal_force", "resonator_displacement", "slider_displacement"};
    for (unsigned i = 0; i < names.size(); ++i) {
        const double difference = std::sqrt(error[i] / norm[i]);
        if (!std::isfinite(difference)) throw std::runtime_error("Invalid contact calibration comparison");
        std::cout << names[i] << ',' << difference << '\n';
    }
    std::cout << "Mean force, penalty: " << force[0] / steps << " N; multiplier: " << force[1] / steps << " N\n"
              << "Maximum multiplier gap residual: " << residual << " m; iterations: " << iterations << '\n';
    std::cout << "method,impulse_ns,resonator_contact_work_j,slider_contact_work_j,damping_j,modal_balance_residual_j\n";
    for (unsigned i = 0; i < 2; ++i)
        std::cout << (i ? "multiplier" : "penalty") << ',' << force[i] * dt << ',' << work[i][0] << ',' << work[i][1] << ',' << damping[i] << ','
                  << Energy(m, states[i]) - initial_energy[i] - gravity_work[i] - work[i][0] - work[i][1] + damping[i] << '\n';
    const double final_contact_energy = .5 * (PenaltyEnergy(m, geometry, states[0].Displacement, DangOffset + steps * dt * speed, scene.Separation) +
                                             PenaltyEnergy(m, geometry, states[0].Previous, DangOffset + (steps - 1) * dt * speed, scene.Separation));
    std::cout << "Penalty sliding work: " << sliding_work << " J; contact energy change: " << final_contact_energy - initial_contact_energy
              << " J; total energy balance residual: " << Energy(m, states[0]) - initial_energy[0] - gravity_work[0] + damping[0] + final_contact_energy - initial_contact_energy - sliding_work << " J\n";
    std::cout << "force_band_hz,penalty_rms_n,multiplier_rms_n,difference_rms_n\n";
    std::array<double, 3> low{}, total{};
    for (unsigned n = 0; n < steps; ++n) {
        total[0] += forces[0][n] * forces[0][n] / steps;
        total[1] += forces[1][n] * forces[1][n] / steps;
        total[2] += std::pow(forces[0][n] - forces[1][n], 2) / steps;
    }
    // Rectangular full-record DFT bins are 1 kHz apart; DC is included in the lower band.
    for (unsigned bin = 0; bin <= 20; ++bin) {
        std::array<double, 2> real{}, imaginary{};
        for (unsigned n = 0; n < steps; ++n) {
            const double angle = 2 * std::numbers::pi * bin * n / steps;
            for (unsigned i = 0; i < 2; ++i) {
                real[i] += forces[i][n] * std::cos(angle) / steps;
                imaginary[i] += forces[i][n] * std::sin(angle) / steps;
            }
        }
        const double weight = bin ? 2 : 1;
        for (unsigned i = 0; i < 2; ++i) low[i] += weight * (real[i] * real[i] + imaginary[i] * imaginary[i]);
        low[2] += weight * (std::pow(real[0] - real[1], 2) + std::pow(imaginary[0] - imaginary[1], 2));
    }
    for (unsigned band = 0; band < 2; ++band) {
        std::cout << (band ? "above_20000" : "0_to_20000");
        for (unsigned i = 0; i < 3; ++i) std::cout << ',' << std::sqrt(std::max(0., band ? total[i] - low[i] : low[i]));
        std::cout << '\n';
    }
}
void DangSweep(const std::filesystem::path &output, double duration, unsigned seeds) {
    if (!std::isfinite(duration) || duration < .001 || duration > .45 || !seeds || seeds > 100)
        throw std::invalid_argument("Dang sweep requires duration 0.001–0.45 s and 1–100 surface seeds");
    const unsigned steps = unsigned(std::llround(duration / DangTimeStep));
    constexpr std::array speeds{.02, .04, .06, .1, .2, .3, .7};
    constexpr unsigned batch_size = 21, cases_per_seed = unsigned(roughness.size() * speeds.size()), canonical_case = 2 * unsigned(speeds.size()) - 1;
    std::filesystem::create_directories(output);
    std::filesystem::remove(output / "Sweep.json");
    std::ofstream metrics(output / "Sweep.csv");
    if (!metrics) throw std::runtime_error("Cannot write Dang sweep metrics");
    metrics << std::setprecision(17) << "seed,ra_label_um,target_rq_m,correlation_m,speed_m_s,duration_s,window_start_s,space_time_velocity_rms_m_s,mean_normal_force_n,whole_record_resonator_work_j,whole_record_slider_work_j,batch_wall_s,batch_cases\n";
    double render_seconds{};
    for (unsigned first = 0; first < seeds * cases_per_seed; first += batch_size) {
        const unsigned count = std::min(batch_size, seeds * cases_per_seed - first);
        auto gpu = CreateGpu();
        std::vector<DangScene> scenes;
        std::vector<GpuContact> devices;
        std::vector<GpuContactRun> runs;
        const unsigned trajectories = count + !first;
        scenes.reserve(trajectories);
        devices.reserve(trajectories);
        for (unsigned local = 0; local < trajectories; ++local) {
            const unsigned index = local < count ? first + local : canonical_case;
            const auto &setting = roughness[(index % cases_per_seed) / speeds.size()];
            scenes.push_back(MakeDang(gpu, setting, 2013 + index / cases_per_seed));
            const auto &scene = scenes.back();
            devices.push_back(UploadDang(gpu, scene));
            runs.push_back({&devices.back(), steps, DangOffset, scene.Separation, speeds[index % speeds.size()]});
        }
        std::cout << "Dang sweep: cases " << first + 1 << "–" << first + count << " of " << seeds * cases_per_seed << ", " << steps << " steps each" << std::endl;
        const auto start = std::chrono::steady_clock::now();
        const auto traces = RenderGpu(gpu, runs);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        render_seconds += seconds;
        if (!first && traces[canonical_case] != traces.back()) throw std::runtime_error("Independent full-record Dang repeat differs");
        for (unsigned local = 0; local < count; ++local) {
            const unsigned index = first + local, seed = 2013 + index / cases_per_seed;
            const auto &setting = roughness[(index % cases_per_seed) / speeds.size()];
            const double speed = speeds[index % speeds.size()];
            const auto &trace = traces[local];
            for (double begin : {0., .1, .2}) {
                const unsigned frame = unsigned(std::llround(begin / DangTimeStep));
                if (frame >= steps) continue;
                const double square = std::accumulate(trace.MeanSquareVelocity.begin() + frame, trace.MeanSquareVelocity.end(), 0.) / (steps - frame);
                const double force = std::accumulate(trace.Force.begin() + frame, trace.Force.end(), 0.) / (steps - frame);
                metrics << seed << ',' << setting.Label << ',' << setting.Rms << ',' << setting.Correlation << ',' << speed << ',' << steps * DangTimeStep << ',' << begin
                        << ',' << std::sqrt(square) << ',' << force << ',' << trace.Events[0].Work << ',' << trace.Events[1].Work << ',' << seconds << ',' << runs.size() << '\n';
            }
            if (seed == 2013 && setting.Label == 5 && speed == .7) SaveDang(gpu, output, scenes[local], trace, setting, speed, seed, seconds, unsigned(runs.size()));
        }
        metrics.flush();
        if (!metrics) throw std::runtime_error("Incomplete Dang sweep metrics");
        std::cout << "Batch render: " << seconds << " s; " << seconds / (runs.size() * steps * DangTimeStep) << " wall seconds per simulated second" << std::endl;
    }
    std::ofstream summary(output / "Sweep.json");
    summary << std::setprecision(17) << "{\"profile_generation\":\"exact_period\",\"cases\":" << seeds * cases_per_seed << ",\"seeds\":" << seeds << ",\"duration_s\":" << steps * DangTimeStep
            << ",\"time_step_s\":" << DangTimeStep << ",\"spacing_m\":" << DangSpacing << ",\"render_wall_s\":" << render_seconds
            << ",\"full_record_batch_repeatability\":true,\"repeated_case\":{\"seed\":2013,\"ra_label_um\":5,\"speed_m_s\":0.7}}\n";
    if (!summary) throw std::runtime_error("Incomplete Dang sweep summary");
}
const PlateProperties AssemienPlate{.6, .4, 7800 * .004, 210e9 * std::pow(.004, 3) / (12 * (1 - .3 * .3)), .00005};
constexpr RigidPlateProperties AssemienSlider{.02, .02, .0312, 1.3e-6, 1.3e-6};
struct PlateScene {
    PlateContactModel Model;
    double Separation;
};
PlateScene MakeAssemien(Gpu &gpu, uint64_t seed, unsigned track_nodes = 23001) {
    constexpr double dt = 1e-6, spacing = 20e-6, rms = 17e-6, correlation = 80e-6, x0 = .07, y0 = .185;
    const uint64_t top_seed = seed ^ 0x9e3779b97f4a7c15ULL;
    const auto bottom_height = GaussianSurface(gpu, track_nodes, 1501, spacing, spacing, rms, correlation, correlation, seed);
    const auto top_height = GaussianSurface(gpu, 1001, 1001, spacing, spacing, rms, correlation, correlation, top_seed);
    auto model = MakePlateContact(MakePlateSurface(AssemienPlate, 200, {track_nodes, 1501, spacing, spacing, x0, y0}, bottom_height, dt), MakeRigidPlateSurface(AssemienSlider, {1001, 1001, spacing, spacing, 0, 0}, top_height, dt, -9.81), dt, {1e16, 3e17});
    double separation = -std::numeric_limits<double>::infinity();
    for (unsigned y = 0; y < 1001; ++y)
        for (unsigned x = 0; x < 1001; ++x) separation = std::max(separation, bottom_height[y * track_nodes + x] + top_height[y * 1001 + x]);
    return {std::move(model), separation};
}
void PlateBenchmark(unsigned steps) {
    if (!steps || steps > 100000) throw std::invalid_argument("Plate benchmark requires 1–100000 steps");
    auto gpu = CreateGpu();
    const auto scene = MakeAssemien(gpu, 2023, 1601);
    const auto &model = scene.Model;
    std::array devices{MakePlateContactGpu(gpu, model, 4096, false), MakePlateContactGpu(gpu, model, 16384, true), MakePlateContactGpu(gpu, model, 16384, true)};
    std::array states{MakePlateState(model), MakePlateState(model), MakePlateState(model)};
    std::array<double, 3> elapsed{};
    double state_error{}, state_norm{}, force_error{}, force_norm{}, residual{};
    unsigned contacts{}, maximum_contacts{}, refreshes{};
    for (unsigned step = 0; step < steps; ++step) {
        std::array<PlateContactResult, 3> result{};
        for (unsigned order = 0; order < 3; ++order) {
            const unsigned method = (step + order) % 3;
            auto &device = devices[method];
            if (method == 1) device.CandidateValid = false;
            const auto previous_bounds = device.CandidateBounds;
            const auto start = std::chrono::steady_clock::now();
            PreparePlateContactGpu(gpu, device, model, states[method], .07 + step * model.TimeStep * .1, .185, scene.Separation);
            result[method] = AdvancePlateContact(model, states[method]);
            elapsed[method] += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (!result[method].Solve.Converged) throw std::runtime_error("Plate benchmark solve did not converge");
            if (method && device.DenseCandidates) throw std::runtime_error("Plate benchmark exceeds its FP64 candidate capacity");
            if (method == 2) refreshes += previous_bounds != device.CandidateBounds;
            residual = std::max(residual, result[method].Solve.Residual);
        }
        const auto &reference = states[1], &actual = states[2], &full = states[0];
        if (reference.Jacobian != actual.Jacobian || reference.Penetration != actual.Penetration || reference.Area != actual.Area ||
            reference.Displacement != actual.Displacement || reference.Previous != actual.Previous || reference.Velocity != actual.Velocity ||
            reference.Force != actual.Force || result[1].Solve.Residual != result[2].Solve.Residual)
            throw std::runtime_error("Candidate reuse changed the FP64 contact trajectory");
        for (unsigned k = 0; k < full.Displacement.size(); ++k) {
            state_error += std::pow(full.Displacement[k] - actual.Displacement[k], 2);
            state_norm += full.Displacement[k] * full.Displacement[k];
        }
        force_error += std::pow(result[0].NormalForce - result[2].NormalForce, 2);
        force_norm += result[0].NormalForce * result[0].NormalForce;
        contacts += result[2].Contacts;
        maximum_contacts = std::max(maximum_contacts, result[2].Contacts);
    }
    std::cout << std::setprecision(9) << "Plate benchmark: " << steps << " steps, 1601 x 1501 track, 1001 x 1001 slider, 203 modes\n"
              << "Full GPU " << elapsed[0] << " s; fresh candidates/FP64 " << elapsed[1] << " s; cached candidates/FP64 " << elapsed[2]
              << " s; speedup vs GPU " << elapsed[0] / elapsed[2] << "; speedup vs fresh candidates " << elapsed[1] / elapsed[2]
              << "; refreshes " << refreshes << "; contact-steps " << contacts << "; maximum contacts " << maximum_contacts
              << "; FP64 trajectories exact; relative displacement vs GPU " << std::sqrt(state_error / std::max(state_norm, 1e-40))
              << "; relative force vs GPU " << std::sqrt(force_error / std::max(force_norm, 1e-40)) << "; maximum residual " << residual << std::endl;
}
void Assemien(const std::filesystem::path &output, double duration, double speed, uint64_t seed) {
    constexpr double dt = 1e-6, rms = 17e-6, correlation = 80e-6, x0 = .07, y0 = .185;
    if (!std::isfinite(duration) || duration < dt || duration > 10 || !std::isfinite(speed) || speed <= 0)
        throw std::invalid_argument("Assemien requires positive duration <= 10 s and positive speed");
    const unsigned steps = unsigned(std::llround(duration / dt));
    const uint64_t top_seed = seed ^ 0x9e3779b97f4a7c15ULL;
    auto gpu = CreateGpu();
    const auto scene = MakeAssemien(gpu, seed);
    const auto &model = scene.Model;
    const double separation = scene.Separation;
    auto state = MakePlateState(model);
    auto device = MakePlateContactGpu(gpu, model, 16384);
    std::vector<double> receiver(200);
    for (unsigned k = 0; k < receiver.size(); ++k) receiver[k] = model.Bottom.XShapes[k * 23001 + 19750] * model.Bottom.YShapes[k * 1501 + 500];
    std::vector<float> velocity(steps);
    double force{}, square{}, dissipation{}, maximum_residual{};
    unsigned maximum_contacts{};
    std::cout << "Assemien: " << steps << " steps, " << speed << " m/s, 23001 x 1501 track, 1001 x 1001 slider, 200 + 3 modes" << std::endl;
    const auto start = std::chrono::steady_clock::now();
    for (unsigned n = 0; n < steps; ++n) {
        PreparePlateContactGpu(gpu, device, model, state, x0 + n * dt * speed, y0, separation);
        const auto result = AdvancePlateContact(model, state);
        if (!result.Solve.Converged) throw std::runtime_error("Assemien contact solve did not converge");
        double value{};
        for (unsigned k = 0; k < receiver.size(); ++k) {
            value += receiver[k] * state.Velocity[k];
            square += state.Velocity[k] * state.Velocity[k] / (AssemienPlate.Length * AssemienPlate.Width);
        }
        velocity[n] = float(value);
        force += result.NormalForce;
        dissipation += result.Dissipation * dt;
        maximum_residual = std::max(maximum_residual, result.Solve.Residual);
        maximum_contacts = std::max(maximum_contacts, result.Contacts);
        if (steps >= 100000 && (n + 1) % (steps / 20) == 0) std::cout << "Assemien " << (n + 1) * 100 / steps << "%" << std::endl;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const double gain = SaveVelocity(gpu, output, velocity, std::array{20u});
    std::ofstream metrics(output / "Metrics.json");
    if (!metrics) throw std::runtime_error("Cannot write Assemien metrics");
    metrics << std::setprecision(17) << "{\"paper\":\"Assemien 2023 thesis Appendix 4\",\"duration_s\":" << steps * dt << ",\"time_step_s\":" << dt
            << ",\"spacing_m\":" << model.Bottom.Grid.StepX << ",\"speed_m_s\":" << speed << ",\"offset_m\":[" << x0 << ',' << y0 << "],\"initial_separation_m\":" << separation
            << ",\"bottom_seed\":\"" << seed << "\",\"top_seed\":\"" << top_seed << "\",\"target_rq_m\":" << rms << ",\"correlation_m\":" << correlation
            << ",\"slider_mass_kg\":" << AssemienSlider.Mass << ",\"slider_inertia_kg_m2\":[" << AssemienSlider.InertiaX << ',' << AssemienSlider.InertiaY << "]"
            << ",\"slider_weight_n\":" << AssemienSlider.Mass * 9.81 << ",\"bottom_modes\":200,\"top_rigid_modes\":3,\"two_pass\":false,\"candidate_reuse\":true"
            << ",\"contact_law\":{\"K\":1e16,\"chi\":3e17,\"n\":1.5,\"m\":1.5},\"receiver_m\":[0.465,0.195],\"sample_rate_hz\":44100"
            << ",\"full_overlap_end_s\":" << (.46 - .02) / speed << ",\"all_overlap_end_s\":" << .46 / speed
            << ",\"render_wall_s\":" << seconds << ",\"wall_per_simulated_second\":" << seconds / (steps * dt)
            << ",\"mean_normal_force_n\":" << force / steps << ",\"space_time_velocity_rms_m_s\":" << std::sqrt(square / steps)
            << ",\"contact_dissipation_j\":" << dissipation << ",\"maximum_contacts\":" << maximum_contacts << ",\"maximum_solver_residual\":" << maximum_residual
            << ",\"audio_gain\":" << gain << ",\"audio_quantity\":\"receiver surface velocity; peak normalized\",\"bottom_profile\":";
    ProfileMetadata(metrics, model.Bottom.Height);
    metrics << ",\"top_profile\":";
    ProfileMetadata(metrics, model.Top.Height);
    metrics << "}\n";
    if (!metrics) throw std::runtime_error("Incomplete Assemien metrics");
    std::cout << "Render: " << seconds << " s; " << seconds / (steps * dt) << " wall seconds per simulated second\n";
}
void Gregoire(const std::filesystem::path &output) {
    std::filesystem::create_directories(output);
    const auto p = InstrumentedSlider();
    const SliderTrack<double> flat{};
    const auto state = EquilibrateSlider(p, flat);
    const auto baseline = EvaluateSlider(p, state, flat);
    std::ofstream metadata(output / "static.json"), trace(output / "sensor.csv");
    if (!metadata || !trace) throw std::runtime_error("Cannot write slider reproduction");
    metadata << std::setprecision(17) << "{\"mass_kg\":" << p.Mass << ",\"inertia_kg_m2\":" << p.InertiaX << ",\"hertz_stiffness\":" << p.Stiffness << ",\"cases\":[";
    for (int removed : {-1, 6, 8}) {
        SliderTrack<double> track{};
        if (removed >= 0) track.Height[removed] = -.001;
        const auto equilibrium = EquilibrateSlider(p, track);
        const auto forces = EvaluateSlider(p, equilibrium, track);
        metadata << (removed == -1 ? "" : ",") << "{\"removed_asperity\":" << removed + 1 << ",\"forces_n\":[";
        for (unsigned j = 0; j < 9; ++j) metadata << (j ? "," : "") << forces.Force[j];
        metadata << "],\"position_m_rad\":[";
        for (unsigned k = 0; k < 3; ++k) metadata << (k ? "," : "") << equilibrium.Position[k];
        metadata << "]}";
    }
    metadata << "]}\n";
    trace << std::setprecision(17) << "time,asperity7_loss,asperity9_loss,asperity9_return\n";
    for (int n = -1000; n <= 10000; ++n) {
        const double t = n / 5000.;
        trace << t << ',' << SensorStep(t, -baseline.Force[6]) << ',' << SensorStep(t, -baseline.Force[8]) << ',' << SensorStep(t, baseline.Force[8]) << '\n';
    }
    if (!metadata || !trace) throw std::runtime_error("Incomplete slider reproduction output");
    std::cout << "Grégoire static contact and sensor traces written to " << output << '\n';
}
}
int main(int argc, char **argv) {
    try {
        if (argc == 3 && std::string_view(argv[1]) == "gregoire") Gregoire(argv[2]);
        else if ((argc == 4 || argc == 5) && std::string_view(argv[1]) == "dang-benchmark") DangBenchmark(unsigned(std::stoul(argv[2])), unsigned(std::stoul(argv[3])), argc == 5 ? unsigned(std::stoul(argv[4])) : 32);
        else if (argc == 5 && std::string_view(argv[1]) == "dang-sweep") DangSweep(argv[2], std::stod(argv[3]), unsigned(std::stoul(argv[4])));
        else if (argc == 7 && std::string_view(argv[1]) == "dang") Dang(argv[2], std::stod(argv[3]), std::stod(argv[4]), unsigned(std::stoul(argv[5])), std::stoull(argv[6]));
        else if ((argc == 2 || argc == 4) && std::string_view(argv[1]) == "dang-calibrate") DangCalibrate(argc == 4 ? std::stoul(argv[2]) : 1, argc == 4 ? std::stod(argv[3]) : 1);
        else if (argc == 3 && std::string_view(argv[1]) == "assemien-benchmark") PlateBenchmark(std::stoul(argv[2]));
        else if (argc == 6 && std::string_view(argv[1]) == "assemien") Assemien(argv[2], std::stod(argv[3]), std::stod(argv[4]), std::stoull(argv[5]));
        else throw std::invalid_argument("Usage: roughReproduce gregoire OUTPUT | dang OUTPUT SECONDS SPEED_M_S RA_LABEL SEED | dang-sweep OUTPUT SECONDS SEEDS | dang-benchmark STEPS BATCH [REFRESH] | dang-calibrate [REFINEMENT PENALTY_SCALE] | assemien-benchmark STEPS | assemien OUTPUT SECONDS SPEED_M_S SEED");
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
