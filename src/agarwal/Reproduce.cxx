#include "Agarwal.h"
#include "core/AudioFile.h"
#include "core/BinaryFile.h"
#include "core/GpuConvolution.h"
#include "core/GpuDecimate.h"
#include "core/GpuFiniteModes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>

using namespace surface_audio;
using namespace surface_audio::agarwal;
namespace {

struct Parameters {
    uint32_t SampleRate{}, Frames{}, Levels{}, Taps{};
    double Spacing{}, HalfWidth{}, Mass{}, Beta1{}, Beta2{}, Radius{}, Eccentricity{}, Stiffness{}, Dissipation{}, Gain{};
    double AlphaScale{1};
};
Parameters ReadParameters(const std::filesystem::path &directory) {
    Parameters p;
    std::ifstream file(directory / "parameters.txt");
    file >> p.SampleRate >> p.Frames >> p.Levels >> p.Taps >> p.Spacing >> p.HalfWidth >> p.Mass >> p.Beta1 >> p.Beta2 >> p.Radius >> p.Eccentricity >> p.Stiffness >> p.Dissipation >> p.Gain;
    if (!file || p.Levels < 2 || !p.Frames || !p.SampleRate || !p.Taps || !(p.Spacing > 0)) throw std::runtime_error("Invalid reproduction parameters");
    file >> std::ws;
    if (!file.eof() && !(file >> p.AlphaScale)) throw std::runtime_error("Invalid reproduction alpha scale");
    if (!(p.AlphaScale > 0) || !std::isfinite(p.AlphaScale)) throw std::runtime_error("Invalid reproduction alpha scale");
    return p;
}

double InterpolateMotionField(const std::vector<double> &motion, size_t sample, uint32_t factor, size_t field) {
    const size_t left = sample / factor, right = std::min(left + 1, motion.size() / 4 - 1);
    return std::lerp(motion[4 * left + field], motion[4 * right + field], double(sample % factor) / factor);
}

struct RetainPlan {
    GpuBuffer Parameters;
    GpuKernel Kernel;
    GpuGrid Threads;
};
RetainPlan CreateRetainPlan(Gpu &gpu, uint32_t frames, uint32_t dense_frames, uint32_t factor, uint32_t fields) {
    struct Block {
        uint32_t Frames, DenseFrames, Factor, Fields;
    };
    return {Upload(gpu, Block{frames, dense_frames, factor, fields}), CreateKernel(gpu, "AgarwalRetainFields"), {frames, fields}};
}
void DispatchRetainedFields(Gpu &gpu, const RetainPlan &plan, GpuBuffer input, GpuBuffer output) {
    const std::array bindings{GpuBinding{plan.Parameters, 0}, GpuBinding{input, 1}, GpuBinding{output, 2}};
    DispatchGpu(gpu, plan.Kernel, bindings, plan.Threads);
}

void DecimateReference(std::span<const double> input, uint32_t dense_frames, uint32_t channels, uint32_t factor, std::span<float> output) {
    const auto coefficients = KaiserDecimateCoefficients(factor);
    const int64_t radius = int64_t(coefficients.size() / 2);
    const uint32_t frames = (dense_frames - 1) / factor + 1;
    if (input.size() != size_t(dense_frames) * channels || output.size() != size_t(frames) * channels) throw std::runtime_error("Reference decimation extent mismatch");
    // Independent serial FP64 FIR, including physical force values before FP32 output conversion.
    for (uint32_t channel = 0; channel < channels; ++channel)
        for (uint32_t n = 0; n < frames; ++n) {
            double sum = 0;
            for (int64_t d = -radius; d <= radius; ++d) {
                const size_t sample = size_t(std::clamp(int64_t(n) * factor + d, int64_t(0), int64_t(dense_frames) - 1));
                sum += coefficients[size_t(d + radius)] * input[size_t(channel) * dense_frames + sample];
            }
            output[size_t(channel) * frames + n] = float(sum);
        }
}

void PrepareOversampledVertical(const std::filesystem::path &directory, const Parameters &p, const std::vector<double> &raw, const std::vector<double> &motion, bool gpu_enabled, uint32_t mode, uint32_t factor, double sigma_ratio, bool bandlimit) {
    const size_t dense_count = size_t(p.Frames - 1) * factor + 1;
    if (dense_count > size_t(std::numeric_limits<int32_t>::max()) || (bandlimit && dense_count > UINT32_MAX / 4) || !(p.HalfWidth >= 0) || !std::isfinite(p.HalfWidth) || p.HalfWidth * (double(p.SampleRate) / 44100) * .05 / .03 * factor > std::numeric_limits<int32_t>::max() / 2.) throw std::runtime_error("Invalid oversampled temporal extent or width");
    const uint32_t force_frames = bandlimit ? uint32_t(dense_count) : p.Frames;
    std::vector<double> dense_motion(bandlimit ? 4 * dense_count : 0);
    for (size_t j = 0; bandlimit && j < dense_count; ++j)
        for (size_t field = 0; field < 4; ++field) dense_motion[4 * j + field] = InterpolateMotionField(motion, j, factor, field);
    const auto &force_motion = bandlimit ? dense_motion : motion;
    struct Location {
        uint32_t Cell;
        float Fraction;
    };
    std::vector<Location> locations(gpu_enabled ? dense_count : 0);
    std::vector<float> normals(gpu_enabled ? dense_count : 0);
    std::vector<double> sampled(gpu_enabled ? 0 : dense_count);
    for (size_t j = 0; j < dense_count; ++j) {
        // Keep the whole spatial coordinate in FP64 until its integer cell is removed.
        const double grid = InterpolateMotionField(motion, j, factor, 0) / p.Spacing;
        const double normal = InterpolateMotionField(motion, j, factor, 2);
        if (grid < 0 || grid > raw.size() - 1) throw std::runtime_error("Reproduction path exceeds measured profile");
        const size_t cell = std::min(size_t(grid), raw.size() - 2);
        if (gpu_enabled) {
            locations[j] = {uint32_t(cell), float(grid - cell)};
            normals[j] = float(normal);
        } else {
            const double alpha = p.AlphaScale * Alpha(normal, {.NormalMin = 1, .NormalMax = 6});
            sampled[j] = mode == 2 ? ConstrainCurvature(std::lerp(raw[cell], raw[cell + 1], grid - cell), alpha) : std::lerp(ConstrainCurvature(raw[cell], alpha), ConstrainCurvature(raw[cell + 1], alpha), grid - cell);
        }
    }
    std::vector<float> scrape(p.Frames), zeros(p.Frames), morph(p.Frames), trajectory(3 * size_t(p.Frames));
    if (gpu_enabled) {
        struct DenseBlock {
            uint32_t Frames, VerticalMode;
            float AlphaScale;
        };
        struct FilterBlock {
            uint32_t Frames, DenseFrames, Factor, SampleRate;
            float HalfWidth, Mass, SigmaRatio;
        };
        auto gpu = CreateGpu();
        const std::vector<float> raw_float(raw.begin(), raw.end()), motion_float(force_motion.begin(), force_motion.end());
        const auto dense_parameters = Upload(gpu, DenseBlock{uint32_t(dense_count), mode, float(p.AlphaScale)}), filter_parameters = Upload(gpu, FilterBlock{force_frames, uint32_t(dense_count), bandlimit ? 1u : factor, p.SampleRate * (bandlimit ? factor : 1), float(p.HalfWidth), float(p.Mass), float(sigma_ratio)});
        const auto surface = Upload<float>(gpu, raw_float), coordinates = Upload<Location>(gpu, locations), normal = Upload<float>(gpu, normals), path = Upload<float>(gpu, motion_float);
        const auto dense = CreateBuffer(gpu, dense_count * sizeof(float)), force = CreateBuffer(gpu, size_t(force_frames) * sizeof(float)), filtered = CreateBuffer(gpu, size_t(force_frames) * sizeof(float));
        const auto sample_kernel = CreateKernel(gpu, "AgarwalDenseVerticalCurvature"), filter_kernel = CreateKernel(gpu, "AgarwalDownsampleVerticalForceV2");
        const std::array sample_bindings{GpuBinding{dense_parameters, 0}, GpuBinding{surface, 1}, GpuBinding{coordinates, 2}, GpuBinding{normal, 3}, GpuBinding{dense, 4}};
        const std::array filter_bindings{GpuBinding{filter_parameters, 0}, GpuBinding{dense, 1}, GpuBinding{path, 2}, GpuBinding{force, 3}, GpuBinding{filtered, 4}};
        const auto final_force = bandlimit ? CreateBuffer(gpu, size_t(p.Frames) * sizeof(float)) : force;
        const auto retained = bandlimit ? CreateBuffer(gpu, size_t(p.Frames) * sizeof(float)) : filtered;
        const auto decimation = bandlimit ? CreateDecimatePlan(gpu, force_frames, 1, factor) : GpuDecimatePlan{};
        const auto retention = bandlimit ? CreateRetainPlan(gpu, p.Frames, force_frames, factor, 1) : RetainPlan{};
        BeginGpu(gpu);
        DispatchGpu(gpu, sample_kernel, sample_bindings, {uint32_t(dense_count)});
        DispatchGpu(gpu, filter_kernel, filter_bindings, {force_frames});
        if (bandlimit) {
            DispatchDecimateGpu(gpu, decimation, force, final_force);
            DispatchRetainedFields(gpu, retention, filtered, retained);
        }
        SubmitGpu(gpu);
        WaitGpu(gpu);
        const auto values = BufferSpan<float>(final_force), curvature = BufferSpan<float>(retained);
        std::copy(values.begin(), values.end(), scrape.begin());
        std::copy(curvature.begin(), curvature.end(), trajectory.begin() + 2 * p.Frames);
    } else {
        std::vector<double> dense_force(bandlimit ? dense_count : 0);
        for (uint32_t n = 0; n < force_frames; ++n) {
            const double width = p.HalfWidth * (double(p.SampleRate) / 44100) * Alpha(force_motion[4 * n + 2], {.NormalMin = 1, .NormalMax = 6}) / .03 * factor;
            const double sigma = std::max(sigma_ratio * width, 1e-12);
            const int64_t center = int64_t(n) * (bandlimit ? 1 : factor), radius = int64_t(std::ceil(width));
            double sum = 0, normalization = 0;
            for (int64_t j = std::max(int64_t(0), center - radius); j <= std::min(int64_t(dense_count - 1), center + radius); ++j) {
                const double distance = double(j - center), weight = std::exp(-.5 * distance * distance / (sigma * sigma));
                sum += weight * sampled[j];
                normalization += weight;
            }
            const double curvature = sum / normalization, velocity = force_motion[4 * n + 1];
            if (bandlimit) {
                dense_force[n] = p.Mass * curvature * velocity * velocity;
                if (n % factor == 0) trajectory[2 * p.Frames + n / factor] = float(curvature);
            } else {
                trajectory[2 * p.Frames + n] = float(curvature);
                scrape[n] = float(p.Mass * curvature * velocity * velocity);
            }
        }
        if (bandlimit) DecimateReference(dense_force, force_frames, 1, factor, scrape);
    }
    for (uint32_t n = 0; n < p.Frames; ++n) {
        if (!std::isfinite(scrape[n]) || !std::isfinite(trajectory[2 * p.Frames + n])) throw std::runtime_error("Nonfinite oversampled vertical force or curvature");
        morph[n] = float(motion[4 * n + 3]);
    }
    WriteBinary<float>(directory / "trajectory.f32", trajectory);
    WriteBinary<float>(directory / "scrape.f32", scrape);
    WriteBinary<float>(directory / "elastic.f32", zeros);
    WriteBinary<float>(directory / "damping.f32", zeros);
    WriteBinary<float>(directory / "morph.f32", morph);
    std::cout << "Prepared measured profile: samples=" << raw.size() << " vertical_mode=" << mode << " oversampling=" << factor << " sigma_ratio=" << sigma_ratio << " bandlimit=" << bandlimit << " dense_frames=" << dense_count << '\n';
}

void Prepare(const std::filesystem::path &directory, const Parameters &p, bool gpu_enabled, bool temporal = false, uint32_t vertical_mode = 0, uint32_t oversampling = 1, double sigma_ratio = .4, bool bandlimit = false) {
    const bool vertical_only = vertical_mode != 0;
    if (bandlimit && (!temporal || oversampling < 2 || p.SampleRate > UINT32_MAX / oversampling)) throw std::invalid_argument("Bandlimited preparation requires temporal sampling with factor at least 2 and a representable dense sample rate");
    const auto heights = ReadBinary<double>(directory / "profile.f64"), motion = ReadBinary<double>(directory / "motion.f64");
    if (heights.size() < 3 || motion.size() != 4 * size_t(p.Frames)) throw std::runtime_error("Invalid reproduction input extents");
    const size_t count = heights.size();
    if (count > std::numeric_limits<uint32_t>::max() / (3 * size_t(p.Levels)) || p.Frames > std::numeric_limits<uint32_t>::max() / 4) throw std::runtime_error("Reproduction GPU extent exceeds uint32 indexing");
    for (double height : heights)
        if (!std::isfinite(height)) throw std::runtime_error("Nonfinite measured profile");
    for (size_t frame = 0; frame < p.Frames; ++frame) {
        for (size_t field = 0; field < 4; ++field)
            if (!std::isfinite(motion[4 * frame + field])) throw std::runtime_error("Nonfinite reproduction motion");
        if (motion[4 * frame + 2] < 1 || motion[4 * frame + 2] > 6 || motion[4 * frame + 3] < 0 || motion[4 * frame + 3] > 1) throw std::runtime_error("Reproduction normal coordinate or morph outside range");
    }
    std::vector<double> raw(count), clipped(vertical_only ? 0 : count), curvature(vertical_only ? 0 : count), weights;
    std::vector<float> atlas(vertical_only ? 0 : 3 * count * p.Levels);
    for (size_t x = 0; x < count; ++x) {
        const size_t center = std::clamp(x, size_t(1), count - 2);
        raw[x] = (heights[center + 1] - 2 * heights[center] + heights[center - 1]) / (p.Spacing * p.Spacing);
    }
    if (vertical_only && (oversampling > 1 || sigma_ratio != .4)) {
        PrepareOversampledVertical(directory, p, raw, motion, gpu_enabled, vertical_mode, oversampling, sigma_ratio, bandlimit);
        return;
    }
    auto gpu = CreateGpu();
    GpuBuffer gpu_curvature;
    const bool relative_correction = temporal && .05 * p.AlphaScale * std::abs(*std::max_element(raw.begin(), raw.end(), [](double a, double b) { return std::abs(a) < std::abs(b); })) <= .1;
    if (gpu_enabled && !vertical_only) {
        struct Block {
            uint32_t Count, Levels;
            float HalfWidth, AlphaScale;
            uint32_t RelativeCorrection;
        };
        const auto parameters = Upload(gpu, Block{uint32_t(count), p.Levels, float(temporal ? 0 : p.HalfWidth), float(p.AlphaScale), uint32_t(relative_correction)});
        const std::vector<float> raw_float(raw.begin(), raw.end());
        const auto input = Upload<float>(gpu, raw_float);
        gpu_curvature = CreateBuffer(gpu, count * p.Levels * sizeof(float));
        const auto kernel = CreateKernel(gpu, "AgarwalAtlasCurvature");
        const std::array bindings{GpuBinding{parameters, 0}, GpuBinding{input, 1}, GpuBinding{gpu_curvature, 2}};
        BeginGpu(gpu);
        DispatchGpu(gpu, kernel, bindings, {uint32_t(count * p.Levels)});
        SubmitGpu(gpu);
        WaitGpu(gpu);
    }
    std::vector<double> integrated_height(vertical_only ? 0 : count), integrated_slope(vertical_only ? 0 : count);
    for (uint32_t level = 0; !vertical_only && level < p.Levels; ++level) {
        const double alpha = std::lerp(0.01, 0.05, double(level) / (p.Levels - 1)), half_width = (temporal ? 0 : p.HalfWidth) * alpha / 0.03, sigma = std::max(half_width * .4, 1e-12);
        const int radius = int(std::ceil(half_width));
        weights.resize(2 * radius + 1);
        for (int d = -radius; d <= radius; ++d) weights[d + radius] = std::exp(-.5 * d * d / (sigma * sigma));
        if (gpu_enabled) {
            const auto values = BufferSpan<float>(gpu_curvature).subspan(size_t(level) * count, count);
            std::copy(values.begin(), values.end(), curvature.begin());
            if (relative_correction)
                for (size_t x = 0; x < count; ++x) curvature[x] = raw[x] * (1 + curvature[x]);
        } else {
            for (size_t x = 0; x < count; ++x) clipped[x] = ConstrainCurvature(raw[x], alpha * p.AlphaScale);
            for (size_t x = 0; x < count; ++x) {
                double sum = 0, normalization = 0;
                const auto begin = std::max(int64_t(0), int64_t(x) - radius), end = std::min(int64_t(count - 1), int64_t(x) + radius);
                for (auto j = begin; j <= end; ++j) {
                    const double weight = weights[j - int64_t(x) + radius];
                    sum += weight * clipped[j];
                    normalization += weight;
                }
                curvature[x] = sum / normalization;
            }
        }
        const size_t base = size_t(level) * 3 * count;
        for (size_t x = 0; x < count; ++x) atlas[base + 2 * count + x] = float(curvature[x]);
        double slope = 0, height = heights[0];
        integrated_height[0] = height;
        integrated_slope[0] = 0;
        for (size_t x = 1; x < count; ++x) {
            height += slope * p.Spacing + (2 * curvature[x - 1] + curvature[x]) * p.Spacing * p.Spacing / 6;
            slope += .5 * (curvature[x - 1] + curvature[x]) * p.Spacing;
            integrated_height[x] = height;
            integrated_slope[x] = slope;
        }
        // Endpoint heights fix the integration constant without changing curvature.
        const double anchor = (heights.back() - height) / ((count - 1) * p.Spacing);
        for (size_t x = 0; x < count; ++x) {
            atlas[base + x] = float(integrated_height[x] + anchor * x * p.Spacing);
            atlas[base + count + x] = float(integrated_slope[x] + anchor);
        }
    }
    const bool resolved_full = temporal && !vertical_only && (oversampling > 1 || sigma_ratio != .4);
    const uint32_t sample_factor = resolved_full ? oversampling : 1;
    const size_t dense_count = size_t(p.Frames - 1) * sample_factor + 1;
    if (resolved_full && (dense_count > std::numeric_limits<uint32_t>::max() / 4 || !(p.HalfWidth >= 0) || !std::isfinite(p.HalfWidth) || p.HalfWidth * (double(p.SampleRate) / 44100) * .05 / .03 * sample_factor > std::numeric_limits<int32_t>::max() / 2.)) throw std::runtime_error("Invalid full-model temporal extent or width");
    const uint32_t sample_frames = uint32_t(dense_count);
    std::vector<double> dense_motion(resolved_full ? 4 * dense_count : 0);
    for (size_t sample = 0; resolved_full && sample < dense_count; ++sample)
        for (size_t field = 0; field < 4; ++field) dense_motion[4 * sample + field] = InterpolateMotionField(motion, sample, sample_factor, field);
    const auto &sample_motion = resolved_full ? dense_motion : motion;
    std::vector<float> scrape(p.Frames), elastic(p.Frames), damping(p.Frames), morph(p.Frames);
    double max_height = 0;
    std::vector<float> filtered_trajectory(temporal ? 3 * size_t(p.Frames) : 0);
    if (gpu_enabled) {
        struct Block {
            uint32_t Count, Levels, Frames;
            float Spacing, Mass, Beta1, Beta2, Radius, Eccentricity, AlphaScale;
            uint32_t VerticalMode;
        };
        const auto parameters = Upload(gpu, Block{uint32_t(count), p.Levels, sample_frames, float(p.Spacing), float(p.Mass), float(p.Beta1), float(p.Beta2), float(p.Radius), float(p.Eccentricity), float(p.AlphaScale), vertical_mode});
        const std::vector<float> motion_float(sample_motion.begin(), sample_motion.end());
        struct Location {
            uint32_t Cell;
            float Fraction;
        };
        std::vector<Location> coordinates(sample_frames);
        for (uint32_t frame = 0; frame < sample_frames; ++frame) {
            const double grid_x = sample_motion[4 * frame] / p.Spacing;
            if (grid_x < 0 || grid_x > count - 1) throw std::runtime_error("Reproduction path exceeds measured profile");
            const size_t cell = std::min(size_t(grid_x), count - 2);
            coordinates[frame] = {uint32_t(cell), float(grid_x - cell)};
        }
        const auto location = Upload<Location>(gpu, coordinates);
        const std::vector<float> vertical_curvature(vertical_only ? raw.begin() : raw.end(), raw.end());
        const auto surface = Upload<float>(gpu, vertical_only ? vertical_curvature : atlas), path = Upload<float>(gpu, motion_float), output = CreateBuffer(gpu, 4 * size_t(p.Frames) * sizeof(float)), status = CreateBuffer(gpu, size_t(sample_frames) * sizeof(uint32_t));
        if (temporal) {
            const auto sampled = CreateBuffer(gpu, 3 * size_t(sample_frames) * sizeof(float));
            const auto kernel = CreateKernel(gpu, vertical_only ? "AgarwalVerticalTrajectoryV2" : "AgarwalAtlasTrajectoryV2");
            const std::array bindings{GpuBinding{parameters, 0}, GpuBinding{surface, 1}, GpuBinding{path, 2}, GpuBinding{sampled, 3}, GpuBinding{status, 4}, GpuBinding{location, 5}};
            BeginGpu(gpu);
            DispatchGpu(gpu, kernel, bindings, {sample_frames});
            SubmitGpu(gpu);
            WaitGpu(gpu);
            for (auto value : BufferSpan<uint32_t>(status))
                if (value) throw std::runtime_error("GPU trajectory path invalid");
            const auto filtered = CreateBuffer(gpu, filtered_trajectory.size() * sizeof(float));
            if (resolved_full) {
                struct DenseFilterBlock {
                    uint32_t Frames, DenseFrames, Factor, SampleRate;
                    float HalfWidth, Mass, Beta1, Beta2, Radius, Eccentricity, SigmaRatio;
                };
                const uint32_t force_frames = bandlimit ? sample_frames : p.Frames;
                const auto filter_parameters = Upload(gpu, DenseFilterBlock{force_frames, sample_frames, bandlimit ? 1u : sample_factor, p.SampleRate * (bandlimit ? sample_factor : 1), float(p.HalfWidth), float(p.Mass), float(p.Beta1), float(p.Beta2), float(p.Radius), float(p.Eccentricity), float(sigma_ratio)});
                const std::vector<float> base_motion(motion.begin(), motion.end());
                const auto base_path = bandlimit ? path : Upload<float>(gpu, base_motion);
                const auto physical_force = bandlimit ? CreateBuffer(gpu, 4 * size_t(force_frames) * sizeof(float)) : output;
                const auto physical_trajectory = bandlimit ? CreateBuffer(gpu, 3 * size_t(force_frames) * sizeof(float)) : filtered;
                const auto decimation = bandlimit ? CreateDecimatePlan(gpu, force_frames, 3, sample_factor) : GpuDecimatePlan{};
                const auto retention = bandlimit ? CreateRetainPlan(gpu, p.Frames, force_frames, sample_factor, 3) : RetainPlan{};
                const auto filter_kernel = CreateKernel(gpu, "AgarwalDownsampleFullForce");
                const std::array filter_bindings{GpuBinding{filter_parameters, 0}, GpuBinding{sampled, 1}, GpuBinding{base_path, 2}, GpuBinding{physical_force, 3}, GpuBinding{status, 4}, GpuBinding{physical_trajectory, 5}};
                BeginGpu(gpu);
                DispatchGpu(gpu, filter_kernel, filter_bindings, {force_frames});
                if (bandlimit) {
                    const GpuBuffer input_planes{physical_force.Data, 3 * size_t(force_frames) * sizeof(float), physical_force.Address};
                    const GpuBuffer output_planes{output.Data, 3 * size_t(p.Frames) * sizeof(float), output.Address};
                    DispatchDecimateGpu(gpu, decimation, input_planes, output_planes);
                    DispatchRetainedFields(gpu, retention, physical_trajectory, filtered);
                }
                SubmitGpu(gpu);
                WaitGpu(gpu);
            } else {
                struct TemporalBlock {
                    uint32_t Frames, SampleRate;
                    float HalfWidth, Mass, Beta1, Beta2, Radius, Eccentricity;
                    uint32_t VerticalMode;
                };
                const auto temporal_parameters = Upload(gpu, TemporalBlock{p.Frames, p.SampleRate, float(p.HalfWidth), float(p.Mass), float(p.Beta1), float(p.Beta2), float(p.Radius), float(p.Eccentricity), vertical_mode});
                const auto temporal_kernel = CreateKernel(gpu, "AgarwalTemporalForcesV2");
                const std::array temporal_bindings{GpuBinding{temporal_parameters, 0}, GpuBinding{sampled, 1}, GpuBinding{path, 2}, GpuBinding{output, 3}, GpuBinding{status, 4}, GpuBinding{filtered, 5}};
                BeginGpu(gpu);
                DispatchGpu(gpu, temporal_kernel, temporal_bindings, {p.Frames});
                SubmitGpu(gpu);
                WaitGpu(gpu);
            }
            const auto values = BufferSpan<float>(filtered);
            std::copy(values.begin(), values.end(), filtered_trajectory.begin());
        } else {
            const auto kernel = CreateKernel(gpu, "AgarwalAtlasForcesV2");
            const std::array bindings{GpuBinding{parameters, 0}, GpuBinding{surface, 1}, GpuBinding{path, 2}, GpuBinding{output, 3}, GpuBinding{status, 4}, GpuBinding{location, 5}};
            BeginGpu(gpu);
            DispatchGpu(gpu, kernel, bindings, {p.Frames});
            SubmitGpu(gpu);
            WaitGpu(gpu);
        }
        for (auto value : BufferSpan<uint32_t>(status))
            if (value) throw std::runtime_error("GPU reproduction path or penetration invalid");
        const auto values = BufferSpan<float>(output);
        std::copy_n(values.begin(), p.Frames, scrape.begin());
        std::copy_n(values.begin() + p.Frames, p.Frames, elastic.begin());
        std::copy_n(values.begin() + 2 * p.Frames, p.Frames, damping.begin());
        if (bandlimit) {
            for (uint32_t n = 0; n < p.Frames; ++n) morph[n] = float(motion[4 * n + 3]);
        } else std::copy_n(values.begin() + 3 * p.Frames, p.Frames, morph.begin());
        for (uint32_t level = 0; !vertical_only && level < p.Levels; ++level)
            for (size_t x = 0; x < count; ++x) max_height = std::max(max_height, std::abs(double(atlas[size_t(level) * 3 * count + x])));
    } else {
        std::vector<Trajectory> sampled(sample_frames);
        for (uint32_t frame = 0; frame < sample_frames; ++frame) {
            const double x = sample_motion[4 * frame], normal = sample_motion[4 * frame + 2];
            const double alpha = Alpha(normal, {.NormalMin = 1, .NormalMax = 6});
            const double gl = (alpha - .01) / .04 * (p.Levels - 1), gx = x / p.Spacing;
            if (gx < 0 || gx > count - 1) throw std::runtime_error("Reproduction path exceeds measured profile");
            const auto l0 = std::min(uint32_t(gl), p.Levels - 2);
            const size_t x0 = std::min(size_t(gx), count - 2);
            const auto sample = [&](uint32_t field) {
                if (temporal) {
                    const double fraction = gx - x0, distance = fraction * p.Spacing;
                    const auto primitive = [&](uint32_t level) {
                        const size_t base = size_t(level) * 3 * count + x0;
                        const double h0 = atlas[base], s0 = atlas[base + count], c0 = atlas[base + 2 * count], delta = double(atlas[base + 2 * count + 1]) - c0;
                        if (field == 0) return h0 + s0 * distance + distance * distance * (.5 * c0 + delta * fraction / 6);
                        if (field == 1) return s0 + distance * (c0 + .5 * delta * fraction);
                        return c0 + delta * fraction;
                    };
                    return std::lerp(primitive(l0), primitive(l0 + 1), gl - l0);
                }
                const size_t a = size_t(l0) * 3 * count + field * count + x0, b = a + 3 * count;
                return std::lerp(std::lerp(double(atlas[a]), double(atlas[a + 1]), gx - x0), std::lerp(double(atlas[b]), double(atlas[b + 1]), gx - x0), gl - l0);
            };
            if (vertical_only) {
                const double constrained = vertical_mode == 2 ? ConstrainCurvature(std::lerp(raw[x0], raw[x0 + 1], gx - x0), alpha * p.AlphaScale) : std::lerp(ConstrainCurvature(raw[x0], alpha * p.AlphaScale), ConstrainCurvature(raw[x0 + 1], alpha * p.AlphaScale), gx - x0);
                sampled[frame] = {.CurvatureX = constrained};
            } else sampled[frame] = {.Height = sample(0), .SlopeX = sample(1), .CurvatureX = sample(2)};
        }
        const uint32_t force_frames = bandlimit ? sample_frames : p.Frames;
        const uint32_t force_stride = bandlimit ? 1 : sample_factor;
        const auto &force_motion = bandlimit ? sample_motion : motion;
        std::vector<double> dense_forces(bandlimit ? 3 * size_t(sample_frames) : 0);
        for (uint32_t frame = 0; frame < force_frames; ++frame) {
            const double x = force_motion[4 * frame], velocity = force_motion[4 * frame + 1], normal = force_motion[4 * frame + 2];
            Trajectory trajectory = sampled[size_t(frame) * force_stride];
            if (temporal) {
                const double half_width = p.HalfWidth * (double(p.SampleRate) / 44100) * Alpha(normal, {.NormalMin = 1, .NormalMax = 6}) / .03 * sample_factor;
                const double sigma = std::max((resolved_full ? sigma_ratio : .4) * half_width, 1e-12);
                const int64_t radius = int64_t(std::ceil(half_width)), center = int64_t(frame) * force_stride;
                double normalization = 0;
                trajectory = {};
                for (int64_t m = std::max(int64_t(0), center - radius); m <= std::min(int64_t(sample_frames) - 1, center + radius); ++m) {
                    const double d = double(m) - center, weight = std::exp(-.5 * d * d / (sigma * sigma));
                    trajectory.Height += weight * sampled[m].Height;
                    trajectory.SlopeX += weight * sampled[m].SlopeX;
                    trajectory.CurvatureX += weight * sampled[m].CurvatureX;
                    normalization += weight;
                }
                trajectory.Height /= normalization;
                trajectory.SlopeX /= normalization;
                trajectory.CurvatureX /= normalization;
                if (!bandlimit || frame % sample_factor == 0) {
                    const uint32_t retained = bandlimit ? frame / sample_factor : frame;
                    filtered_trajectory[retained] = float(trajectory.Height);
                    filtered_trajectory[p.Frames + retained] = float(trajectory.SlopeX);
                    filtered_trajectory[2 * p.Frames + retained] = float(trajectory.CurvatureX);
                }
            }
            if (!bandlimit) morph[frame] = float(motion[4 * frame + 3]);
            if (vertical_only) {
                scrape[frame] = float(p.Mass * trajectory.CurvatureX * velocity * velocity);
                continue;
            }
            // Mass is also the vertical-term coefficient in component ablations.
            const Force force{p.Beta1 * std::pow(std::abs(velocity * trajectory.SlopeX), p.Beta2), p.Mass * trajectory.CurvatureX * velocity * velocity, 0};
            if (!bandlimit) scrape[frame] = float(force.Horizontal + force.Vertical);
            const double rho = p.Radius - p.Eccentricity * std::cos(x / p.Radius) + trajectory.Height;
            if (!(rho >= 0)) throw std::runtime_error("Negative reproduction rolling penetration");
            const double rho_dot = p.Eccentricity / p.Radius * velocity * std::sin(x / p.Radius) + velocity * trajectory.SlopeX;
            if (bandlimit) {
                dense_forces[frame] = force.Horizontal + force.Vertical;
                dense_forces[sample_frames + frame] = rho * std::sqrt(rho);
                dense_forces[2 * size_t(sample_frames) + frame] = rho * std::sqrt(rho) * rho_dot;
            } else {
                elastic[frame] = float(rho * std::sqrt(rho));
                damping[frame] = float(rho * std::sqrt(rho) * rho_dot);
            }
            max_height = std::max(max_height, std::abs(trajectory.Height));
        }
        if (bandlimit) {
            std::vector<float> components(3 * size_t(p.Frames));
            DecimateReference(dense_forces, sample_frames, 3, sample_factor, components);
            std::copy_n(components.begin(), p.Frames, scrape.begin());
            std::copy_n(components.begin() + p.Frames, p.Frames, elastic.begin());
            std::copy_n(components.begin() + 2 * p.Frames, p.Frames, damping.begin());
            for (uint32_t n = 0; n < p.Frames; ++n) morph[n] = float(motion[4 * n + 3]);
        }
    }
    if (resolved_full) {
        for (const auto *field : {&scrape, &elastic, &damping, &morph, &filtered_trajectory})
            for (float value : *field)
                if (!std::isfinite(value)) throw std::runtime_error("Nonfinite resolved full-model trajectory or force");
    }
    if (temporal) WriteBinary<float>(directory / "trajectory.f32", filtered_trajectory);
    WriteBinary<float>(directory / "scrape.f32", scrape);
    WriteBinary<float>(directory / "elastic.f32", elastic);
    WriteBinary<float>(directory / "damping.f32", damping);
    WriteBinary<float>(directory / "morph.f32", morph);
    std::cout << "Prepared measured profile: samples=" << count << " alpha_levels=" << p.Levels << " vertical_mode=" << vertical_mode << " oversampling=" << sample_factor << " sigma_ratio=" << sigma_ratio << " bandlimit=" << bandlimit << " max_trajectory_height=" << max_height << '\n';
}

void Render(const std::filesystem::path &directory, const std::filesystem::path &prefix, const Parameters &p) {
    auto force = ReadBinary<float>(directory / "scrape.f32");
    const auto elastic = ReadBinary<float>(directory / "elastic.f32"), damping = ReadBinary<float>(directory / "damping.f32");
    auto morph = ReadBinary<float>(directory / "morph.f32");
    if (force.size() != p.Frames || elastic.size() != p.Frames || damping.size() != p.Frames || morph.size() != p.Frames) throw std::runtime_error("Prepared force extent mismatch");
    for (uint32_t frame = 0; frame < p.Frames; ++frame) force[frame] += float(p.Stiffness * elastic[frame] + p.Dissipation * damping[frame]);
    std::vector<FiniteMode> modes;
    std::ifstream mode_file(directory / "modes.txt");
    FiniteMode mode;
    while (mode_file >> mode.Frequency >> mode.Decay >> mode.Amplitude0 >> mode.Amplitude1) modes.push_back(mode);
    if (modes.empty()) throw std::runtime_error("No reproduction modes");
    auto gpu = CreateGpu();
    const auto output = ConvolveFiniteModesGpu(gpu, force, morph, modes, p.SampleRate, p.Taps, float(p.Gain));
    WriteWave(prefix.string() + ".wav", p.SampleRate, 1, output);
    WriteWave(prefix.string() + "-force.wav", p.SampleRate, 1, force);
    std::cout << "Rendered " << prefix << " modes=" << modes.size() << " frames=" << output.size() << '\n';
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc == 6 && std::string(argv[1]) == "filter") {
            const auto force = ReadBinary<float>(argv[2]), response = ReadBinary<float>(argv[3]);
            auto gpu = CreateGpu();
            const auto output = ConvolveFixedGpu(gpu, force, response);
            WriteWave(argv[4], uint32_t(std::stoul(argv[5])), 1, output);
            return 0;
        }
        if (argc < 3) throw std::invalid_argument("Usage: agarwalReproduce prepare[-reference] CASE_DIRECTORY | prepare-temporal[-reference] CASE_DIRECTORY [OVERSAMPLING [SIGMA_RATIO]] [--bandlimit] | prepare-temporal-vertical[-reference] CASE_DIRECTORY [OVERSAMPLING [SIGMA_RATIO]] [--bandlimit] | prepare-temporal-vertical-pointwise[-reference] CASE_DIRECTORY [OVERSAMPLING [SIGMA_RATIO]] [--bandlimit] | render CASE_DIRECTORY OUTPUT_PREFIX");
        const bool bandlimit = argc > 3 && std::string(argv[argc - 1]) == "--bandlimit";
        const int positional_argc = argc - int(bandlimit);
        const std::filesystem::path directory = argv[2];
        const auto parameters = ReadParameters(directory);
        const std::string command{argv[1]};
        const bool temporal_command = command.starts_with("prepare-temporal");
        const uint32_t oversampling = [&] {
            if (!temporal_command || positional_argc < 4) return uint32_t(1);
            const std::string argument{argv[3]};
            if (argument.empty() || argument.find_first_not_of("0123456789") != std::string::npos) throw std::invalid_argument("Oversampling must be an integer from 1 through 128");
            const unsigned long value = std::stoul(argument);
            if (value < 1 || value > 128) throw std::invalid_argument("Oversampling must be an integer from 1 through 128");
            return uint32_t(value);
        }();
        const double sigma_ratio = [&] {
            if (!temporal_command || positional_argc != 5) return .4;
            const std::string argument{argv[4]};
            size_t consumed{};
            const double value = std::stod(argument, &consumed);
            if (consumed != argument.size() || !std::isfinite(value) || value < .05 || value > 1) throw std::invalid_argument("Gaussian sigma ratio must be finite and from .05 through 1");
            return value;
        }();
        if (bandlimit && (!temporal_command || oversampling < 2)) throw std::invalid_argument("--bandlimit requires a temporal command with oversampling at least 2");
        if ((temporal_command && positional_argc != 3 && positional_argc != 4 && positional_argc != 5) || (!temporal_command && command != "render" && positional_argc != 3)) throw std::invalid_argument("Invalid reproduction argument count");
        if (std::string(argv[1]) == "prepare" || std::string(argv[1]) == "prepare-reference") Prepare(directory, parameters, std::string(argv[1]) == "prepare");
        else if (std::string(argv[1]) == "prepare-temporal" || std::string(argv[1]) == "prepare-temporal-reference") Prepare(directory, parameters, std::string(argv[1]) == "prepare-temporal", true, 0, oversampling, sigma_ratio, bandlimit);
        else if (std::string(argv[1]) == "prepare-temporal-vertical" || std::string(argv[1]) == "prepare-temporal-vertical-reference") Prepare(directory, parameters, std::string(argv[1]) == "prepare-temporal-vertical", true, 1, oversampling, sigma_ratio, bandlimit);
        else if (std::string(argv[1]) == "prepare-temporal-vertical-pointwise" || std::string(argv[1]) == "prepare-temporal-vertical-pointwise-reference") Prepare(directory, parameters, std::string(argv[1]) == "prepare-temporal-vertical-pointwise", true, 2, oversampling, sigma_ratio, bandlimit);
        else if (std::string(argv[1]) == "render" && positional_argc == 4) Render(directory, argv[3], parameters);
        else throw std::invalid_argument("Invalid reproduction command");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
