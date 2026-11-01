#include "Collision.h"
#include <array>
#include <fstream>

namespace surface_audio::collision {
Model<double> ReadModel(const std::filesystem::path &path) {
    std::ifstream input(path);
    Model<double> model;
    auto &p = model.Config;
    if (!(input >> p.Dofs >> p.Contacts >> p.StiffnessEntries >> p.LossEntries >> p.ContactEntries >> p.Smoothing >> p.Dt >> p.BowForce >> p.BowSpeed >> p.BowShape)
        || !p.Dofs || p.Dofs > 20000 || p.Contacts > 1024 || p.Smoothing > 2 || !(p.Dt > 0) || !(p.BowShape > 0)
        || p.StiffnessEntries > 100 * p.Dofs || p.LossEntries > 100 * p.Dofs || p.ContactEntries > p.Contacts * p.Dofs)
        throw std::invalid_argument("Invalid collision model dimensions");
    for (double value : {p.Dt, p.BowForce, p.BowSpeed, p.BowShape})
        if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite collision time or bow control");
    if (p.BowForce < 0) throw std::invalid_argument("Negative bow force");
    const unsigned n = p.Dofs, c = p.Contacts;
    model.Data.resize(4 * n + 5 * c + p.StiffnessEntries + p.LossEntries + p.ContactEntries + c * c);
    for (double &value : model.Data) if (!(input >> value) || !std::isfinite(value)) throw std::invalid_argument("Invalid collision coefficient");
    model.Indices.resize(2 * (n + 1) + c + 1 + p.StiffnessEntries + p.LossEntries + p.ContactEntries);
    for (auto &index : model.Indices) if (!(input >> index)) throw std::invalid_argument("Invalid collision sparse index");
    model.State.resize(StateSize(p));
    for (unsigned i = 0; i < 2 * n; ++i) if (!(input >> model.State[i]) || !std::isfinite(model.State[i])) throw std::invalid_argument("Invalid collision initial state");
    for (unsigned i = 0; i < c; ++i) if (!(input >> model.State[3 * n + i]) || !std::isfinite(model.State[3 * n + i])) throw std::invalid_argument("Invalid collision auxiliary state");
    input >> std::ws;
    if (!input.eof()) throw std::invalid_argument("Extra collision model field");
    unsigned offset = 0;
    for (auto [rows, entries] : {std::pair{n, p.StiffnessEntries}, std::pair{n, p.LossEntries}, std::pair{c, p.ContactEntries}}) {
        const auto row = model.Indices.data() + offset, col = row + rows + 1;
        if (row[0] != 0 || row[rows] != entries) throw std::invalid_argument("Invalid collision sparse rows");
        for (unsigned i = 0; i < rows; ++i) if (row[i] > row[i + 1]) throw std::invalid_argument("Unsorted collision sparse rows");
        for (unsigned i = 0; i < entries; ++i) if (col[i] >= n) throw std::invalid_argument("Collision column out of bounds");
        offset += rows + 1 + entries;
    }
    for (unsigned i = 0; i < n; ++i) if (model.Data[i] <= 0 || model.Data[n + i] < 0) throw std::invalid_argument("Invalid collision mass or damping");
    for (unsigned a = 0; a < c; ++a) if (model.Data[4 * n + c + a] < 0 || model.Data[4 * n + 2 * c + a] < 1) throw std::invalid_argument("Invalid collision potential");
    for (unsigned i = 0; i < n; ++i) model.State[n + i] = model.State[i] - model.State[n + i];
    model.State.back() = -p.BowSpeed;
    return model;
}
Model<float> FloatModel(const Model<double> &model) {
    const auto &p = model.Config;
    Model<float> result{{p.Dofs, p.Contacts, p.StiffnessEntries, p.LossEntries, p.ContactEntries, p.Smoothing, float(p.Dt), float(p.BowForce), float(p.BowSpeed), float(p.BowShape)}, {}, {}, model.Indices};
    result.Data.assign(model.Data.begin(), model.Data.end());
    result.State.assign(model.State.begin(), model.State.end());
    for (float value : result.Data) if (!std::isfinite(value)) throw std::invalid_argument("Collision coefficient exceeds float range");
    for (float value : result.State) if (!std::isfinite(value)) throw std::invalid_argument("Collision state exceeds float range");
    for (float value : {result.Config.Dt, result.Config.BowForce, result.Config.BowSpeed, result.Config.BowShape})
        if (!std::isfinite(value)) throw std::invalid_argument("Collision control exceeds float range");
    if (result.Config.Dt <= 0) throw std::invalid_argument("Collision time step underflows float");
    for (unsigned i = 0; i < p.Dofs; ++i) if (result.Data[i] <= 0) throw std::invalid_argument("Collision mass underflows float");
    return result;
}
template<typename T> void Render(const Model<T> &model, std::span<T> state, std::span<Sample<T>> output) {
    if (state.size() != StateSize(model.Config)) throw std::invalid_argument("Invalid collision state size");
    std::vector<unsigned> active(model.Config.Contacts);
    for (auto &sample : output) {
        sample = Step(model.Config, model.Data.data(), model.Indices.data(), state.data(), active.data());
        if (!std::isfinite(sample.Energy) || !std::isfinite(sample.Output)) throw std::runtime_error("Nonfinite collision result");
    }
}
template void Render(const Model<double> &, std::span<double>, std::span<Sample<double>>);
template void Render(const Model<float> &, std::span<float>, std::span<Sample<float>>);
GpuSource CreateGpuSource(Gpu &gpu, const Model<float> &model, unsigned voices, unsigned frames) {
    if (!voices || !frames || uint64_t(voices) * StateSize(model.Config) > UINT32_MAX || uint64_t(voices) * frames > UINT32_MAX) throw std::invalid_argument("Invalid collision GPU dimensions");
    std::vector<float> states;
    for (unsigned i = 0; i < voices; ++i) states.insert(states.end(), model.State.begin(), model.State.end());
    return {model.Config, voices, frames, model.Config.Dofs < 32 ? 1u : 128u, Upload<float>(gpu, model.Data), Upload<unsigned>(gpu, model.Indices), Upload<float>(gpu, states),
        CreateBuffer(gpu, std::max(1u, voices * model.Config.Contacts) * sizeof(unsigned)), CreateBuffer(gpu, uint64_t(voices) * frames * sizeof(Sample<float>)), CreateKernel(gpu, "CollisionSource")};
}
void EncodeSource(Gpu &gpu, const GpuSource &s) {
    const auto config = BatchUpload(gpu, s.Config), block = BatchUpload(gpu, std::array{s.Voices, s.Frames});
    const std::array bindings{GpuBinding{s.Data, 0}, GpuBinding{s.Indices, 1}, GpuBinding{s.State, 2}, GpuBinding{s.Active, 3}, GpuBinding{s.Output, 4}, GpuBinding{config, 5}, GpuBinding{block, 6}};
    DispatchGroupsGpu(gpu, s.Kernel, bindings, {s.Voices}, {s.Threads});
}
}
