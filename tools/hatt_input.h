#pragma once
#include "hatt/Hatt.h"
#include <fstream>
#include <string>

namespace surface_audio::hatt {
struct Input {
    uint32_t Frames;
    std::vector<Texture> Textures;
    std::vector<Control> Controls;
    std::vector<float> Noise;
};
inline Input ReadInput(const std::string &path) {
    std::ifstream file(path);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    uint32_t count{}, frames{};
    file >> count >> frames;
    if (!count || count > 100 || !frames || frames > 1000000) throw std::invalid_argument("Invalid HaTT input dimensions");
    Input result{frames, std::vector<Texture>(count), std::vector<Control>(size_t(count) * frames), std::vector<float>(size_t(count) * frames)};
    for (uint32_t voice = 0; voice < count; ++voice) {
        auto &t = result.Textures[voice];
        uint32_t models{}, triangles{}, ar{}, ma{};
        file >> t.SampleRate >> t.Friction >> t.MaximumSpeed >> t.MaximumForce >> models >> triangles >> ar >> ma;
        if (!models || models > 10000 || !triangles || triangles > 20000 || ar > MaximumOrder || ma > MaximumOrder) throw std::invalid_argument("Invalid HaTT model dimensions");
        t.Models.resize(models);
        t.Triangles.resize(triangles);
        for (auto &m : t.Models) {
            file >> m.Speed >> m.Force >> m.Variance >> m.Gain;
            m.ArLsf.resize(ar);
            m.MaLsf.resize(ma);
            for (auto &v : m.ArLsf) file >> v;
            for (auto &v : m.MaLsf) file >> v;
        }
        for (auto &triangle : t.Triangles)
            for (auto &v : triangle) file >> v;
        for (size_t i = size_t(voice) * frames; i < size_t(voice + 1) * frames; ++i) file >> result.Controls[i].Speed >> result.Controls[i].Force >> result.Noise[i];
    }
    return result;
}
}
