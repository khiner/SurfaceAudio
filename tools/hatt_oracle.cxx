#include "AccSynthHashMatrix.h"
#include "hatt_input.h"
#include "sharedInit.h"
#include <iostream>

int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::invalid_argument("Usage: hattOracle input.txt coefficients.f64");
        const auto input = surface_audio::hatt::ReadInput(argv[1]);
        std::ofstream output(argv[2], std::ios::binary);
        for (size_t voice = 0; voice < input.Textures.size(); ++voice) {
            const auto &t = input.Textures[voice];
            std::vector<float> speeds, forces;
            std::vector<int> first, second, third;
            for (const auto &m : t.Models) {
                speeds.push_back(float(m.Speed));
                forces.push_back(float(m.Force));
            }
            for (const auto &v : t.Triangles) {
                first.push_back(int(v[0] + 1));
                second.push_back(int(v[1] + 1));
                third.push_back(int(v[2] + 1));
            }
            AccSynthHashTable table(0, int(t.Models.size()), speeds.data(), forces.data());
            isARMA = !t.Models.front().MaLsf.empty();
            for (size_t i = 0; i < t.Models.size(); ++i) {
                const auto &m = t.Models[i];
                std::vector<float> ar(m.ArLsf.begin(), m.ArLsf.end()), ma(m.MaLsf.begin(), m.MaLsf.end());
                table.hashMap[i] = isARMA ? AccSynthHashEntry(0, forces[i], speeds[i], ar.data(), ma.data(), float(m.Variance), float(m.Gain), int(ar.size()), int(ma.size()), int(first.size()), int(speeds.size()), first.data(), second.data(), third.data(), float(t.MaximumSpeed), float(t.MaximumForce), float(t.Friction)) : AccSynthHashEntry(0, forces[i], speeds[i], ar.data(), float(m.Variance), int(ar.size()), int(first.size()), int(speeds.size()), first.data(), second.data(), third.data(), float(t.MaximumSpeed), float(t.MaximumForce), float(t.Friction));
                if (!isARMA) table.hashMap[i].numMACoeff = 0;
            }
            for (const auto &control : std::span(input.Controls).subspan(voice * input.Frames, input.Frames)) {
                table.HashAndInterp2(float(control.Speed), float(control.Force));
                const auto ar = SynthesisFlag_Buffer1 ? filtCoeff_buf1 : filtCoeff_buf2;
                const auto ma = SynthesisFlag_Buffer1 ? filtMACoeff_buf1 : filtMACoeff_buf2;
                const double gain = isARMA ? (SynthesisFlag_Buffer1 ? filtGain_buf1 : filtGain_buf2) : 1;
                std::array<double, 53> record{};
                record[0] = 1;
                record[26] = gain;
                for (int i = 0; i < coeffNum; ++i) record[i + 1] = ar[i];
                for (int i = 0; i < MAcoeffNum; ++i) record[i + 27] = gain * ma[i];
                record[52] = SynthesisFlag_Buffer1 ? filtVariance_buf1 : filtVariance_buf2;
                output.write(reinterpret_cast<const char *>(record.data()), sizeof(record));
            }
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
