#include "rough/PatchContact.h"
#include "core/BinaryFile.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

using namespace surface_audio;
using namespace surface_audio::rough;
int main(int argc, char **argv) {
    try {
        if (argc != 3 && argc != 4) throw std::invalid_argument("Usage: anderssonReproduce INPUT OUTPUT [--equilibrium]");
        const bool equilibrium = argc == 4 && std::string(argv[3]) == "--equilibrium";
        if (argc == 4 && !equilibrium) throw std::invalid_argument("Unknown mode");
        const std::filesystem::path input = argv[1], output = argv[2];
        std::ifstream settings(input / "dimensions.txt");
        unsigned nx{}, ny{}, taps{}, knots{};
        double rate{};
        if (!(settings >> nx >> ny >> taps >> knots >> rate) || !nx || !ny || size_t(nx) * ny > 4096 || taps < 2 ||
            taps > 65536 || knots < 2 || knots > 65536 || !std::isfinite(rate) || rate <= 0)
            throw std::invalid_argument("Invalid prepared contact dimensions");
        const size_t n = size_t(nx) * ny;
        auto heights = ReadBinary<double>(input / "heights.f64");
        const auto table = ReadBinary<std::array<double, 2>>(input / "springs.f64");
        const auto compact = ReadBinary<double>(input / "green.f64");
        const auto schedule = ReadBinary<std::array<double, 2>>(input / "schedule.f64");
        if (heights.size() != n || table.size() != n * knots || compact.size() != n * taps || schedule.empty())
            throw std::invalid_argument("Prepared contact input dimensions disagree");
        const double maximum = *std::max_element(heights.begin(), heights.end());
        for (double &height : heights) {
            if (!std::isfinite(height)) throw std::invalid_argument("Nonfinite surface height");
            height -= maximum;
        }
        for (const auto [indentation, hold] : schedule)
            if (!std::isfinite(indentation) || !std::isfinite(hold) || hold < 1 || hold > 1e9 || hold != std::floor(hold))
                throw std::invalid_argument("Invalid indentation schedule");
        std::vector<PatchSpring> springs;
        for (size_t i = 0; i < n; ++i) springs.push_back(MakePatchSpring(std::span(table).subspan(i * knots, knots)));
        const unsigned response_taps = equilibrium ? 1 : taps;
        std::vector<double> green(n * n * response_taps);
        for (unsigned lag = 0; lag < taps; ++lag)
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j) {
                    const size_t dx = size_t(std::abs(int(i % nx) - int(j % nx)));
                    const size_t dy = size_t(std::abs(int(i / nx) - int(j / nx)));
                    green[((equilibrium ? 0 : lag) * n + i) * n + j] += compact[lag * n + dy * nx + dx];
                }
        const auto model = MakePatchContact(std::move(springs), response_taps, 1 / rate, green);
        green.clear(); green.shrink_to_fit();
        auto state = MakePatchContactState(model);
        std::vector<double> approach(n);
        std::filesystem::create_directories(output);
        std::ofstream curve(output / "curve.csv"), trace(output / "force.f64", std::ios::binary);
        curve.exceptions(std::ios::badbit | std::ios::failbit);
        trace.exceptions(std::ios::badbit | std::ios::failbit);
        curve << std::setprecision(17) << "time_s,indentation_m,force_N,residual_m,iterations\n";
        uint64_t step{};
        for (const auto [indentation, hold] : schedule) {
            for (size_t i = 0; i < n; ++i) approach[i] = indentation + heights[i];
            PatchContactResult result;
            double total{};
            for (uint64_t tick = 0; tick < (equilibrium ? 1 : uint64_t(hold)); ++tick) {
                result = StepPatchContact(model, state, approach, 1e-13, 100);
                if (!result.Converged)
                    throw std::runtime_error("Contact failed at sample " + std::to_string(step) + "; residual=" + std::to_string(result.Residual) +
                                             ". Check spring domain, time resolution and input response; no force was clipped.");
                total = std::accumulate(state.Force.begin(), state.Force.end(), 0.);
                if (!equilibrium) trace.write(reinterpret_cast<const char *>(&total), sizeof(total));
                ++step;
            }
            curve << (equilibrium ? 0 : double(step - 1) / rate) << ',' << indentation << ',' << total << ',' << result.Residual << ',' << result.Iterations << '\n';
        }
        WriteBinary<double>(output / "final_force.f64", state.Force);
        WriteBinary<double>(output / "final_compression.f64", state.Compression);
        WriteBinary<double>(output / "final_displacement.f64", state.Displacement);
        std::cout << schedule.size() << " indentation levels, " << n << " elements, " << step << (equilibrium ? " equilibrium solves\n" : " time steps\n");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
