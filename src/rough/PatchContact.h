#pragma once
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace surface_audio::rough {
struct PatchSpringKnot {
    double Compression, Stiffness, Force;
};
using PatchSpring = std::vector<PatchSpringKnot>;
struct PatchContactModel {
    std::vector<PatchSpring> Springs;
    std::vector<double> Response;
    unsigned Taps{};
};
struct PatchContactState {
    std::vector<double> Compression, Force, Displacement, History;
    uint64_t Samples{};
    bool operator==(const PatchContactState &) const = default;
};
struct PatchContactResult {
    double Residual{};
    unsigned Iterations{};
    bool Converged{};
};
// Samples contain compression in metres and stiffness in N/m, starting at zero compression with nondecreasing stiffness.
// Stiffness is interpolated linearly and integrated exactly; compression beyond the supplied domain is rejected.
PatchSpring MakePatchSpring(std::span<const std::array<double, 2>> samples);
// Returns force in N and stiffness in N/m; separated springs exert no force.
std::array<double, 2> EvaluatePatchSpring(const PatchSpring &, double compression);
// Green functions have units m/(N s), ordered [lag][receiving element][force element]; construction multiplies by dt once.
// Both bodies' responses may be summed before construction; the caller supplies physically valid response functions.
PatchContactModel MakePatchContact(std::vector<PatchSpring>, unsigned taps, double dt, std::span<const double> green);
PatchContactState MakePatchContactState(const PatchContactModel &);
// approach = z1 - z2 in metres; positive values indent the undeformed bodies, and displacement opens the interface.
// Solves Andersson-Kropp Equations 29-33 in FP64; failed steps preserve the entire state, including convolution history.
PatchContactResult StepPatchContact(const PatchContactModel &, PatchContactState &, std::span<const double> approach,
                                    double tolerance = 1e-12, unsigned iterations = 50);
}
