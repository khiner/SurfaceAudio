#pragma once
#include "core/Flexural.h"
#include <array>
#include <span>
#include <vector>

namespace surface_audio::rough {
struct Surface {
    double Step{};
    std::vector<double> Height, Shapes, Gravity;
    std::vector<ModalDynamics> Modes;
};
struct Model {
    Surface Bottom, Top;
    double TimeStep{}, Penalty{};
};
enum class ContactMethod { Penalty,
                           Multiplier };
struct State {
    std::vector<double> Displacement, Previous, Next, Force;
    std::vector<double> Jacobian, Gap, Weight, Multiplier;
    unsigned Rows{};
};
struct ContactResult {
    double NormalForce{}, MaxPenetration{}, Residual{}, ElasticEnergy{};
    unsigned Contacts{}, Iterations{};
    bool Converged{};
};
struct Interpolation {
    std::array<unsigned, 4> Node{};
    std::array<double, 4> Weight{};
};

// Heights protrude toward contact; displacement and gravity use a common upward-positive world axis.
Surface MakeBeamSurface(BeamProperties, unsigned bending_modes, std::span<const double> heights, double dt, double gravity = 0);
Model MakeModel(Surface bottom, Surface top, double dt, double penalty);
// Initial acceleration includes gravity; set Previous explicitly for an initially loaded contact equilibrium.
State MakeState(const Model &, std::span<const double> displacement = {}, std::span<const double> velocity = {});
Interpolation Interpolate(unsigned nodes, double coordinate);
// Offset locates the top origin on the bottom; separation measures the undeformed reference-plane distance.
void PrepareContact(const Model &, State &, double offset, double separation);
// Penalty evaluates the supplied geometry at t_n; multipliers enforce it at t_(n+1).
// Unconverged steps preserve current and previous displacements.
ContactResult Step(const Model &, State &, ContactMethod, double offset, double separation, double tolerance = 1e-11, unsigned iterations = 2000);
double Energy(const Model &, const State &);
}
