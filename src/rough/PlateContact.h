#pragma once
#include "core/ContactDamping.h"
#include "core/Flexural.h"
#include <array>

namespace surface_audio::rough {
struct PlateGrid {
    unsigned XNodes{}, YNodes{};
    double StepX{}, StepY{}, OriginX{}, OriginY{};
};
struct RigidPlateProperties {
    double Length{}, Width{}, Mass{}, InertiaX{}, InertiaY{};
};
struct PlateSurface {
    PlateGrid Grid;
    std::vector<double> Height, XShapes, YShapes, Gravity;
    std::vector<ModalDynamics> Modes;
};
struct PlateContactModel {
    PlateSurface Bottom, Top;
    NormalContactLaw Law;
    double TimeStep{};
    bool TwoPass{};
};
struct PlateContactState {
    std::vector<double> Displacement, Previous, Next, Force, Velocity, FreeVelocity, Mobility, Row;
    std::vector<double> Jacobian, Penetration, Area;
    ContactDampingState Damping;
};
struct PlateContactResult {
    double NormalForce{}, MaxPenetration{}, ElasticEnergy{}, Dissipation{};
    unsigned Contacts{};
    ContactDampingResult Solve;
};
// Surface grids may cover a patch within the body; storage has contiguous x coordinates.
PlateSurface MakePlateSurface(PlateProperties, unsigned modes, PlateGrid, std::span<const double> heights, double dt, double gravity = 0);
PlateSurface MakeRigidPlateSurface(RigidPlateProperties, PlateGrid, std::span<const double> heights, double dt, double gravity = 0);
PlateContactModel MakePlateContact(PlateSurface bottom, PlateSurface top, double dt, NormalContactLaw, bool two_pass = false);
// Initial acceleration includes gravity; loaded equilibria require an explicit Previous state.
PlateContactState MakePlateState(const PlateContactModel &, std::span<const double> displacement = {}, std::span<const double> velocity = {});
// Populates penetrating contact rows at the supplied geometry without advancing time.
void PreparePlateContact(const PlateContactModel &, PlateContactState &, double offset_x, double offset_y, double separation);
// Evaluates sorted unique material-node IDs, with top nodes preceding bottom nodes; include every possible contact.
// displacement_bound must bound the maximum reduction of separation from both bodies' modal displacement.
void PreparePlateContactNodes(const PlateContactModel &, PlateContactState &, std::span<const unsigned> nodes, double offset_x, double offset_y, double separation, double displacement_bound);
// Returns half-open x and y index ranges in the slave grid.
std::array<unsigned, 4> PlateOverlap(PlateGrid slave, PlateGrid master, double offset_x, double offset_y);
// Advances using the state's prepared contact rows, allowing CPU or GPU contact detection.
PlateContactResult AdvancePlateContact(const PlateContactModel &, PlateContactState &, double tolerance = 1e-10, unsigned iterations = 50);
// Returns forces and centered velocities at t_n; an unconverged solve preserves Displacement and Previous.
PlateContactResult StepPlateContact(const PlateContactModel &, PlateContactState &, double offset_x, double offset_y, double separation, double tolerance = 1e-10, unsigned iterations = 50);
double PlateEnergy(const PlateContactModel &, const PlateContactState &);
}
