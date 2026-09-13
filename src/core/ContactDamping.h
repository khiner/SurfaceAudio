#pragma once
#include <span>
#include <vector>

namespace surface_audio {
struct NormalContactLaw {
    double Stiffness{}, Damping{}, Exponent{1.5}, DampingExponent{1.5};
};
struct ContactDampingState {
    std::vector<double> Velocity, Force;
    std::vector<double> ScaledJacobian, Damping, Offset, Scale, Value, Trial, Gradient, Hessian, Direction, WeightedJacobian, ContactValue;
    std::vector<size_t> ActiveRows;
};
struct ContactDampingResult {
    double Residual{};
    unsigned Iterations{};
    bool Converged{};
};
// Solves v = free_velocity + mobility * J^T * force, with penetration rate -J*v and nonnegative contact force.
// J is row-major; mobility is positive and diagonal; geometry and penetration remain fixed during the solve.
ContactDampingResult SolveContactDamping(std::span<const double> jacobian, std::span<const double> penetration, std::span<const double> area, std::span<const double> free_velocity, std::span<const double> mobility, NormalContactLaw, ContactDampingState &, double tolerance = 1e-10, unsigned iterations = 50);
}
