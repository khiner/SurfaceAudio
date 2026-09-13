#pragma once
#include <vector>

namespace surface_audio {
enum class BeamBoundary { SimplySupported,
                          Free };
struct BeamProperties {
    double Length{}, MassPerLength{}, BendingStiffness{}, DampingRatio{};
    BeamBoundary Boundary{BeamBoundary::SimplySupported};
};
struct BeamMode {
    double WaveNumber{}, Omega{}, Scale{};
    unsigned Kind{};
};

// Shapes have unit spatial L2 norm; free beams include translation and rotation before their bending modes.
std::vector<BeamMode> MakeBeamModes(BeamProperties, unsigned bending_modes);
double BeamShape(BeamProperties, const BeamMode &, double x, unsigned derivative = 0);

struct PlateProperties {
    double Length{}, Width{}, MassPerArea{}, BendingStiffness{}, DampingRatio{};
};
struct PlateMode {
    unsigned X{}, Y{};
    double Omega{};
};

// Returns the lowest-frequency simply supported Kirchhoff plate modes with unit spatial L2 norm.
std::vector<PlateMode> MakePlateModes(PlateProperties, unsigned modes);
double PlateShape(PlateProperties, PlateMode, double x, double y);

struct ModalDynamics {
    double Mass{}, Omega{}, DampingRatio{}, Stiffness{}, Retention{}, Compliance{};
    bool operator==(const ModalDynamics &) const = default;
};
ModalDynamics MakeModalDynamics(double mass, double omega, double damping_ratio, double dt);
double AdvanceMode(ModalDynamics, double displacement, double previous, double force);
double PreviousMode(ModalDynamics, double displacement, double velocity, double force, double dt);
// Half-step energy satisfies an exact discrete work balance for the central-difference recurrence.
double ModalEnergy(ModalDynamics, double displacement, double previous, double dt);
}
