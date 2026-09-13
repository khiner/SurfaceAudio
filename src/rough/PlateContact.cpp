#include "PlateContact.h"
#include "core/NormalContact.h"
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace surface_audio::rough {
namespace {
constexpr double Pi = std::numbers::pi;
unsigned Modes(const PlateContactModel &m) { return unsigned(m.Bottom.Modes.size() + m.Top.Modes.size()); }
const ModalDynamics &Mode(const PlateContactModel &m, unsigned k) { return k < m.Bottom.Modes.size() ? m.Bottom.Modes[k] : m.Top.Modes[k - m.Bottom.Modes.size()]; }
double Gravity(const PlateContactModel &m, unsigned k) { return k < m.Bottom.Modes.size() ? m.Bottom.Gravity[k] : m.Top.Gravity[k - m.Bottom.Modes.size()]; }
double Dot(std::span<const double> a, std::span<const double> b) { return cblas_ddot(int(a.size()), a.data(), 1, b.data(), 1); }
void ValidateGrid(PlateGrid g, double length, double width, std::span<const double> heights, double gravity) {
    if (g.XNodes < 2 || g.YNodes < 2 || uint64_t(g.XNodes) * g.YNodes > UINT32_MAX || heights.size() != size_t(g.XNodes) * g.YNodes ||
        !std::isfinite(g.StepX) || g.StepX <= 0 || !std::isfinite(g.StepY) || g.StepY <= 0 || !std::isfinite(g.OriginX) || !std::isfinite(g.OriginY) ||
        g.OriginX < 0 || g.OriginY < 0 || g.OriginX + (g.XNodes - 1) * g.StepX > length * (1 + 1e-12) ||
        g.OriginY + (g.YNodes - 1) * g.StepY > width * (1 + 1e-12) || !std::isfinite(gravity)) throw std::invalid_argument("Invalid plate surface grid");
    for (double h : heights)
        if (!std::isfinite(h)) throw std::invalid_argument("Nonfinite plate surface height");
}
void ValidateState(const PlateContactModel &m, const PlateContactState &s) {
    const unsigned count = Modes(m);
    for (const auto *v : {&s.Displacement, &s.Previous, &s.Next, &s.Force, &s.Velocity, &s.FreeVelocity, &s.Mobility, &s.Row})
        if (v->size() != count) throw std::invalid_argument("Invalid plate contact state dimensions");
}
void ResetContacts(const PlateContactModel &m, PlateContactState &s, double offset_x, double offset_y, double separation) {
    ValidateState(m, s);
    if (!std::isfinite(offset_x) || !std::isfinite(offset_y) || !std::isfinite(separation)) throw std::invalid_argument("Nonfinite plate contact geometry");
    s.Jacobian.clear();
    s.Penetration.clear();
    s.Area.clear();
}
double Shape(const PlateSurface &s, unsigned mode, unsigned x, unsigned y) { return s.XShapes[mode * s.Grid.XNodes + x] * s.YShapes[mode * s.Grid.YNodes + y]; }
std::array<unsigned, 2> Window(unsigned count, double origin, double step, double low, double high) {
    const double tolerance = 0x1p-43 * count;
    return {unsigned(std::clamp(std::ceil((low - origin) / step - tolerance), 0., double(count))), unsigned(std::clamp(std::floor((high - origin) / step + tolerance) + 1, 0., double(count)))};
}
void AppendContact(const PlateContactModel &m, PlateContactState &s, bool top, unsigned x, unsigned y, double offset_x, double offset_y, double separation, double displacement_bound) {
    const auto &slave = top ? m.Top : m.Bottom, &master = top ? m.Bottom : m.Top;
    const auto &a = slave.Grid, &b = master.Grid;
    const double dx = top ? offset_x : -offset_x, dy = top ? offset_y : -offset_y;
    const double cx = std::clamp((a.OriginX + x * a.StepX + dx - b.OriginX) / b.StepX, 0., double(b.XNodes - 1));
    const double cy = std::clamp((a.OriginY + y * a.StepY + dy - b.OriginY) / b.StepY, 0., double(b.YNodes - 1));
    const unsigned ix = std::min(unsigned(cx), b.XNodes - 2), iy = std::min(unsigned(cy), b.YNodes - 2);
    const double tx = cx - ix, ty = cy - iy;
    const std::array weights{(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty};
    const std::array nodes{iy * b.XNodes + ix, iy * b.XNodes + ix + 1, (iy + 1) * b.XNodes + ix, (iy + 1) * b.XNodes + ix + 1};
    double height = slave.Height[y * a.XNodes + x];
    for (unsigned i = 0; i < 4; ++i) height += weights[i] * master.Height[nodes[i]];
    if (height - separation + displacement_bound < 0) return;
    for (unsigned k = 0; k < slave.Modes.size(); ++k) s.Row[(top ? m.Bottom.Modes.size() : 0) + k] = (top ? 1 : -1) * Shape(slave, k, x, y);
    for (unsigned k = 0; k < master.Modes.size(); ++k) {
        double value{};
        for (unsigned i = 0; i < 4; ++i) value += weights[i] * Shape(master, k, nodes[i] % b.XNodes, nodes[i] / b.XNodes);
        s.Row[(top ? 0 : m.Bottom.Modes.size()) + k] = (top ? -1 : 1) * value;
    }
    const double penetration = height - separation - Dot(s.Row, s.Displacement);
    if (penetration <= 0) return;
    const double area = a.StepX * a.StepY * (x == 0 || x + 1 == a.XNodes ? .5 : 1) * (y == 0 || y + 1 == a.YNodes ? .5 : 1);
    s.Penetration.push_back(penetration);
    s.Area.push_back(area);
    s.Jacobian.insert(s.Jacobian.end(), s.Row.begin(), s.Row.end());
}

}
PlateSurface MakePlateSurface(PlateProperties p, unsigned count, PlateGrid grid, std::span<const double> heights, double dt, double gravity) {
    const auto modes = MakePlateModes(p, count);
    ValidateGrid(grid, p.Length, p.Width, heights, gravity);
    PlateSurface s{grid, {heights.begin(), heights.end()}, std::vector<double>(size_t(count) * grid.XNodes), std::vector<double>(size_t(count) * grid.YNodes), std::vector<double>(count), {}};
    s.Modes.reserve(count);
    for (unsigned k = 0; k < count; ++k) {
        const auto mode = modes[k];
        s.Modes.push_back(MakeModalDynamics(p.MassPerArea, mode.Omega, p.DampingRatio, dt));
        for (unsigned x = 0; x < grid.XNodes; ++x) s.XShapes[k * grid.XNodes + x] = std::sqrt(2 / p.Length) * std::sin(mode.X * Pi * (grid.OriginX + x * grid.StepX) / p.Length);
        for (unsigned y = 0; y < grid.YNodes; ++y) s.YShapes[k * grid.YNodes + y] = std::sqrt(2 / p.Width) * std::sin(mode.Y * Pi * (grid.OriginY + y * grid.StepY) / p.Width);
        s.Gravity[k] = mode.X % 2 && mode.Y % 2 ? p.MassPerArea * gravity * 8 * std::sqrt(p.Length * p.Width) / (mode.X * mode.Y * Pi * Pi) : 0;
    }
    return s;
}
PlateSurface MakeRigidPlateSurface(RigidPlateProperties p, PlateGrid grid, std::span<const double> heights, double dt, double gravity) {
    for (double value : {p.Length, p.Width, p.Mass, p.InertiaX, p.InertiaY})
        if (!std::isfinite(value) || value <= 0) throw std::invalid_argument("Invalid rigid plate properties");
    ValidateGrid(grid, p.Length, p.Width, heights, gravity);
    const double area = p.Length * p.Width, translation = p.Mass / area, rotation_x = 12 * p.InertiaX / (area * p.Width * p.Width), rotation_y = 12 * p.InertiaY / (area * p.Length * p.Length);
    PlateSurface s{grid, {heights.begin(), heights.end()}, std::vector<double>(3 * grid.XNodes), std::vector<double>(3 * grid.YNodes), {p.Mass * gravity / std::sqrt(area), 0, 0}, {MakeModalDynamics(translation, 0, 0, dt), MakeModalDynamics(rotation_x, 0, 0, dt), MakeModalDynamics(rotation_y, 0, 0, dt)}};
    for (unsigned x = 0; x < grid.XNodes; ++x) {
        s.XShapes[x] = s.XShapes[grid.XNodes + x] = 1 / std::sqrt(p.Length);
        s.XShapes[2 * grid.XNodes + x] = -std::sqrt(12 / (p.Length * p.Length * p.Length)) * (grid.OriginX + x * grid.StepX - p.Length / 2);
    }
    for (unsigned y = 0; y < grid.YNodes; ++y) {
        s.YShapes[y] = s.YShapes[2 * grid.YNodes + y] = 1 / std::sqrt(p.Width);
        s.YShapes[grid.YNodes + y] = std::sqrt(12 / (p.Width * p.Width * p.Width)) * (grid.OriginY + y * grid.StepY - p.Width / 2);
    }
    return s;
}
PlateContactModel MakePlateContact(PlateSurface bottom, PlateSurface top, double dt, NormalContactLaw law, bool two_pass) {
    if (!std::isfinite(dt) || dt <= 0 || bottom.Modes.size() + top.Modes.size() == 0 || bottom.Modes.size() + top.Modes.size() > INT_MAX)
        throw std::invalid_argument("Invalid plate integration dimensions");
    for (const auto *surface : {&bottom, &top}) {
        const auto &g = surface->Grid;
        if (g.XNodes < 2 || g.YNodes < 2 || surface->Height.size() != size_t(g.XNodes) * g.YNodes || surface->Gravity.size() != surface->Modes.size() ||
            surface->XShapes.size() != surface->Modes.size() * g.XNodes || surface->YShapes.size() != surface->Modes.size() * g.YNodes) throw std::invalid_argument("Invalid plate surface dimensions");
        for (const auto &mode : surface->Modes)
            if (mode != MakeModalDynamics(mode.Mass, mode.Omega, mode.DampingRatio, dt)) throw std::invalid_argument("Plate time steps differ");
    }
    return {std::move(bottom), std::move(top), law, dt, two_pass};
}
PlateContactState MakePlateState(const PlateContactModel &m, std::span<const double> displacement, std::span<const double> velocity) {
    const unsigned count = Modes(m);
    if ((!displacement.empty() && displacement.size() != count) || (!velocity.empty() && velocity.size() != count)) throw std::invalid_argument("Invalid initial plate state");
    PlateContactState s{.Displacement = std::vector<double>(count), .Previous = std::vector<double>(count), .Next = std::vector<double>(count), .Force = std::vector<double>(count), .Velocity = std::vector<double>(count), .FreeVelocity = std::vector<double>(count), .Mobility = std::vector<double>(count), .Row = std::vector<double>(count), .Jacobian = {}, .Penetration = {}, .Area = {}, .Damping = {}};
    for (unsigned k = 0; k < count; ++k) {
        const double q = displacement.empty() ? 0 : displacement[k], v = velocity.empty() ? 0 : velocity[k];
        if (!std::isfinite(q) || !std::isfinite(v)) throw std::invalid_argument("Nonfinite initial plate state");
        s.Displacement[k] = q;
        s.Previous[k] = PreviousMode(Mode(m, k), q, v, Gravity(m, k), m.TimeStep);
        s.Mobility[k] = Mode(m, k).Compliance / (2 * m.TimeStep);
    }
    return s;
}
std::array<unsigned, 4> PlateOverlap(PlateGrid a, PlateGrid b, double dx, double dy) {
    const auto x = Window(a.XNodes, a.OriginX + dx, a.StepX, b.OriginX, b.OriginX + (b.XNodes - 1) * b.StepX);
    const auto y = Window(a.YNodes, a.OriginY + dy, a.StepY, b.OriginY, b.OriginY + (b.YNodes - 1) * b.StepY);
    return {x[0], x[1], y[0], y[1]};
}
void PreparePlateContact(const PlateContactModel &m, PlateContactState &s, double offset_x, double offset_y, double separation) {
    ResetContacts(m, s, offset_x, offset_y, separation);
    for (unsigned pass = 0; pass < (m.TwoPass ? 2u : 1u); ++pass) {
        const bool top = pass == 0;
        const auto &slave = top ? m.Top : m.Bottom, &master = top ? m.Bottom : m.Top;
        const auto &a = slave.Grid, &b = master.Grid;
        const double dx = top ? offset_x : -offset_x, dy = top ? offset_y : -offset_y;
        const auto window = PlateOverlap(a, b, dx, dy);
        for (unsigned y = window[2]; y < window[3]; ++y)
            for (unsigned x = window[0]; x < window[1]; ++x) AppendContact(m, s, top, x, y, offset_x, offset_y, separation, INFINITY);
    }
}
void PreparePlateContactNodes(const PlateContactModel &m, PlateContactState &s, std::span<const unsigned> nodes, double offset_x, double offset_y, double separation, double displacement_bound) {
    if (std::isnan(displacement_bound)) throw std::invalid_argument("Nonfinite plate contact geometry");
    ResetContacts(m, s, offset_x, offset_y, separation);
    const size_t top_nodes = m.Top.Height.size();
    const auto top_window = PlateOverlap(m.Top.Grid, m.Bottom.Grid, offset_x, offset_y);
    const auto bottom_window = PlateOverlap(m.Bottom.Grid, m.Top.Grid, -offset_x, -offset_y);
    for (size_t i = 0; i < nodes.size(); ++i) {
        if ((i && nodes[i] <= nodes[i - 1]) || nodes[i] >= top_nodes + (m.TwoPass ? m.Bottom.Height.size() : 0)) throw std::invalid_argument("Invalid ordered plate candidate nodes");
        const bool top = nodes[i] < top_nodes;
        const auto &grid = top ? m.Top.Grid : m.Bottom.Grid;
        const auto &window = top ? top_window : bottom_window;
        const unsigned node = top ? nodes[i] : unsigned(nodes[i] - top_nodes), x = node % grid.XNodes, y = node / grid.XNodes;
        if (x >= window[0] && x < window[1] && y >= window[2] && y < window[3]) AppendContact(m, s, top, x, y, offset_x, offset_y, separation, displacement_bound);
    }
}
PlateContactResult StepPlateContact(const PlateContactModel &m, PlateContactState &s, double offset_x, double offset_y, double separation, double tolerance, unsigned iterations) {
    PreparePlateContact(m, s, offset_x, offset_y, separation);
    return AdvancePlateContact(m, s, tolerance, iterations);
}
PlateContactResult AdvancePlateContact(const PlateContactModel &m, PlateContactState &s, double tolerance, unsigned iterations) {
    ValidateState(m, s);
    const unsigned modes = Modes(m);
    for (unsigned k = 0; k < modes; ++k) {
        const double free = AdvanceMode(Mode(m, k), s.Displacement[k], s.Previous[k], Gravity(m, k));
        s.FreeVelocity[k] = (free - s.Previous[k]) / (2 * m.TimeStep);
    }
    const auto solve = SolveContactDamping(s.Jacobian, s.Penetration, s.Area, s.FreeVelocity, s.Mobility, m.Law, s.Damping, tolerance, iterations);
    PlateContactResult result{.Solve = solve};
    if (!solve.Converged) return result;
    for (unsigned k = 0; k < modes; ++k) s.Force[k] = Gravity(m, k);
    for (size_t row = 0; row < s.Penetration.size(); ++row) {
        const auto j = std::span(s.Jacobian).subspan(row * modes, modes);
        const auto law = m.Law;
        const auto contact = EvaluateNormalContact(s.Penetration[row], -Dot(j, s.Damping.Velocity), law.Stiffness, law.Damping, law.Exponent, law.DampingExponent);
        cblas_daxpy(int(modes), s.Damping.Force[row], j.data(), 1, s.Force.data(), 1);
        result.NormalForce += s.Damping.Force[row];
        result.Contacts += s.Damping.Force[row] > 0;
        result.MaxPenetration = std::max(result.MaxPenetration, s.Penetration[row]);
        result.ElasticEnergy += s.Area[row] * contact.Energy;
        result.Dissipation += s.Area[row] * contact.Dissipation;
    }
    for (unsigned k = 0; k < modes; ++k) {
        s.Next[k] = AdvanceMode(Mode(m, k), s.Displacement[k], s.Previous[k], s.Force[k]);
        s.Velocity[k] = (s.Next[k] - s.Previous[k]) / (2 * m.TimeStep);
        if (!std::isfinite(s.Next[k])) throw std::runtime_error("Plate contact integration became nonfinite");
    }
    s.Previous.swap(s.Displacement);
    s.Displacement.swap(s.Next);
    return result;
}
double PlateEnergy(const PlateContactModel &m, const PlateContactState &s) {
    ValidateState(m, s);
    double result{};
    for (unsigned k = 0; k < Modes(m); ++k) result += ModalEnergy(Mode(m, k), s.Displacement[k], s.Previous[k], m.TimeStep);
    return result;
}
}
