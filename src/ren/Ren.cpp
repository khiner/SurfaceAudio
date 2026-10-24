#include "Ren.h"

#include <array>
#include <vector>

namespace surface_audio::ren {
void Validate(const Parameters &p, std::span<const Normal> normals, std::span<const float> roughness) {
    if (!std::isfinite(p.SampleRate) || p.SampleRate <= 0 || !std::isfinite(p.Mass) || p.Mass <= 0 || !std::isfinite(p.RoughnessGain) || p.RoughnessGain < 0 || !std::isfinite(p.RoughnessSpacing) || p.RoughnessSpacing <= 0 || !p.Width || !p.Height || uint64_t(p.Width) * p.Height > UINT32_MAX || normals.size() != uint64_t(p.Width) * p.Height || p.TableSize < 2 || p.TableSize > (1u << 20) || roughness.size() != p.TableSize) throw std::invalid_argument("Invalid Ren surface or wavetable parameters");
    for (const auto &n : normals)
        if (!std::isfinite(n.X) || !std::isfinite(n.Y) || !std::isfinite(n.Z) || n.Z <= 0 || !std::isfinite(n.X / n.Z) || !std::isfinite(n.Y / n.Z)) throw std::invalid_argument("Ren normals must face the positive surface normal");
    for (float x : roughness)
        if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite Ren roughness sample");
}
void ValidateContacts(std::span<const Contact> contacts) {
    for (const auto &c : contacts) {
        for (float x : {c.U, c.V, c.VelocityX, c.VelocityY, c.NormalVelocity, c.NormalForce, c.MacroImpulse})
            if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite Ren contact");
        if (c.NormalForce < 0 || c.Overlap > 1 || !std::isfinite(c.VelocityX * c.VelocityX + c.VelocityY * c.VelocityY)) throw std::invalid_argument("Invalid Ren contact load or velocity");
    }
}
void Render(const Parameters &p, State &s, std::span<const Normal> normals, std::span<const float> roughness, std::span<const Contact> contacts, std::span<Impulses> output) {
    Validate(p, normals, roughness);
    ValidateContacts(contacts);
    if (contacts.size() != output.size() || !std::isfinite(s.Momentum) || !std::isfinite(s.Phase) || !std::isfinite(s.PhaseError) || s.Phase < 0 || s.Phase >= p.TableSize || s.Lasting > 1) throw std::invalid_argument("Invalid Ren state or output size");
    for (size_t i = 0; i < contacts.size(); ++i) output[i] = Step(p, s, contacts[i], normals.data(), roughness.data());
}
GpuSource CreateGpuSource(Gpu &gpu, const Parameters &p, std::span<const Normal> normals, std::span<const float> roughness, unsigned voices, unsigned frames) {
    Validate(p, normals, roughness);
    if (!voices || !frames || uint64_t(voices) * frames > UINT32_MAX) throw std::invalid_argument("Invalid Ren GPU dimensions");
    const auto count = size_t(voices) * frames;
    const std::vector<State> states(voices);
    return {voices, frames, Upload(gpu, p), Upload<State>(gpu, states), Upload<Normal>(gpu, normals), Upload<float>(gpu, roughness), CreateBuffer(gpu, count * sizeof(Contact)), CreateBuffer(gpu, count * sizeof(Impulses)), CreateKernel(gpu, "RenSource")};
}
void EncodeSource(Gpu &gpu, const GpuSource &s) {
    const auto block = BatchUpload(gpu, std::array{s.Voices, s.Frames});
    const std::array bindings{GpuBinding{s.Config, 0}, GpuBinding{s.States, 1}, GpuBinding{s.Normals, 2}, GpuBinding{s.Roughness, 3}, GpuBinding{s.Contacts, 4}, GpuBinding{s.Output, 5}, GpuBinding{block, 6}};
    DispatchGpu(gpu, s.Kernel, bindings, {s.Voices});
}
}
