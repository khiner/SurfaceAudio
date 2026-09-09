#include "Gpu.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <exception>
#include <vector>

namespace surface_audio {
struct GpuState {
    struct Pipeline {
        std::string Name;
        id<MTLComputePipelineState> State;
    };
    id<MTLDevice> Device;
    id<MTLLibrary> Library;
    id<MTL4Compiler> Compiler;
    id<MTL4CommandQueue> Queue;
    id<MTLResidencySet> Residency;
    id<MTLSharedEvent> Complete;
    id<MTL4CommandAllocator> Allocator;
    id<MTL4CommandBuffer> Commands;
    std::array<id<MTL4ArgumentTable>, 32> Tables;
    std::vector<id<MTLBuffer>> Buffers;
    std::vector<Pipeline> Pipelines;
    uint64_t Submitted{0};
    uint32_t Encoded{0};
    bool Recording{false};

    ~GpuState() {
        // A stalled device must neither deadlock teardown nor free in-flight allocations.
        if (Submitted && ![Complete waitUntilSignaledValue:Submitted timeoutMS:30000]) std::terminate();
        [Residency endResidency];
    }
};

namespace {
std::runtime_error Error(std::string_view operation, NSError *error) { return std::runtime_error(std::string(operation) + ": " + (error ? error.localizedDescription.UTF8String : "Metal returned nil")); }
NSString *String(std::string_view text) { return [[NSString alloc] initWithBytes:text.data() length:text.size() encoding:NSUTF8StringEncoding]; }

void Encode(Gpu &gpu, GpuKernel kernel, std::span<const GpuBinding> bindings, GpuGrid grid, GpuGrid group, bool groups) {
    auto &s = *gpu.State;
    if (!s.Recording || s.Encoded == s.Tables.size()) throw std::logic_error("GPU batch not begun or dispatch capacity exceeded");
    if (kernel.Index >= s.Pipelines.size() || !group.X || !group.Y || !group.Z || uint64_t(group.X) * group.Y * group.Z > kernel.MaxThreads) throw std::invalid_argument("Invalid GPU dispatch geometry");
    if (!grid.X || !grid.Y || !grid.Z) return;
    @autoreleasepool {
        auto table = s.Tables[s.Encoded++];
        for (const auto &binding : bindings) {
            if (binding.Index >= 16 || binding.Offset >= binding.Buffer.Size) throw std::invalid_argument("Invalid GPU buffer binding");
            [table setAddress:binding.Buffer.Address + binding.Offset atIndex:binding.Index];
        }
        auto encoder = [s.Commands computeCommandEncoder];
        [encoder barrierAfterQueueStages:MTLStageDispatch beforeStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        [encoder setArgumentTable:table];
        [encoder setComputePipelineState:s.Pipelines[kernel.Index].State];
        const auto grid_size = MTLSizeMake(grid.X, grid.Y, grid.Z), group_size = MTLSizeMake(group.X, group.Y, group.Z);
        if (groups) [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:group_size];
        else [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
        [encoder endEncoding];
    }
}
} // namespace

Gpu CreateGpu(std::string_view library_path) {
    @autoreleasepool {
        auto s = std::make_shared<GpuState>();
        s->Device = MTLCreateSystemDefaultDevice();
        if (!s->Device || ![s->Device supportsFamily:MTLGPUFamilyMetal4] || !s->Device.hasUnifiedMemory) throw std::runtime_error("SurfaceAudio requires Apple Silicon with Metal 4");
        NSError *error = nil;
        s->Library = [s->Device newLibraryWithURL:[NSURL fileURLWithPath:String(library_path.empty() ? SURFACE_AUDIO_METALLIB : library_path)] error:&error];
        if (!s->Library) throw Error("Load metallib", error);
        s->Compiler = [s->Device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
        if (!s->Compiler) throw Error("Create Metal 4 compiler", error);
        s->Queue = [s->Device newMTL4CommandQueueWithDescriptor:[MTL4CommandQueueDescriptor new] error:&error];
        if (!s->Queue) throw Error("Create Metal 4 queue", error);
        s->Residency = [s->Device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
        if (!s->Residency) throw Error("Create residency set", error);
        [s->Queue addResidencySet:s->Residency];
        s->Complete = [s->Device newSharedEvent];
        s->Allocator = [s->Device newCommandAllocator];
        s->Commands = [s->Device newCommandBuffer];
        if (!s->Complete || !s->Allocator || !s->Commands) throw Error("Create command resources", nil);
        auto descriptor = [MTL4ArgumentTableDescriptor new];
        descriptor.maxBufferBindCount = 16;
        for (auto &table : s->Tables) {
            table = [s->Device newArgumentTableWithDescriptor:descriptor error:&error];
            if (!table) throw Error("Create argument table", error);
        }
        return {std::move(s)};
    }
}

std::string DeviceName(const Gpu &gpu) { return gpu.State->Device.name.UTF8String; }

GpuBuffer CreateBuffer(Gpu &gpu, size_t bytes) {
    auto &s = *gpu.State;
    if (!bytes || s.Recording) throw std::invalid_argument("GPU allocation requires nonzero size outside a batch");
    WaitGpu(gpu);
    auto buffer = [s.Device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!buffer) throw std::runtime_error("GPU buffer allocation failed");
    s.Buffers.push_back(buffer);
    [s.Residency addAllocation:buffer];
    [s.Residency commit];
    [s.Residency requestResidency];
    return {buffer.contents, bytes, buffer.gpuAddress};
}

GpuKernel CreateKernel(Gpu &gpu, std::string_view name) {
    @autoreleasepool {
        auto &s = *gpu.State;
        const auto found = std::ranges::find(s.Pipelines, name, &GpuState::Pipeline::Name);
        if (found != s.Pipelines.end()) return {uint32_t(found - s.Pipelines.begin()), uint32_t(found->State.maxTotalThreadsPerThreadgroup)};
        auto function = [MTL4LibraryFunctionDescriptor new];
        function.library = s.Library;
        function.name = String(name);
        auto descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        NSError *error = nil;
        auto pipeline = [s.Compiler newComputePipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
        if (!pipeline) throw Error(name, error);
        const auto index = uint32_t(s.Pipelines.size());
        s.Pipelines.push_back({std::string(name), pipeline});
        return {index, uint32_t(pipeline.maxTotalThreadsPerThreadgroup)};
    }
}

void WaitGpu(Gpu &gpu) {
    auto &s = *gpu.State;
    if (s.Submitted && ![s.Complete waitUntilSignaledValue:s.Submitted timeoutMS:30000]) throw std::runtime_error("GPU completion timeout");
}

void BeginGpu(Gpu &gpu) {
    auto &s = *gpu.State;
    if (s.Recording) throw std::logic_error("GPU batch already begun");
    WaitGpu(gpu);
    [s.Allocator reset];
    [s.Commands beginCommandBufferWithAllocator:s.Allocator];
    s.Encoded = 0;
    s.Recording = true;
}

void DispatchGpu(Gpu &gpu, GpuKernel kernel, std::span<const GpuBinding> bindings, GpuGrid threads, GpuGrid group) { Encode(gpu, kernel, bindings, threads, group, false); }
void DispatchGroupsGpu(Gpu &gpu, GpuKernel kernel, std::span<const GpuBinding> bindings, GpuGrid groups, GpuGrid group) { Encode(gpu, kernel, bindings, groups, group, true); }

uint64_t SubmitGpu(Gpu &gpu) {
    auto &s = *gpu.State;
    if (!s.Recording) throw std::logic_error("GPU batch not begun");
    [s.Commands endCommandBuffer];
    id<MTL4CommandBuffer> commands[]{s.Commands};
    [s.Queue commit:commands count:1];
    // Queue signalling publishes all GPU writes to shared CPU memory before reuse.
    [s.Queue signalEvent:s.Complete value:++s.Submitted];
    s.Recording = false;
    return s.Submitted;
}
} // namespace surface_audio
