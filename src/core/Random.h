#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define SURFACE_THREAD thread
#else
#include <cmath>
#include <cstdint>
#define SURFACE_THREAD
#endif

namespace surface_audio {
#ifdef __METAL_VERSION__
using RandomWord = uint;
using RandomWide = ulong;
#else
using RandomWord = uint32_t;
using RandomWide = uint64_t;
#endif

struct RandomState {
    RandomWide State{0}, Sequence{1};
};

inline RandomWord NextRandom(SURFACE_THREAD RandomState &r) {
    // PCG-XSH-RR state advances independently of dispatch boundaries.
    const auto previous = r.State;
    r.State = previous * RandomWide(6364136223846793005ULL) + r.Sequence;
    const auto bits = RandomWord(((previous >> 18u) ^ previous) >> 27u);
    const auto rotation = RandomWord(previous >> 59u);
    return (bits >> rotation) | (bits << ((-rotation) & 31u));
}

inline RandomState MakeRandom(RandomWide seed, RandomWide stream = 0) {
    RandomState r{0, (stream << 1u) | 1u};
    NextRandom(r);
    r.State += seed;
    NextRandom(r);
    return r;
}

inline RandomWide MixRandom(RandomWide x) {
    x = (x ^ (x >> 30)) * RandomWide(0xbf58476d1ce4e5b9ULL);
    x = (x ^ (x >> 27)) * RandomWide(0x94d049bb133111ebULL);
    return x ^ (x >> 31);
}

// Mix independently sampled indices to avoid correlations in adjacent PCG stream prefixes.
inline RandomState MakeIndexedRandom(RandomWide seed, RandomWide index) {
    return MakeRandom(MixRandom(seed + index * RandomWide(0x9e3779b97f4a7c15ULL)));
}

// Open interval (0,1), including after float rounding, for logarithms and inverse CDFs.
inline float Uniform(SURFACE_THREAD RandomState &r) { return (float(NextRandom(r) >> 9u) + .5f) * (1.f / 8388608.f); }

inline float Normal(SURFACE_THREAD RandomState &r) {
    const float u = Uniform(r), v = Uniform(r);
#ifdef __METAL_VERSION__
    return metal::sqrt(-2.f * metal::log(u)) * metal::cos(6.283185307179586f * v);
#else
    return std::sqrt(-2.f * std::log(u)) * std::cos(6.283185307179586f * v);
#endif
}
}

#undef SURFACE_THREAD
