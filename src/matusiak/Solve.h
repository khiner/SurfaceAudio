// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#ifdef __METAL_VERSION__
#define MAT_THREAD thread
#else
#include <cmath>
#define MAT_THREAD
#endif
namespace surface_audio::matusiak {
template<typename T> bool Solve(MAT_THREAD T *a, MAT_THREAD T *b, unsigned n, T minimum_pivot) {
#ifndef __METAL_VERSION__
    using std::abs;
#endif
    for (unsigned j = 0; j < n; ++j) {
        unsigned pivot = j;
        for (unsigned i = j + 1; i < n; ++i)
            if (abs(a[i * n + j]) > abs(a[pivot * n + j])) pivot = i;
        if (abs(a[pivot * n + j]) < minimum_pivot) return false;
        for (unsigned k = j; k < n; ++k) {
            const T value = a[j * n + k];
            a[j * n + k] = a[pivot * n + k];
            a[pivot * n + k] = value;
        }
        const T value = b[j];
        b[j] = b[pivot];
        b[pivot] = value;
        for (unsigned i = j + 1; i < n; ++i) {
            const T q = a[i * n + j] / a[j * n + j];
            for (unsigned k = j + 1; k < n; ++k) a[i * n + k] -= q * a[j * n + k];
            b[i] -= q * b[j];
        }
    }
    for (unsigned i = n; i-- > 0;) {
        for (unsigned j = i + 1; j < n; ++j) b[i] -= a[i * n + j] * b[j];
        b[i] /= a[i * n + i];
    }
    return true;
}
}
#undef MAT_THREAD
