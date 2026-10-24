#pragma once

#include <stdexcept>

inline void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
