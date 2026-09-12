#pragma once

namespace surface_audio::hatt {
enum : unsigned { MaximumOrder = 25,
                  PolynomialSize = MaximumOrder + 1,
                  ModelStride = 2 * PolynomialSize,
                  FilterStride = ModelStride + 1 };
}
