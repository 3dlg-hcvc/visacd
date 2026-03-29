#pragma once

#include <core.hpp>

namespace neural_acd {

void manifold_preprocess(Mesh &m, double scale = 50.0f,
                         double level_set = 0.55f);

} // namespace neural_acd