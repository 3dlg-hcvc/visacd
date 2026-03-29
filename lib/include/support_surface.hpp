#pragma once

#include "core.hpp"
#include <unordered_map>
#include <vector>

namespace neural_acd {

struct Surface {
    std::vector<int> triangle_ids;
    double area = 0.0;
    Plane plane;
};

std::vector<Surface> extract_surfaces(const Mesh &mesh, double min_area);

}