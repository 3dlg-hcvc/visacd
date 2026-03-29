#pragma once

#include <core.hpp>
#include <optixUtils.hpp>
#include <vector>

namespace neural_acd {

struct RayGenData {
  const float *points;
  const unsigned int *new_mask;
  long long n_points;
  unsigned int has_mask;
  unsigned int *uM;
  OptixTraversableHandle gas;
};
struct MissData {};
struct HitgroupData {
  const float *vertices;
  const uint3 *indices;
};

std::vector<std::pair<unsigned int, unsigned int>>
compute_intersection_matrix(Mesh &mesh, Mesh &cage,
                            OptixDeviceContext &context);
// Returns upper-triangular part of intersection matrix (row-major)

} // namespace neural_acd