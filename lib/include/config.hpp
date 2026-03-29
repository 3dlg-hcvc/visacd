#pragma once

#include <string>

namespace neural_acd {
class Config {
public:
  int pcd_res;
  float remesh_res;
  float remesh_threshold;

  double cost_rv_k;

  bool return_parts;
  bool separate_disjoint;

  std::string score_mode; // "edge" or "concavity"

  double support_surface_min_area;
  bool use_support_surfaces;
  double support_surface_k;

  bool use_merging;

  Config() {
    pcd_res = 3000;

    remesh_res = 50.0f;
    remesh_threshold = 0.05f;

    cost_rv_k = 0.03;

    return_parts = false;
    separate_disjoint = true;

    score_mode = "edge";

    support_surface_min_area = 0.1;
    use_support_surfaces = true;
    support_surface_k = 2.0;
    use_merging = false;
  }
};

inline Config config;

} // namespace neural_acd