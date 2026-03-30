#pragma once

#include <string>

namespace neural_acd {
class Config {
public:
  bool return_parts;

  std::string score_mode; // "edge" or "concavity"

  double support_surface_min_area;
  bool use_support_surfaces;
  double support_surface_k;

  bool use_merging;

  Config() {
    return_parts = false;

    score_mode = "concavity";

    support_surface_min_area = 0.1;
    use_support_surfaces = true;
    support_surface_k = 2.0;
    use_merging = false;
  }
};

inline Config config;

} // namespace neural_acd