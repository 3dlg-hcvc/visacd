#pragma once

#include <core.hpp>
#include <vector>

namespace neural_acd {


struct ProcessResult {
  MeshList parts;
  double concavity;
  int num_parts;
};

ProcessResult process(Mesh mesh, double concavity, int num_parts);
double compute_final_concavity(MeshList &parts, MeshList &hulls);


} // namespace neural_acd