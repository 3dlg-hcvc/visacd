#pragma once

#include <core.hpp>
#include <vector>

namespace neural_acd {


MeshList process(Mesh mesh, double concavity, int num_parts,std::string stats_file);
double compute_final_concavity(MeshList &parts, MeshList &hulls);


} // namespace neural_acd