#pragma once

#include <core.hpp>
#include <support_surface.hpp>
#include <vector>

namespace neural_acd {


MeshList process(Mesh mesh, double concavity, int num_parts,std::string stats_file);
double compute_final_concavity(MeshList &parts, MeshList &hulls);
std::vector<Surface> get_support_surfaces(Mesh &mesh);
Mesh preprocess_mesh(Mesh mesh);


} // namespace neural_acd