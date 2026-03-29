#pragma once

#include <core.hpp>

namespace neural_acd {
constexpr double Pi = 3.14159265;
double get_mesh_volume(Mesh &mesh);
double compute_rv(Mesh &cvx1, Mesh &cvx2, Mesh &cvxCH, double epsilon = 0.0001);
double compute_hb(Mesh &cvx1, Mesh &cvx2, Mesh &cvxCH, unsigned int resolution);
double compute_hb(Mesh &tmesh1, Mesh &tmesh2, unsigned int resolution,
                  bool flag = false);
double compute_h(Mesh &cvx1, Mesh &cvx2, Mesh &cvxCH, double k,
                 unsigned int resolution, double epsilon = 0.0001);
double compute_h(Mesh &tmesh1, Mesh &tmesh2, double k, unsigned int resolution,
                 double epsilon = 0.0001, bool flag = false);
double mesh_dist(Mesh &tmesh1, Mesh &tmesh2);

} // namespace neural_acd